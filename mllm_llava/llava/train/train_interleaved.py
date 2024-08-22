# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import io
import os
import copy
from dataclasses import dataclass, field
import time
import json
import base64
import logging
import pathlib
from einops import rearrange
from typing import Dict, Optional, Sequence, List

import torch
from torch.utils.data import Dataset, ConcatDataset

import transformers
import tokenizers

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.train.llava_trainer import LLaVAInterLeavedTrainer, SetSharedEpochCallback

from llava import conversation as conversation_lib
from llava import conversation_2 as conversation_lib_2
from llava.model import *
from llava.mm_utils import tokenizer_image_token

from PIL import Image


local_rank = None
rank = None
world_size = None


def rank0_print(*args):
    if rank == 0:
        print(*args)
        
        
from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    pixel_shuffle_ratio: Optional[float] = field(default=None)
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_use_end_of_chunk: bool = field(default=False)
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data.", "nargs": "+"})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None, metadata={"nargs": "+"})
    image_aspect_ratio: str = 'square'
    data_type: str = field(default="mmc4")
    mmc4_max_num_images: int = field(default=6)
    incontext_data_path: str = field(default=None,
                                     metadata={"help": "Path to the incontext training data.", 
                                               "nargs": "+"})
    incontext_image_path: Optional[str] = field(default=None, metadata={"nargs": "+"})
    sft_max_num_images: int = field(default=6)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_multi_images_length: bool = field(default=False)
    group_by_modality_length: bool = field(default=False)
    proxies: Optional[str] = field(default=None)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        # if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
        if trainer.args.process_index == 0 or trainer.args.process_index == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                num_imags = sentence['value'].count(DEFAULT_IMAGE_TOKEN)
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                for _ in range(num_imags):
                    sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "  # " " + "ASSISTANT" + ": " = " ASSISTANT: " 注意  ASSISTANT:  前后一定有空格
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX

        rounds = conversation.split(conv.sep2)  # "</s>" 标号单一个 2
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)  # " ASSISTANT: "  前面的表示 instruction 要 ignore，后面的表示 response，要保留
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)  # 其实是 -1 +1，-1 是bos token，+1 是 </s> 的 sep2，注意 </s> 前后都没有空格，因此没有空格的问题
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2  # 2 因为开头的 bos 和结尾冒号后面的空格 （如果空格后面有短语，就不会认为空格应当tokenize，而是当成分割符，但是如果空格后面没有，则会被解析成 29871）
                
            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
            
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
                
    DEBUG_CHECK = False
    if DEBUG_CHECK:
        for _input_ids, _target, _conversation in zip(input_ids, targets, conversations):
            _input_ids = _input_ids.clone()
            _target = _target.clone()
            tokenizer.add_special_tokens({"additional_special_tokens": ["<image>", "[IGNORE]"]})
            tokenized_image_id = tokenizer.additional_special_tokens_ids[
                tokenizer.additional_special_tokens.index("<image>")]
            ignored_image_id = tokenizer.additional_special_tokens_ids[
                tokenizer.additional_special_tokens.index("[IGNORE]")]
            _input_ids[_input_ids == IMAGE_TOKEN_INDEX] = tokenized_image_id
            _target[_target == IGNORE_INDEX] = ignored_image_id
            input_txt = tokenizer.decode(_input_ids)
            label_txt = tokenizer.decode(_target)
            print(f"META INPUT:\n{_conversation}\n\nDECODED INPUT:\n{input_txt}\n\nDECODED LABEL:\n{label_txt}")
            

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
        
    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            
            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1
                
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess_internlm2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    
    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]
        
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
        
    # Tokenize conversations
    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
        
    targets = input_ids.clone()
    
    assert conv.sep_style == conversation_lib_2.SeparatorStyle.INTERNLM2_CHAT
    
    for conversation, target in zip(conversations, targets):
        # debug_target = target.clone()  # int(debug_target.ne(tokenizer.pad_token_id).sum()) 
        total_len = int(target.ne(tokenizer.pad_token_id).sum())  # 浦语里面 pad_token_id == eos_token_id
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX  # <s>
        
        rounds = conversation.split("<|im_end|>\n<|im_start|>user\n")
        for i, rou in enumerate(rounds):
            # NOTE: round 划分要保证 <im_end> 后面的 \n 被 ignored
            if len(rounds) == 1:
                pass
            elif 0 < i < len(rounds) - 1:
                # rou = "<|im_start|>user\n" + rou + "<|im_end|>\n"
                rou = "\n<|im_start|>user\n" + rou + "<|im_end|>"
            elif i == 0:
                # rou = rou + "<|im_end|>\n"
                rou = rou + "<|im_end|>"
            elif i == len(rounds) - 1:
                # rou = "<|im_start|>user\n" + rou
                rou = "\n<|im_start|>user\n" + rou
            else:
                raise ValueError(f"i={i}, len(rounds)={len(rounds)}")
            
            sep = "<|im_end|>\n<|im_start|>assistant\n"
            parts = rou.split(sep)  # 前面的表示 instruction 要 ignore，后面的表示 response，要保留
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer)) - 1
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids) - 1  # 开头的 bos
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1  # 开头的 bos

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        
        target[cur_len:] = IGNORE_INDEX
        
        if conversation.endswith("<|im_end|>\n"):
            # NOTE 确保最后一个 <im_end> 后面的 \n 会被 ignored
            target[-1:] = IGNORE_INDEX
            
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                # import ipdb; ipdb.set_trace()
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}.")
                target[:] = IGNORE_INDEX
                
    # import ipdb; ipdb.set_trace()
    # print(f"conversations:")
    # print([c.replace('\n', '\\n') for c in conversations])
    DEBUG_CHECK = False
    if DEBUG_CHECK:
        for _input_ids, _target, _conversation in zip(input_ids, targets, conversations):
            _input_ids = _input_ids.clone()
            _target = _target.clone()
            tokenizer.add_special_tokens({"additional_special_tokens": ["<image>", "[IGNORE]"]})
            tokenized_image_id = tokenizer.additional_special_tokens_ids[
                tokenizer.additional_special_tokens.index("<image>")]
            ignored_image_id = tokenizer.additional_special_tokens_ids[
                tokenizer.additional_special_tokens.index("[IGNORE]")]
            _input_ids[_input_ids == IMAGE_TOKEN_INDEX] = tokenized_image_id
            _target[_target == IGNORE_INDEX] = ignored_image_id
            input_txt = tokenizer.decode(_input_ids).replace("\n", "\\n")
            label_txt = tokenizer.decode(_target).replace("\n", "\\n")
            _conversation = _conversation.replace("\n", "\\n")
            print(f"META INPUT:\n{_conversation}\n\nDECODED INPUT:\n{input_txt}\n\nDECODED LABEL:\n{label_txt}")
            
    
    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "internlm2":
        return preprocess_internlm2(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
                 data_path: str,
                 image_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        self.data_path = data_path
        self.image_path = image_path
        list_data_dict = self.load_list_data_dict(data_path)

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.max_num_images = getattr(data_args, "sft_max_num_images", 6)

    def __len__(self):
        return len(self.list_data_dict)
    
    @staticmethod
    def load_list_data_dict(data_path):
        if data_path.endswith("json"):
            return json.load(open(data_path, "r"))
        elif data_path.endswith("jsonl"):
            return [json.loads(line) for line in open(data_path, "r").readlines() if line.strip()]
        else:
            raise ValueError(f"Unsupport data format: {data_path}")

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 576 if 'image' in sample else 0  # 128 originally
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list
    
    @property
    def multi_images_lengths(self):
        # The LazySupervisedDataset only support single image.
        img_len = 576
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len + img_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list
    
    def pad_to_max_num_img(self, all_images):
        # T, c, h, w
        assert len(all_images) <= self.max_num_images, (len(all_images), self.max_num_images)
        h, w = all_images.shape[2:]
        if len(all_images) < self.max_num_images:
            pad_num = self.max_num_images - len(all_images)
            zero_padding = torch.zeros((pad_num, 3, h, w), dtype=torch.float)
            all_images = torch.cat((all_images, zero_padding), dim=0)
        return all_images
    
    def load_image(self, image):
        try:
            image_fname = image
            image_fpath = os.path.join(self.image_path, image_fname)
            if "s3://" in self.image_path:
                if not hasattr(self, "ceph_client"):
                    from petrel_client import Client
                    self.ceph_client = Client("~/ceph_config.conf")
                image = Image.open(io.BytesIO(self.ceph_client.get(image_fpath))).convert('RGB')
            else:
                image = Image.open(image_fpath).convert('RGB')
        except Exception as e:
            raise Exception(f"\nimage path {self.image_path}\ndata_path {self.data_path}") from e
        return image

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            processor = self.data_args.image_processor
            image = self.load_image(self.list_data_dict[i]['image'])
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            # data_dict['image'] = image
            data_dict['image'] = self.pad_to_max_num_img(image.unsqueeze(0)).unsqueeze(1)
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            # data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
            data_dict['image'] = torch.zeros(6, 1, 3, crop_size['height'], crop_size['width'])
        return data_dict
    
   
class LazySupervisedDatasetWIncontext(LazySupervisedDataset):
    """Dataset for supervised fine-tuning with in-context examples."""
    def __init__(self, 
                 data_path: str,
                 image_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super().__init__(data_path=data_path, image_path=image_path, tokenizer=tokenizer, data_args=data_args)
        if image_path.endswith(".json"):
            self.incontext_image_path_fmt = "json"
            rank0_print(f"Loading images of in-context samples from {image_path}")
            _start_time = time.time()
            self.images = json.load(open(image_path, "r"))
            rank0_print(f"Successfully Loaded images of in-context samples from {image_path}, which takes {time.time() - _start_time}s")
        else:
            self.incontext_image_path_fmt = "folder"
            self.incontext_image_folder = self.image_path
    
    @property
    def multi_images_lengths(self):
        # The LazySupervisedDataset only support single image.
        img_len = 576
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            
            num_images = 0
            if 'incontext_images' in sample:
                incontext_images = sample['incontext_images']
                assert isinstance(incontext_images, Sequence) and not isinstance(incontext_images, str), "incontext_images should be `tuple` or `list` of images"
                num_images += len(incontext_images)
            if not isinstance(sample['image'], str):  # NOTE we allow both list of multiple images and single image str in this implementation
                num_images += len(sample['image'])
            else:
                num_images += 1
            
            cur_len = cur_len + img_len * num_images if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list
            
    def preprocess_image(self, image):
        processor = self.data_args.image_processor
        if self.data_args.image_aspect_ratio == 'pad':
            image = self.expand2square(image, tuple(int(x*255) for x in processor.image_mean))
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        else:
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        return image
            
    @staticmethod
    def expand2square(pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result
        
    def load_image(self, image):
        if self.incontext_image_path_fmt == "json":
            image_id = image
            base64_image = self.images[image_id]
            image_bytes = base64.b64decode(base64_image)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')  # may raise ValueError: Decompressed Data Too Large
        elif self.incontext_image_path_fmt == "folder":
            image_fname = image
            image_fpath = os.path.join(self.incontext_image_folder, image_fname)
            if "s3://" in self.image_path:
                if not hasattr(self, "ceph_client"):
                    from petrel_client import Client
                    self.ceph_client = Client("~/ceph_config.conf")
                image = Image.open(io.BytesIO(self.ceph_client.get(image_fpath))).convert('RGB')
            else:
                image = Image.open(image_fpath).convert('RGB')
        else:
            raise ValueError(f"Unsupport incontext_image_path_fmt: {self.incontext_image_path_fmt}")
        return image
    
    def load_and_preprocess_imgs(self, images):
        processed_imgs = []
        for img in images:
            try:
                processed_imgs.append(self.preprocess_image(self.load_image(img)))
            except Exception as e:
                raise Exception(f"\ndata_path: {self.data_path}\nimage_path: {self.image_path}\nimage: {img}") from e
        return processed_imgs
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
            
        if isinstance(i, int):
            sources = [sources]  # i.e. [self.list_data_dict[i]]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        
        # NOTE the samples with incontext_xxx must have "image" in dict
        if 'image' in sources[0]:  # source[0] is self.list_data_dict[i]
            all_images = []
            if 'incontext_images' in self.list_data_dict[i]:
                incontext_images = self.list_data_dict[i]['incontext_images']
                assert isinstance(incontext_images, Sequence) and not isinstance(incontext_images, string), "incontext_images should be `tuple` or `list` of images"
                all_images.extend(incontext_images)
            if not isinstance(self.list_data_dict[i]['image'], str):  # NOTE we allow both list of multiple images and single image str in this implementation
                all_images.extend(self.list_data_dict[i]['image'])
            else:
                all_images.append(self.list_data_dict[i]['image'])
            all_images = self.load_and_preprocess_imgs(all_images)
            all_images = torch.stack(all_images, dim=0)  # T, c, h, w
            # >>> Check num images >>>
            num_images_1 = len(all_images)
            # <<< Check num images <<<
            all_images = self.pad_to_max_num_img(all_images).unsqueeze(1)  # T, F, c, h, w (T should be 6, F=1)

            if len(sources[0].get("incontext_conversations", [])) > 0:
                sources = copy.deepcopy(sources)
                incontext_conversations = sources[0].pop("incontext_conversations")
                conversations_without_incontext = sources[0].pop("conversations")
                sources[0]["conversations"] = incontext_conversations + conversations_without_incontext
                sources[0]["conversations_without_incontext"] = conversations_without_incontext
            
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=True)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])
            # >>> Check num images >>>
            if 'image' in self.list_data_dict[i]:
                input_ids = data_dict["input_ids"]
                num_images_2 = sum(input_ids == IMAGE_TOKEN_INDEX)
                assert num_images_1 == num_images_2, (num_images_1, num_images_2, i)
            # <<< Check num images <<<

        if 'image' in self.list_data_dict[i]:
            # image exist in the data
            data_dict['image'] = all_images
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(6, 1, 3, crop_size['height'], crop_size['width'])
        return data_dict
    
    
class LazySupervisedConcatDataset(ConcatDataset):
    @property
    def lengths(self):
        length_list = []
        for dataset in self.datasets:
            length_list.extend(dataset.lengths)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for dataset in self.datasets:
            length_list.extend(dataset.modality_lengths)
        return length_list
    
    @property
    def multi_images_lengths(self):
        length_list = []
        for dataset in self.datasets:
            length_list.extend(dataset.multi_images_lengths)
        return length_list


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch


@dataclass
class DataCollatorOpenflamingo(object):
    """Collate examples for MMC4 dataset."""

    # tokenizer: transformers.PreTrainedTokenizer
    pad_token_id: int
    media_token_id: int
    endofchunk_token_id: Optional[int]
    
    def __call__(self, openflamingo_batch_inputs):
        # TODO: add these to collate fn
        img_inputs, txt_inputs = openflamingo_batch_inputs[0], openflamingo_batch_inputs[1]
        #### MMC4 FORWARD PASS ####
        # images = img_inputs.to(device_id, dtype=cast_dtype, non_blocking=True)
        images = img_inputs
        images = rearrange(images, "b (t f) c h w -> b t f c h w", f=1)
        input_ids = torch.stack([x[0] for x in txt_inputs]).squeeze(1)
        attention_mask = torch.stack([x[1] for x in txt_inputs]).squeeze(1)

        # set up labels; language model is expected to handle shifting
        labels = input_ids.clone()
        labels[labels == self.pad_token_id] = -100
        for i in range(labels.shape[0]):
            # remove loss for any token before the first <image> token
            label_idx = 0
            while (
                label_idx < labels.shape[1] and labels[i][label_idx] != self.media_token_id
            ):
                labels[i][label_idx] = -100
                label_idx += 1

            if self.endofchunk_token_id is not None:
                # get index of all endofchunk tokens in the sequence
                endofchunk_idxs = torch.where(labels[i] == self.endofchunk_token_id)[0]
                for endofchunk_idx in endofchunk_idxs:
                    token_idx = endofchunk_idx + 1
                    while (
                        token_idx < labels.shape[1]
                        and labels[i][token_idx] != self.media_token_id
                    ):
                        labels[i][token_idx] = -100
                        token_idx += 1

        labels[labels == self.media_token_id] = -100
        # labels = labels.to(device_id)
        
        batch = dict(
            images=images, 
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )
        return batch
    
    
class DataCollatorForInterleaved(DataCollatorOpenflamingo):
    
    def __call__(self, batch_inputs):
        img_inputs, txt_inputs = [], []
        for sample in batch_inputs:
            img_inputs.append(sample[0])
            txt_inputs.append(sample[1])
        img_inputs = torch.stack(img_inputs)
        return super(DataCollatorForInterleaved, self).__call__((img_inputs, txt_inputs))


def make_mmc4_format_data_module(tokenizer, data_args):
    assert len(data_args.data_path) == 1, data_args.data_path
    assert len(data_args.image_folder) == 1, data_args.image_folder
    data_args.data_path = data_args.data_path[0]
    data_args.image_folder = data_args.image_folder[0]
    data_path = data_args.data_path
    img_path = data_args.image_folder
    
    from pretrain_llava.llava.train.datasets import MMC4FormatDataset
    print("Using MMC4FormatDataset")
    train_dataset = MMC4FormatDataset(data_path, img_path, tokenizer, data_args)
        
    data_collator = DataCollatorForInterleaved(
        pad_token_id=tokenizer.pad_token_id, 
        media_token_id=IMAGE_TOKEN_INDEX, 
        endofchunk_token_id=None)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def make_obelisc_format_data_module(tokenizer, data_args):
    assert len(data_args.data_path) == 1, data_args.data_path
    assert len(data_args.image_folder) == 1, data_args.image_folder
    data_args.data_path = data_args.data_path[0]
    data_args.image_folder = data_args.image_folder[0]
    data_path = data_args.data_path
    img_path = data_args.image_folder
    
    from pretrain_llava.llava.train.datasets import ObelicsFormatDataset
    train_dataset = ObelicsFormatDataset(data_path, img_path, tokenizer, data_args)
    data_collator = DataCollatorForInterleaved(
        pad_token_id=tokenizer.pad_token_id, 
        media_token_id=IMAGE_TOKEN_INDEX, 
        endofchunk_token_id=None)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args, return_redundancy: bool = False) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    assert len(data_args.data_path) == len(data_args.image_folder), (data_args.data_path, data_args.image_folder)  # because of nargs="+"
    if "mmc4" in data_args.data_type.lower():
        return make_mmc4_format_data_module(tokenizer, data_args)
    elif "obelisc" in data_args.data_type.lower():
        return make_obelisc_format_data_module(tokenizer, data_args)
    elif data_args.data_type.lower() == "incontext_sft":
        outputs = {"eval_dataset": None}
        train_datasets = []
        for data_path, image_folder in zip(data_args.data_path, data_args.image_folder):
            train_datasets.append(
                LazySupervisedDataset(tokenizer=tokenizer, 
                                      data_path=data_path, 
                                      image_path=image_folder, 
                                      data_args=data_args)
                )
        if data_args.incontext_data_path is not None:
            assert len(data_args.incontext_data_path) == len(data_args.incontext_image_path), (data_args.incontext_data_path, data_args.incontext_image_path)  # because of nargs="+"
            for data_path, image_path in zip(data_args.incontext_data_path, data_args.incontext_image_path):
                train_datasets.append(
                    LazySupervisedDatasetWIncontext(tokenizer=tokenizer, 
                                                    data_path=data_path, 
                                                    image_path=image_path, 
                                                    data_args=data_args)
                )
        all_data_path = [ds.data_path for ds in train_datasets]
        if len(train_datasets) == 1:
            print(f"The dataset is {train_datasets[0].__class__} of: all_data_path={all_data_path}")
            train_dataset = train_datasets[0]
        else:
            print(f"The dataset is concatdataset of {len(train_datasets)} datasets: all_data_path={all_data_path}")
            train_dataset = LazySupervisedConcatDataset(train_datasets)
        outputs["train_dataset"] = train_dataset
        if return_redundancy:
            outputs["train_datasets"] = train_datasets
        outputs["data_collator"] = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
        return outputs
    else: 
        raise NotImplementedError(data_args.data_type)
    
    
def print_trainable_parameters(model):
    def _get_num_parameters(parameters: list):
        """Modified from print_trainable_parameters of peft"""
        trainable_params = 0
        all_param = 0
        for param in parameters:
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel
            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        return trainable_params, all_param
    
    trainable_params, all_param = _get_num_parameters(model.parameters())
    rank0_print(f"[WHOLE_MODEL] trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}")
    
    if isinstance(model, LlavaLlamaForCausalLM) or isinstance(model, LlavaInternLM2ForCausalLM):
        for part_name, part_module in [
            ("VISION TOWER", model.get_vision_tower()), 
            ("MULTIMODAL PROJ", model.get_model().mm_projector), 
            ("LANGUAGE LAYERS", model.get_model().layers)
        ]:
            trainable_params, all_param = _get_num_parameters(part_module.parameters())
            rank0_print(f"[{part_name}] trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}")


def check_pretrained_load(model, model_args):
    model_name_or_path = model_args.model_name_or_path
    archive_file = os.path.join(model_name_or_path, "pytorch_model.bin.index.json")
    if not os.path.isfile(archive_file):
        rank0_print(f"[PRETRAINED LOADING CHECK] archive_file {archive_file} does not exist, skip checking. (pretrained path may be a hub url, whose checking is not implemented for this llava-based codebase)")
        return
    
    import gc
    from transformers.modeling_utils import load_state_dict as load_state_dict_from_checkpoint_file
    from transformers.utils.hub import get_checkpoint_shard_files
    from transformers.deepspeed import is_deepspeed_zero3_enabled
    
    resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(
        model_name_or_path,
        archive_file,
        resume_download=False,
        local_files_only=False,
        user_agent={'file_type': 'model', 'framework': 'pytorch', 'from_auto_class': False},
        revision='main',
        subfolder='',
    )
    
    if is_deepspeed_zero3_enabled():
        import deepspeed
        with deepspeed.zero.GatheredParameters(model.parameters(), modifier_rank=0):
            model_state_dict = {k:v.cpu() for k, v in model.state_dict().items()}
    else:
        model_state_dict = {k:v.cpu() for k, v in model.state_dict().items()}
    
    loaded_state_dict = {}
    for shard_file in resolved_archive_file:
        loaded_state_dict.update(load_state_dict_from_checkpoint_file(shard_file))
    
    missing_keys = [key for key in model_state_dict if key not in loaded_state_dict]
    unexpected_keys = [key for key in loaded_state_dict if key not in model_state_dict]
    mismatched_keys = []

    for key in model_state_dict.keys():
        if key in loaded_state_dict:
            model_p = model_state_dict[key]
            loaded_p = loaded_state_dict[key]
            if model_p.dtype != loaded_p.dtype:
                loaded_p = loaded_p.to(model_p.dtype)
            if model_p.device != loaded_p.device:
                loaded_p = loaded_p.to(device=model_p.device)
            if not torch.allclose(model_p, loaded_p):
                mismatched_keys.append(key)
                
    if not missing_keys and not unexpected_keys and not mismatched_keys:
        rank0_print("[PRETRAINED LOADING CHECK] All pretrained parameters have been successfully loaded into the model.")
    else:
        if missing_keys:
            rank0_print(f"[PRETRAINED LOADING CHECK] The following parameters are missing in the pretrained state dict and could not be loaded: {missing_keys}")
        if unexpected_keys:
            rank0_print(f"[PRETRAINED LOADING CHECK] The following pretrained parameters were not found in the model and are considered extra: {unexpected_keys}")
        if mismatched_keys:
            rank0_print(f"[PRETRAINED LOADING CHECK] The following parameters could not be correctly loaded due to not allclose: {mismatched_keys}")
    
    # force memory release
    del model_state_dict
    del loaded_state_dict
    gc.collect()
    

def train(attn_implementation=None):  
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    global local_rank
    local_rank = training_args.local_rank
    global rank
    rank = training_args.process_index
    global world_size
    world_size = training_args.world_size
    # print(f"##### local_rank={local_rank}")
    # print(training_args)
    # print(f"##### local_rank={local_rank} rank={training_args.process_index} world_size={training_args.world_size}")
    
    if 'internlm2' in model_args.model_name_or_path:
        data_args.internlm2_chat_style = True  # used in MMC4FormatDataset
    
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    if model_args.vision_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(
                model_args.model_name_or_path, 
                trust_remote_code=True, 
                proxies=json.loads(training_args.proxies), 
            )
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = LlavaMptForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                proxies=json.loads(training_args.proxies), 
                **bnb_model_from_pretrained_args
            )
        elif 'internlm2' in model_args.model_name_or_path:
            revision = "5b50661e5ba16c9ded1047a51e394280b3b9bda1" if "7b" in model_args.model_name_or_path.lower() else "main"
            config = LlavaInternLM2Config.from_pretrained(
                model_args.model_name_or_path, 
                proxies=json.loads(training_args.proxies), 
                attn_implementation="flash_attention_2", 
                revision=revision, 
            )
            model = LlavaInternLM2ForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir, 
                revision=revision,
                proxies=json.loads(training_args.proxies), 
                **bnb_model_from_pretrained_args
            )
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                proxies=json.loads(training_args.proxies), 
                **bnb_model_from_pretrained_args
            )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            proxies=json.loads(training_args.proxies), 
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False
    
    # set interleaved flag for model
    model.use_openflamingo_interleaved_preparing = True
    model.config.use_openflamingo_interleaved_preparing = True

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        # gradient_checkpoint 本身在 trainer 里调用了 model.gradient_checkpointing_enable() 开启
        # 所以在 trainer 初始化后，可通过 model.get_model().gradient_checkpointing 查看是否开启
        if hasattr(model, "enable_input_require_grads"):
            # 这里主要是配合 peft 和 gradient_checkpointing 一起使用的情况
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right", 
            proxies=json.loads(training_args.proxies), 
        )
    elif "internlm2" in model_args.model_name_or_path:
        from llava.model.language_model.internlm_chat.tokenization_internlm2 import InternLM2Tokenizer
        tokenizer = InternLM2Tokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right", 
            use_fast=False, 
            revision="5b50661e5ba16c9ded1047a51e394280b3b9bda1" if "7b" in model_args.model_name_or_path.lower() else "main",
            proxies=json.loads(training_args.proxies), 
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
            proxies=json.loads(training_args.proxies), 
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        elif model_args.version in conversation_lib_2.conv_templates:
            conversation_lib.default_conversation = conversation_lib_2.conv_templates[model_args.version]
            conversation_lib.default_conversation.version = model_args.version
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
            
    # copied from mmc4 support in OpenFlamingo FIXME
    # NOTE, the additional_special_tokens <image> and <|endofchunk|> are only used for tokenize conveniently, hence, the input_embeddings of the model is not resized.
    model.config.mm_use_end_of_chunk = data_args.mm_use_end_of_chunk = training_args.mm_use_end_of_chunk = model_args.mm_use_end_of_chunk
    if model_args.mm_use_end_of_chunk:
        tokenizer.add_special_tokens({"additional_special_tokens": ["<|endofchunk|>", "<image>"]})  # add Flamingo special tokens to the tokenizer
    else:
        tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_args.per_device_train_batch_size = training_args.per_device_train_batch_size
    data_args.dataloader_num_workers = training_args.dataloader_num_workers
    
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    rank0_print("data modules ready")
    
    train_with_mmc4 = (
        ("mmc4" in data_args.data_type.lower() and "new" not in data_args.data_type.lower())
        or "laion" in data_args.data_type.lower() 
        or ("obelisc_format_lmmcc" not in data_args.data_type.lower() and "lmmcc" in data_args.data_type.lower())
    )
    if data_args.data_type.lower() == "incontext_sft":
        # Otherwise, the Image.open() may raise "ValueError: Decompressed Data Too Large"
        from PIL import PngImagePlugin
        MaximumDecompressedSize = 1024
        MegaByte = 2 ** 20
        PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte
    callbacks = [SetSharedEpochCallback] if train_with_mmc4 else []
    # import ipdb; ipdb.set_trace()
    if (
        ("mmc4" in data_args.data_type.lower() and "new" in data_args.data_type.lower())
        or ("obelisc" in data_args.data_type.lower())
    ):
        # import ipdb; ipdb.set_trace()
        training_args.use_sequential_sampler = True
    
    trainer = LLaVAInterLeavedTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=callbacks, 
        **data_module, 
    )
    trainer.train_with_mmc4 = train_with_mmc4  # will work in get_train_dataloader when trainer.train()
    
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    else:
        print_trainable_parameters(model)
        
    # # check loaded
    # check_pretrained_load(model, model_args)
    
    rank0_print("start training")

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        # if training_args.local_rank == 0 or training_args.local_rank == -1:
        if training_args.process_index == 0 or training_args.process_index == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
    