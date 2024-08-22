# coding=gb2312
"""
srun -p VC2 --job-name='conv_1' --gres=gpu:1 --cpus-per-task=10 --quotatype="auto" python -u translate_husky_batch.py --offset 15
"""

import os
import json

from tqdm import tqdm

import torch
from datasets import load_dataset

import deepspeed
from husky.conversation import get_conv_template
from transformers import (
    LlamaTokenizer,
    LlamaForCausalLM,
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)

from husky.compression import compress_module

import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"

conv_tempt = "husky_v2.0"
model_path = "/mnt/petrelfs/share_data/zhangqinglong/Husky/work_dirs/llm/husky-13b_v2_01"

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops, encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False

def build_transform(tokenizer):
    def transform(example):
        text = example["caption"]
        black_list = ["Sure, ", "Sure. ", "<image>\n", "Certainly, ", "Certainly. "]
        for black_word in black_list:
            text = text.replace(black_word, "").strip()
        text = f"Translate the following Chinese sentence into English sentence: \"{text}\""
        conversations = []
        conv = get_conv_template(conv_tempt)
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        conversations.append(conv.get_prompt())
        model_inputs = tokenizer(conversations)
        return dict(
            caption=conversations[0],
            input_ids=model_inputs['input_ids'][0],
            attention_mask=model_inputs["attention_mask"][0]
        )

    return transform

def get_gpu_memory(max_gpus=None):
    gpu_memory = []
    num_gpus = (
        torch.cuda.device_count()
        if max_gpus is None
        else min(max_gpus, torch.cuda.device_count())
    )

    for gpu_id in range(num_gpus):
        with torch.cuda.device(gpu_id):
            device = torch.cuda.current_device()
            gpu_properties = torch.cuda.get_device_properties(device)
            total_memory = gpu_properties.total_memory / (2048 ** 3)
            allocated_memory = torch.cuda.memory_allocated() / (2048 ** 3)
            available_memory = total_memory - allocated_memory
            gpu_memory.append(available_memory)
    return gpu_memory

def load_model(
        model_path, device, num_gpus, max_gpu_memory=None, load_8bit=False, debug=False
):
    if device == "cpu":
        kwargs = {}
    elif device == "cuda":
        kwargs = {"torch_dtype": torch.float16}
        if num_gpus == "auto":
            kwargs["device_map"] = "auto"
        else:
            num_gpus = int(num_gpus)
            if num_gpus != 1:
                kwargs["device_map"] = "auto"
                if max_gpu_memory is None:
                    kwargs[
                        "device_map"
                    ] = "sequential"  # This is important for not the same VRAM sizes
                    available_gpu_memory = get_gpu_memory(num_gpus)
                    kwargs["max_memory"] = {
                        i: str(int(available_gpu_memory[i] * 0.85)) + "GiB"
                        for i in range(num_gpus)
                    }
                else:
                    kwargs["max_memory"] = {i: max_gpu_memory for i in range(num_gpus)}
    else:
        raise ValueError(f"Invalid device: {device}")

    tokenizer = LlamaTokenizer.from_pretrained(
        model_path, use_fast=False)
    model = LlamaForCausalLM.from_pretrained(
        model_path, low_cpu_mem_usage=True, **kwargs
    )

    if load_8bit:
        compress_module(model, device)

    if (device == "cuda" and num_gpus == 1) or device == "mps":
        model.to(device)

    if debug:
        print(model)

    model = model.eval()
    return model, tokenizer


class TranslationTool:
    def __init__(self):
        world_size = int(os.getenv('WORLD_SIZE', '1'))
        model, tokenizer = load_model(model_path, device=device, num_gpus=world_size)
        tokenizer.padding_side = 'left'

        transform = build_transform(tokenizer)
        model = deepspeed.init_inference(
            model,
            mp_size=world_size,
            dtype=torch.float16,
            checkpoint=None,
            replace_with_kernel_inject=True,
            max_out_tokens=3200
        )

        stop_words = ["###", "Human: ", "Assistant: ", "<human>", "<bot>", "\n\n"]
        stop_words_ids = [tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze() for stop_word in stop_words]
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

        generation_config = GenerationConfig(
            pad_token_id=0,
            bos_token_id=1,
            do_sample=True,
            top_k=20,
            top_p=0.25,
            temperature=0.7,
            max_new_tokens=3200,
            stopping_criteria=stopping_criteria
        )
    
    # def translate

if __name__ == "__main__":
    
    import ipdb; ipdb.set_trace()
    
    # input_ids = torch.tensor(input_ids, device=device)
    # attention_mask = torch.tensor(attention_mask, device=device)
    
    # with torch.no_grad():
    #     generation_output = model.generate(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         generation_config=generation_config,
    #         return_dict_in_generate=True,
    #         output_scores=True
    #     )
    # preds = generation_output.sequences
    # preds = preds[:, input_ids.shape[1]:]
    # outputs = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # # outputs = [output[len(batch_data["caption"][i]) - 11:].strip() for i, output in
    # #            enumerate(outputs)]

    # for ind, out in enumerate(outputs):
    #     data = {key: batch_data[key][ind] for key in batch_data.keys()}
    #     # pop conversation
    #     data['caption'] = out
    #     json_line = json.dumps(data, ensure_ascii=False) + "\n"
    #     fw.write(json_line)
