import argparse
import torch
import os
import re
import json
import random
import warnings
import numpy as np
from tqdm import tqdm
import shortuuid

from transformers import StoppingCriteriaList

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.conversation_2 import conv_templates as conv_templates_2
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, KeywordsStoppingCriteria
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math

import open_flamingo
from open_flamingo.eval.eval_datasets import CaptionDataset as OpenflamingoCaptionDataset
from llava.eval.rices import RICES

from open_flamingo.eval.coco_metric import compute_cider


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def split_list_v2(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    base_chunk_size = len(lst) // n  # integer division
    remainder = len(lst) % n  # remaining elements
    chunks = []
    for i in range(n):
        chunk_size = base_chunk_size + (i < remainder)  # add one to the chunk size for the first 'remainder' chunks
        start = i * base_chunk_size + min(i, remainder)  # calculate the start index
        chunks.append(lst[start:start+chunk_size])
    return chunks


def get_chunk(lst, n, k):
    chunks = split_list_v2(lst, n)
    return chunks[k]


def set_random_seed(seed, deterministic: bool = False) -> int:
    print(f"Setting seed {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        if torch.backends.cudnn.benchmark:
            print(
                'torch.backends.cudnn.benchmark is going to be set as '
                '`False` to cause cuDNN to deterministically select an '
                'algorithm')
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # if digit_version() >= digit_version('1.10.0'):  # NOTE check manually
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return seed


# Custom dataset class
class CustomFewShotDataset(Dataset):
    def __init__(self, questions, conv_mode, image_folder, tokenizer, image_processor, model_config, 
                 num_shots=None, rices=None, sampled_query_set=None):
        self.questions = questions
        self.conv_mode = conv_mode
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        # >>> few-shot init from open_flamingo >>>
        self.num_shots = num_shots
        self.rices = rices
        if self.rices:
            self.rices_dataset = sampled_query_set
        else:
            self.query_set = sampled_query_set

    def __getitem__(self, index):
        line = self.questions[index]
        if args.conv_mode in conv_templates:
            conv = conv_templates[args.conv_mode].copy()
        else:
            conv = conv_templates_2[args.conv_mode].copy()
        
        image_file = line["filename"]
        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        
        # >>> few-shot sampling from open_flamingo >>>
        if self.rices:
            batch_demo_samples = self.rices_dataset.find([image], self.num_shots)
        else:
            batch_demo_samples = open_flamingo.eval.utils.sample_batch_demos_from_query_set(
                self.query_set, self.num_shots, len([image]))
        # <<< few-shot sampling from open_flamingo <<<
        
        # >>> few-shot from open_flamingo combine with image processing of llava >>>
        context_images = [x["image"].convert('RGB') for x in batch_demo_samples[0]] if self.num_shots > 0 else []
        batch_images = context_images + [image]
        image_tensor = process_images(batch_images, self.image_processor, self.model_config).unsqueeze(1)
        # <<< few-shot from open_flamingo combine with image processing of llava <<<
        
        # >>> few-shot from open_flamingo combine with conversation of llava >>>
        for x in batch_demo_samples[0]:
            _context_q = "<image>\n" + "\nProvide a one-sentence caption for the provided image."
            _context_a = x["caption"]
            conv.append_message(conv.roles[0], _context_q)
            conv.append_message(conv.roles[1], _context_a)
        # <<< few-shot from open_flamingo combine with conversation of llava <<<
        
        qs = "\nProvide a one-sentence caption for the provided image."
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        
        prompt = conv.get_prompt()
        
        if index <= 5:
            print(f"######## Sample {index} ########")
            print(prompt)
            print(f"######## Sample {index} ########")
        
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        return input_ids, image_tensor

    def __len__(self):
        return len(self.questions)


def create_data_loader(questions, conv_mode, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4, num_shots=None, rices=None, sampled_query_set=None):
    assert batch_size == 1, "batch_size must be 1"
    assert num_workers == 0, "num_workers must be 0"
    dataset = CustomFewShotDataset(questions, conv_mode, image_folder, tokenizer, image_processor, model_config, num_shots=num_shots, rices=rices, sampled_query_set=sampled_query_set)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


def eval_model(args):
    if args.distributed_launch:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    
    num_shots = args.shots
    
    if args.dataset_name == "coco":
        assert args.coco_train_image_dir_path is not None, "coco_train_image_dir_path must be provided for coco dataset"
        assert args.coco_val_image_dir_path is not None, "coco_val_image_dir_path must be provided for coco dataset"
        assert args.coco_karpathy_json_path is not None, "coco_karpathy_json_path must be provided for coco dataset"
        image_train_dir_path = args.coco_train_image_dir_path
        image_val_dir_path = args.coco_val_image_dir_path
        annotations_path = args.coco_karpathy_json_path
        cached_demonstration_features = None if args.cached_demonstration_features is None \
            else f"{args.cached_demonstration_features}/coco_rices/features/coco.pkl" 
    elif args.dataset_name == "flickr":
        assert args.flickr_image_dir_path is not None, "flickr_image_dir_path must be provided for flickr dataset"
        assert args.flickr_karpathy_json_path is not None, "flickr_karpathy_json_path must be provided for flickr dataset"
        image_train_dir_path = (
            args.flickr_image_dir_path
        )  # Note: calling this "train" for consistency with COCO but Flickr only has one split for images
        image_val_dir_path = None
        annotations_path = args.flickr_karpathy_json_path
        cached_demonstration_features = None if args.cached_demonstration_features is None \
            else f"{args.cached_demonstration_features}/flickr_rices/features/flickr30.pkl"
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")
    
    # load cached demonstration features for RICES
    if cached_demonstration_features is not None:
        cached_features = torch.load(cached_demonstration_features, map_location="cpu")
    else:
        cached_features = None
    
    train_dataset = OpenflamingoCaptionDataset(
        image_train_dir_path=image_train_dir_path,
        image_val_dir_path=image_val_dir_path,
        annotations_path=annotations_path,
        is_train=True,
        dataset_name=args.dataset_name if args.dataset_name != "nocaps" else "coco",
    )
    
    print(f"[Creating query_set...]")
    if args.rices:
        print(f"[Creating RICES...]")
        rices_dataset = RICES(
            train_dataset,
            device='cuda',
            batch_size=1,
            cached_features=cached_features,
            vision_encoder_path=args.rices_vision_encoder_path,
            vision_encoder_pretrained=args.rices_vision_encoder_pretrained,
        )
        sampled_query_set = rices_dataset
        if args.num_workers != 0:
            _may_be_err_msg = "RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method"
            warnings.warn(f"Set num_workers = 0, or you may raise:\n\t{_may_be_err_msg}")
            args.num_workers = 0
    else:
        query_set = open_flamingo.eval.utils.get_query_set(train_dataset, args.query_set_size)
        sampled_query_set = query_set
    
    set_random_seed(args.seed, args.deterministic)
    
    # Model
    print("[Loading Model...]")
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, device=args.device, llama_use_rope_scaling=args.shots >= 8)
    
    # Load questions
    print(f"[Loading questions...]")
    full_annotations = json.load(open(annotations_path))["images"]
    questions = get_chunk(
        [image for image in full_annotations if image['split'] == 'val'],
        args.num_chunks,
        args.chunk_idx
    )
    
    answers_file = os.path.expandvars(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    
    result_output = []
    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')
        
    print(f"[Creating DataLoader...]")
    data_loader = create_data_loader(questions, args.conv_mode, image_val_dir_path if image_val_dir_path is not None else image_train_dir_path,
                                     tokenizer, image_processor, model.config, num_workers=args.num_workers, 
                                     num_shots=num_shots, rices=args.rices, sampled_query_set=sampled_query_set)

    # prepare generating kwargs
    if hasattr(args, "generate_kwargs"):
        generate_kwargs = args.generate_kwargs
    else:
        generate_kwargs = dict(
            do_sample=True if args.temperature > 0 else False,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )
        if args.temperature > 0:
            generate_kwargs["temperature"] = args.temperature
        if args.top_p is not None:
            generate_kwargs["top_p"] = args.top_p

    # NOTE use interleaved prepare
    model.use_openflamingo_interleaved_preparing = True
    model.disable_tokenizer_max_length = True
    model.disable_max_num_images = True
    
    if args.distributed_launch:
        torch.distributed.barrier()
    
    for (input_ids, image_tensor), line in tqdm(zip(data_loader, questions), total=len(questions), desc=answers_file):
        idx = line["cocoid"] if "cocoid" in line else line["filename"].split('.')[0] # else flickr
        
        input_ids = input_ids.to(device=args.device, non_blocking=True)
        
        if "internlm2" in model_path:
            # add stopping word
            generate_kwargs["stopping_criteria"] = StoppingCriteriaList([
                KeywordsStoppingCriteria(['<|im_end|>', '<\s>'], tokenizer, input_ids)])

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device=args.device, non_blocking=True),
                **generate_kwargs)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        outputs = re.sub(r'\s*(<\\s>|<\\s|<\|im_end\|>|<\|im_end\|)$', '', outputs)

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"img_id": idx,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        # ans_file.flush()
        result_output.append({
            "image_id": idx,
            "caption": outputs
        })
    
    # save result_output to result_path (temporary file)
    if not args.distributed_launch:
        # NOTE that in distributed_launch mode, we re-build result_output from ans_file.
        result_file = os.path.expandvars(args.results_path)
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        with open(result_file, "w") as f:
            f.write(
                json.dumps(result_output, indent=4,)
            )
        
        metrics = compute_cider(
            result_path=args.results_path,
            annotations_path=args.coco_annotations_json_path
            if args.dataset_name == "coco"
            else args.flickr_annotations_json_path,
        )
        print('\nCIDEr =', metrics["CIDEr"] * 100.0)
        ans_file.write(json.dumps(metrics) + "\n")
        
        # remove temporary file
        os.remove(args.results_path)
        
    ans_file.close()
    
    if args.distributed_launch:
        torch.distributed.barrier()
    
    
def eval_distributed_launch(args):
    if args.dataset_name == "coco":
        annotations_path=args.coco_annotations_json_path
    elif args.dataset_name == "flickr":
        annotations_path=args.flickr_annotations_json_path
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")
    
    # compute CIDEr with the temporary result file
    num_chunks = args.eval_chunk_num
    folder = args.answers_file.replace(".jsonl", "")
    answers_file_list = [f"{folder}/{num_chunks}_{chunk_idx}.jsonl" for chunk_idx in range(num_chunks)]
    all_answers = []
    for chunk_answers_file in answers_file_list:
        with open(chunk_answers_file, "r") as fp:
            all_answers.extend([json.loads(line_str) for line_str in fp.readlines()])
            
    result_output = [{"image_id": ans["img_id"], "caption": ans["text"]} for ans in all_answers]
    print(f"Got {len(result_output)} results")
    
    result_file = os.path.expandvars(args.results_path)
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    with open(result_file, "w") as f:
        f.write(json.dumps(result_output, indent=4,))
    
    metrics = compute_cider(
        result_path=args.results_path,
        annotations_path=annotations_path,
    )
    print('\nCIDEr =', metrics["CIDEr"] * 100.0)
    
    answers_file = os.path.expandvars(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    with open(answers_file, "w") as ans_file:
        for ans in all_answers:
            ans_file.write(json.dumps(ans) + "\n")
        ans_file.write(json.dumps(metrics) + "\n")
    
    os.remove(args.results_path)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--answers_file", type=str, default="answer.jsonl")  # answer format of llava
    parser.add_argument("--results_path", type=str, default="tmp.json")  # result format for compute_cider
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    # Few-shot functions from OpenFlamingo
    parser.add_argument("--shots", default=0, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--deterministic", type=bool, default=False)
    parser.add_argument("--query_set_size", type=int, default=2048, help="Size of demonstration query set")
    parser.add_argument("--rices", type=bool, default=True)
    parser.add_argument("--rices_vision_encoder_path", default="ViT-L-14", type=str)
    parser.add_argument("--rices_vision_encoder_pretrained", default="openai", type=str)
    parser.add_argument("--cached_demonstration_features", default=None)
    parser.add_argument("--coco_train_image_dir_path", default=None, type=str)
    parser.add_argument("--coco_val_image_dir_path", default=None, type=str)
    parser.add_argument("--coco_karpathy_json_path", default=None, type=str)
    parser.add_argument("--coco_annotations_json_path", default=None, type=str)
    parser.add_argument("--flickr_image_dir_path", default=None, type=str)
    parser.add_argument("--flickr_karpathy_json_path", default=None, type=str)
    parser.add_argument("--flickr_annotations_json_path", default=None, type=str)
    parser.add_argument("--dataset_name", default=None, type=str)
    parser.add_argument("--distributed_launch", type=bool, default=False)
    parser.add_argument("--eval_distributed_launch", type=bool, default=False)  # NOTE require eval_chunk_num
    parser.add_argument("--eval_chunk_num", type=int, default=None)  # NOTE only for eval_distributed_launch
    parser.add_argument("--genargs-v1", type=bool, default=False)
        
    args = parser.parse_args()
    
    if args.model_base is None and "::" in args.model_path:
        model_base_and_path = args.model_path
        args.model_base, args.model_path = model_base_and_path.split("::")
        print(f"model_base_and_path ({model_base_and_path}) has been split into model_path ({args.model_path}) and model_base ({args.model_base})")

    if args.eval_distributed_launch:
        assert args.eval_chunk_num is not None, args.eval_chunk_num
        assert not args.distributed_launch
        eval_distributed_launch(args)
    else:
        if args.genargs_v1:
            args.generate_kwargs = dict(
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=1.0,
                num_beams=3,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                max_length=10,
                min_length=0, 
                length_penalty=0.,
            )
            print(f"Using genargs-v1. The generate_kwargs has been hardcoded with {args.generate_kwargs}")
            
        if args.distributed_launch:
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            args.num_chunks = world_size
            args.chunk_idx = rank
            folder = args.answers_file.replace(".jsonl", "")
            args.answers_file = f"{folder}/{args.num_chunks}_{args.chunk_idx}.jsonl"
            print(f"Rank {rank}/{world_size} is running chunk {args.chunk_idx}/{args.num_chunks}, "
                f"results will be saved to {args.answers_file}")
            # manage device
            device_id = rank % torch.cuda.device_count()
            args.device = f'cuda:{device_id}'
        else:
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        eval_model(args)
    