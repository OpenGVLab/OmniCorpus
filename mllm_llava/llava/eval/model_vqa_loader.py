import argparse
import torch
import os
import re
import json
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


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        if args.conv_mode in conv_templates:
            conv = conv_templates[args.conv_mode].copy()
        else:
            conv = conv_templates_2[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        if index <= 5:
            print(f"######## Sample {index} ########")
            print(prompt)
            print(f"######## Sample {index} ########")

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def eval_model(args):
    if args.distributed_launch:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, device=args.device)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)
    
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
            
    if args.distributed_launch:
        torch.distributed.barrier()

    for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions), desc=answers_file):
        idx = line["question_id"]
        cur_prompt = line["text"]

        input_ids = input_ids.to(device=args.device, non_blocking=True)
        
        if "internlm2" in model_path:
            # add stopping word
            generate_kwargs["stopping_criteria"] = StoppingCriteriaList([
                KeywordsStoppingCriteria(['<|im_end|>', '<\s>'], tokenizer, input_ids)])

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device=args.device, non_blocking=True),
                image_sizes=image_sizes,
                **generate_kwargs)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        outputs = re.sub(r'\s*(<\\s>|<\\s|<\|im_end\|>|<\|im_end\|)$', '', outputs)

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        # ans_file.flush()
    ans_file.close()
    
    if args.distributed_launch:
        torch.distributed.barrier()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--distributed_launch", type=bool, default=False)
    parser.add_argument("--genargs-v1", type=bool, default=False)
    parser.add_argument("--use-postprocess", type=bool, default=False)
    args = parser.parse_args()
    
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
    
    if args.model_base is None and "::" in args.model_path:
        model_base_and_path = args.model_path
        args.model_base, args.model_path = model_base_and_path.split("::")
        print(f"model_base_and_path ({model_base_and_path}) has been split into model_path ({args.model_path}) and model_base ({args.model_base})")

    eval_model(args)
