import argparse
import os
import random
import json
from tqdm import tqdm
from copy import deepcopy
from collections import Counter

from PIL import Image
import torch
# pip install open-flamingo
import open_flamingo
from open_flamingo.eval.eval_datasets import VQADataset as _OpenflamingoVQADataset
from llava.eval.rices import RICES, CustomRICES


# Custom dataset class
class FewShotSampler:
    def __init__(self, image_folder, num_shots=None, rices=None, sampled_query_set=None):
        self.image_folder = image_folder
        # >>> few-shot init from open_flamingo >>>
        self.num_shots = num_shots
        self.rices = rices
        if self.rices:
            self.rices_dataset = sampled_query_set
        else:
            self.query_set = sampled_query_set

    def get_fewshot_samples(self, image_file):
        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        # >>> few-shot sampling from open_flamingo >>>
        if self.rices:
            batch_demo_samples = self.rices_dataset.find([image], self.num_shots)
        else:
            batch_demo_samples = open_flamingo.eval.utils.sample_batch_demos_from_query_set(
                self.query_set, self.num_shots, len([image]))
        # <<< few-shot sampling from open_flamingo <<<
        return batch_demo_samples[0]

    def __len__(self):
        return len(self.questions)
    
    
class OpenflamingoVQADataset(_OpenflamingoVQADataset):
    def __getitem__(self, idx):
        question = self.questions[idx]
        answers = self.answers[idx]
        return question, answers
    

def get_sampler(num_shots=5, ):
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-folder", type=str, default="./playground/data/eval/okvqa/train2014")
    parser.add_argument("--question-file", type=str, default="./playground/data/eval/okvqa/llava_okvqa_mscoco_val2014.jsonl")
    parser.add_argument("--query_set_size", type=int, default=2048)
    parser.add_argument("--rices", type=bool, default=True)
    parser.add_argument("--rices_vision_encoder_path", default="ViT-L-14", type=str)
    parser.add_argument("--rices_vision_encoder_pretrained", default="openai", type=str)
    parser.add_argument("--cached_demonstration_features", default="/mnt/petrelfs/liqingyun/open_flamingo/open_flamingo/scripts/okvqa_rices/features")
    parser.add_argument("--image_dir_path", type=str, default="/mnt/petrelfs/share_data/chenzhe1/data/coco/train2014")
    parser.add_argument("--question_path", type=str, default="/mnt/petrelfs/share_data/wangweiyun/datasets/OK-VQA/OpenEnded_mscoco_train2014_questions.json")
    parser.add_argument("--annotations_path", type=str, default="/mnt/petrelfs/share_data/wangweiyun/datasets/OK-VQA/mscoco_train2014_annotations.json")
    parser.add_argument("--dataset_name", default="ok_vqa", type=str)
    args = parser.parse_args()
    # >>> few-shot init from open_flamingo >>>
    seed = 42
    # load cached demonstration features for RICES
    if args.cached_demonstration_features is not None:
        cached_features = torch.load(
            f"{args.cached_demonstration_features}/{args.dataset_name}.pkl", map_location="cpu"
        )
    else:
        cached_features = None
    
    train_dataset = OpenflamingoVQADataset(
        image_dir_path=args.image_dir_path,
        question_path=args.question_path,
        annotations_path=args.annotations_path,
        is_train=True,
        dataset_name=args.dataset_name,
    )
    if args.rices:
        rices_dataset = RICES(
            train_dataset,
            device='cuda',
            batch_size=1,  # useless if cached_features is provided
            cached_features=cached_features,
            vision_encoder_path=args.rices_vision_encoder_path,
            vision_encoder_pretrained=args.rices_vision_encoder_pretrained,
        )
        sampled_query_set = rices_dataset
    else:
        query_set = open_flamingo.eval.utils.get_query_set(train_dataset, args.query_set_size)
        sampled_query_set = query_set
    
    open_flamingo.eval.utils.random_seed(seed, getattr(args, "rank", 0))
    few_shot_sampler = FewShotSampler(args.image_folder, num_shots=num_shots, rices=args.rices, sampled_query_set=sampled_query_set)
    # <<< few-shot init from open_flamingo <<<
    return few_shot_sampler
        
        
def test_okvqa_training_samples():
    llava_data_path = "playground/data/llava_v1_5_mix665k.json"
    okvqa_root = "/mnt/petrelfs/share_data/wangweiyun/datasets/OK-VQA/"
    annotations_path = f"{okvqa_root}/mscoco_train2014_annotations.json"
    questions_path = f"{okvqa_root}/OpenEnded_mscoco_train2014_questions.json"
    
    llava_data = json.load(open(llava_data_path, "r"))
    filtered_llava_data = [d for d in llava_data if len(d["conversations"][1]["value"].split()) < 3]
    for _d in filtered_llava_data[:5]:
        print(_d)
    
    annotations = json.load(open(annotations_path, "r"))["annotations"]
    questions = json.load(open(questions_path, "r"))["questions"]
    print(f"len(annotations)={len(annotations)}")
    print(f"len(questions)={len(questions)}")
    
    for ann, que in zip(annotations, questions):
        assert que["image_id"] == ann["image_id"]
        assert que["question_id"] == ann["question_id"]
        
        
def test_aokvqa_training_samples():
    llava_data_path = "playground/data/llava_v1_5_mix665k.json"
    aokvqa_root = "/mnt/petrelfs/share_data/wangweiyun/datasets/A-OK-VQA/"
    annotations_path = f"{aokvqa_root}/aokvqa_v1p0_train.json"
    
    annotations = json.load(open(annotations_path, "r"))  # List[Dict], with keys: 'split', 'image_id', 'question_id', 'question', 'choices', 'correct_choice_idx', 'direct_answers', 'difficult_direct_answer', 'rationales'
    print(f"len(annotations)={len(annotations)}")
    
    # try filtering out aokvqa samples in llava-mix665k with tail prompt
    llava_data = json.load(open(llava_data_path, "r"))
    LLAVA_AOKVQA_TAIL_PROMPT = "\nAnswer with the option's letter from the given choices directly."
    filtered_samples = [d for d in llava_data if d["conversations"][0]["value"].endswith(LLAVA_AOKVQA_TAIL_PROMPT)]
    
    
def test_4vqa_training_samples():
    llava_data_path = "playground/data/llava_v1_5_mix665k.json"
    
    # try filtering out aokvqa samples in llava-mix665k with tail prompt
    llava_data = json.load(open(llava_data_path, "r"))
    LLAVA_4VQA_TAIL_PROMPT = "\nAnswer the question using a single word or phrase."
    filtered_samples = [d for d in llava_data if d["conversations"][0]["value"].endswith(LLAVA_4VQA_TAIL_PROMPT)]
    
    imgid2qalist = {}
    for sample in filtered_samples:
        if sample["id"] not in imgid2qalist:
            imgid2qalist[sample["id"]] = []
        imgid2qalist[sample["id"]].append(sample)
        
    imgpath2qalist = {}
    for sample in filtered_samples:
        if sample["image"] not in imgpath2qalist:
            imgpath2qalist[sample["image"]] = []
        imgpath2qalist[sample["image"]].append(sample)
        
    for qalist in imgid2qalist.values():
        if len(qalist) == 1:
            continue
        img_path = qalist[0]["image"]
        for qa in qalist[1:]:
            assert qa["image"] == img_path, (qa, img_path)
            
    for qalist in imgpath2qalist.values():
        if len(qalist) == 1:
            continue
        img_id = qalist[0]["id"]
        for qa in qalist[1:]:
            assert qa["id"] == img_id, (qa, img_id)
        
    
def get_conversation_from_okvqa_sample(que, ann, which_response: str = "most", with_image_token: bool = True):
    assert which_response in ["most", "first"]
    conversations = []
        
    assert que["image_id"] == ann["image_id"]
    assert que["question_id"] == ann["question_id"]
    
    prompt = "<image>\n" if with_image_token else ""
    prompt = prompt + que["question"] + "\nAnswer the question using a single word or phrase."
    answers = [ans["answer"] for ans in ann["answers"] if ans["answer_confidence"] and len(ans["answer"]) > 0]
    if len(answers) == 0:
        answers = [ann["answers"][0]["answer"]]
    
    if which_response == "most":
        response = Counter(answers).most_common(1)[0][0]
    elif which_response == "first":
        response = answers[0]
    else:
        raise NotImplementedError(which_response)
    
    conversations.append({"from": "human", "value": prompt})
    conversations.append({"from": "gpt", "value": response[0].upper() + response[1:]})
    return conversations


def main_convert_okvqa_trainset():
    okvqa_root = "/mnt/petrelfs/share_data/wangweiyun/datasets/OK-VQA/"
    annotations_path = f"{okvqa_root}/mscoco_train2014_annotations.json"
    questions_path = f"{okvqa_root}/OpenEnded_mscoco_train2014_questions.json"
    save_path = f"./llava_okvqa9k.json"
    
    annotations = json.load(open(annotations_path, "r"))["annotations"]
    questions = json.load(open(questions_path, "r"))["questions"]
    
    qid2question = {_q["question_id"]: _q for _q in questions}
    iid2annotations = {}
    for ann in annotations:
        image_id = ann["image_id"]
        if image_id not in iid2annotations:
            iid2annotations[image_id] = []
        iid2annotations[image_id].append(ann)
    
    new_annotations = []
    for image_id, img_annotations in iid2annotations.items():
        conversations = []
        for ann in img_annotations:
            question_id = ann["question_id"]
            que = qid2question[question_id]
            conversations.extend(get_conversation_from_okvqa_sample(que, ann))
            
        data = {
            "id": f"okvqa_{image_id}", 
            "image": f"COCO_train2014_{image_id:012d}.jpg", 
            "conversations": conversations
        }
        new_annotations.append(data)
    
    print(f"len(new_annotations) = {len(new_annotations)}")
    with open(save_path, "w") as f:
        json.dump(new_annotations, f)
        
        
def main_convert_okvqa_fewshot_trainset(num_shots=4):
    okvqa_root = "/mnt/petrelfs/share_data/wangweiyun/datasets/OK-VQA/"
    annotations_path = f"{okvqa_root}/mscoco_train2014_annotations.json"
    questions_path = f"{okvqa_root}/OpenEnded_mscoco_train2014_questions.json"
    save_path = f"./llava_{num_shots}shots_okvqa9k.json"
    
    annotations = json.load(open(annotations_path, "r"))["annotations"]
    questions = json.load(open(questions_path, "r"))["questions"]
    
    qid2question = {_q["question_id"]: _q for _q in questions}
    iid2annotations = {}
    for ann in annotations:
        image_id = ann["image_id"]
        if image_id not in iid2annotations:
            iid2annotations[image_id] = []
        iid2annotations[image_id].append(ann)
        
    num_img_more_than_one_ann = []
    img_more_than_one_ann = {}
    new_annotations = []
    for image_id, img_annotations in tqdm(iid2annotations.items()):
        if len(img_annotations) > 1:
            img_more_than_one_ann[image_id] = img_annotations
            num_img_more_than_one_ann.append(len(img_annotations))
        conversations = []
        for img_ann_id, ann in enumerate(img_annotations):
            question_id = ann["question_id"]
            que = qid2question[question_id]
            conversations.extend(get_conversation_from_okvqa_sample(que, ann, with_image_token=img_ann_id==0))

        data = {
            "image_id": image_id, 
            "id": f"{num_shots}shots_okvqa_{image_id}", 
            "image": f"COCO_train2014_{image_id:012d}.jpg", 
            "conversations": conversations
        }
        new_annotations.append(data)
        
    print(f"len(new_annotations) = {len(new_annotations)}")
    
    few_shots_sampler = get_sampler(num_shots=num_shots+2)
    for ann in tqdm(new_annotations):
        few_shots_samples = few_shots_sampler.get_fewshot_samples(ann["image"])
        for _ in range(2 if ann["image_id"] in img_more_than_one_ann else 1):
            _, last_fs_ann = few_shots_samples.pop(-1)
            assert last_fs_ann["image_id"] == ann["image_id"]
        incontext_images = []
        incontext_conversations = []
        for fs_sample in few_shots_samples[:num_shots]:
            fs_image_id = fs_sample[1]["image_id"]
            assert fs_image_id != ann["image_id"]
            incontext_images.append(f"COCO_train2014_{fs_image_id:012d}.jpg")
            incontext_conversations.extend(get_conversation_from_okvqa_sample(*fs_sample, which_response="first"))
        ann["incontext_images"] = incontext_images
        ann["incontext_conversations"] = incontext_conversations
        
    # check
    for ann_id, ann in tqdm(enumerate(new_annotations)):
        txt = ""
        for conv in ann["incontext_conversations"] + ann["conversations"]:
            txt += conv["value"]
        assert txt.count("<image>") == num_shots + 1, (ann_id, ann)
        assert len(ann["incontext_images"]) == num_shots
        
    
    with open(save_path, "w") as f:
        json.dump(new_annotations, f)
        
        
LLAVA_PROMPT = {
    "vqa": "\nAnswer the question using a single word or phrase.", 
    "aokvqa": "\nAnswer with the option's letter from the given choices directly.", 
    "textcap": "\nProvide a one-sentence caption for the provided image.\nReference OCR token:", 
    "grouonding1": "<image>\nPlease provide the bounding box coordinate of the region this sentence describes:", 
    "grouonding2": "<image>\nPlease provide a short description for this region:",
}
        
        
def disassemble_conversations_of_llava_samples(samples_list, dataset_name):
    new_samples_list = []
    for sample in deepcopy(samples_list):
        conversations = sample.pop("conversations")
        assert len(conversations) % 2 == 0, (len(conversations), conversations)
        for _conversations in [conversations[i:i+2] for i in range(0, len(conversations), 2)]:
            _sample = deepcopy(sample)
            if not _conversations[0]["value"].startswith("<image>"):
                _conversations[0]["value"] = "<image>\n" + _conversations[0]["value"]
            if dataset_name in ["vqav2", "okvqa", "gqa", "ocrvqa"]:
                if not _conversations[0]["value"].endswith(LLAVA_PROMPT["vqa"]):
                    _conversations[0]["value"] = _conversations[0]["value"] + LLAVA_PROMPT["vqa"]
            elif dataset_name in ["aokvqa"]:
                if not _conversations[0]["value"].endswith(LLAVA_PROMPT["aokvqa"]):
                    _conversations[0]["value"] = _conversations[0]["value"] + LLAVA_PROMPT["aokvqa"]
            elif dataset_name in ["textcaps"]:
                pass
            else:
                raise NotImplementedError(f"Unsupport dataset_name {dataset_name}")
            _sample["conversations"] = _conversations
            new_samples_list.append(_sample)
    print(f"disassemble conversations: {len(samples_list)} -> {len(new_samples_list)}")
    return new_samples_list
        
        
def convert_task_prompt_based_fewshot_trainset(
    num_shots=2,
    image_root="playground/data",
    llava_data_or_path = "playground/data/llava_v1_5_mix665k.json", 
    cached_features_path="./cached_features.pkl", 
    save_path=None, 
    filter_rule=lambda d: True,
    batch_size=None,
    disassemble_conversations=True, 
    dataset_name=None,
    post_incontext=True,
):
    if isinstance(llava_data_or_path, str):
        llava_data = json.load(open(llava_data_or_path, "r"))
    else:
        llava_data = llava_data_or_path
    
    filtered_samples = [d for d in llava_data if filter_rule(d)]
    
    if disassemble_conversations:
        filtered_samples = disassemble_conversations_of_llava_samples(filtered_samples, dataset_name)
    
    imgid2qalist = {}
    for sample in deepcopy(filtered_samples):
        if sample["id"] not in imgid2qalist:
            imgid2qalist[sample["id"]] = []
        imgid2qalist[sample["id"]].append(sample)
    
    # 为数据集里每个图提取特征
    image_data_list = [{
        "image_id": imgid, 
        "image_path": os.path.join(image_root, imgid2qalist[imgid][0]["image"]), 
    } for imgid in imgid2qalist.keys()]
    rices_sampler = CustomRICES(
        cached_features_path=cached_features_path,
        data_list=image_data_list, 
    )
    features = rices_sampler.prepare_cached_feature()
    
    # 这里直接单独计算一个image对image的映射对应关系
    image_ids = list(imgid2qalist.keys())
    with torch.no_grad():
        image_features = torch.stack([features[imgid]["features"] for imgid in image_ids]).cuda()  # [16540, 512]
        
        if batch_size is None:
            cos_sim = (image_features @ image_features.T).detach().cpu()
        else:
            cos_sim = torch.zeros((image_features.shape[0], image_features.shape[0]))
            for i in range(0, image_features.shape[0], batch_size):
                batch_features = image_features[i:i+batch_size]
                cos_sim[i:i+batch_size] = (batch_features @ image_features.T).detach().cpu()
        
        cos_sim.fill_diagonal_(-float('inf'))
        _, indices = torch.topk(cos_sim, num_shots, dim=1)  # [16540, num_shots]
    
    # 为每个图提取 num_shots 个样本
    new_annotations = []
    for i, imgid in enumerate(image_ids):
        qalist = imgid2qalist[imgid]
        if disassemble_conversations:
            # 拆开对话的话，每个图只保留一个作为结尾的样本
            qa = random.choice(qalist)
            incontext_image_indices = indices[i].clone().tolist()
            ic_qa_list = [random.choice(imgid2qalist[image_ids[ic_idx]]) for ic_idx in incontext_image_indices]
            
            if not post_incontext:
                incontext_images = []
                incontext_conversations = []
                for ic_qa in ic_qa_list:
                    incontext_images.append(ic_qa["image"])
                    incontext_conversations.extend(ic_qa["conversations"])
                    
                new_annotations.append({
                    "id": f"{num_shots}shots_aokvqa_{imgid}", 
                    "incontext_images": incontext_images, 
                    "incontext_conversations": incontext_conversations, 
                    "image": qa["image"],
                    "conversations": qa["conversations"]
                })
            else:
                images = [deepcopy(qa["image"])]
                conversations = deepcopy(qa["conversations"])
                for ic_qa in ic_qa_list:
                    images.append(ic_qa["image"])
                    conversations.extend(ic_qa["conversations"])
                    
                new_annotations.append({
                    "id": f"{num_shots}shots_aokvqa_{imgid}", 
                    "image": images,
                    "conversations": conversations
                })
        else:
            # 不拆对话的话，每个对话保留一个作为结尾的样本
            for j, qa in enumerate(qalist):
                incontext_image_indices = indices[i].clone().tolist()
                random.shuffle(incontext_image_indices)
                ic_qa_list = [random.choice(imgid2qalist[image_ids[ic_idx]]) for ic_idx in incontext_image_indices]
                
                if not post_incontext:
                    incontext_images = []
                    incontext_conversations = []
                    for ic_qa in ic_qa_list:
                        incontext_images.append(ic_qa["image"])
                        incontext_conversations.extend(ic_qa["conversations"])
                        
                    new_annotations.append({
                        "id": f"{num_shots}shots_aokvqa_{imgid}_{j}", 
                        "incontext_images": incontext_images, 
                        "incontext_conversations": incontext_conversations, 
                        "image": qa["image"],
                        "conversations": qa["conversations"],
                    })
                else:
                    images = [deepcopy(qa["image"])]
                    conversations = deepcopy(qa["conversations"])
                    for ic_qa in ic_qa_list:
                        images.append(ic_qa["image"])
                        conversations.extend(ic_qa["conversations"])
                        
                    new_annotations.append({
                        "id": f"{num_shots}shots_aokvqa_{imgid}_{j}", 
                        "image": images,
                        "conversations": conversations,
                    })
    
    if save_path is not None:
        with open(save_path, "w") as f:
            json.dump(new_annotations, f)
    return new_annotations


def main_convert_4vqa_fewshot_trainset(num_shots=2):
    """legacy"""
    image_root = "playground/data/"
    save_path = f"./llava_{num_shots}shots_4vqa244k.json"
    cached_features_path="./cached_features_4vqa.pkl"
    LLAVA_4VQA_TAIL_PROMPT = LLAVA_PROMPT["vqa"]
    filter_rule = lambda d: d["conversations"][0]["value"].endswith(LLAVA_4VQA_TAIL_PROMPT)
    new_annotations = convert_task_prompt_based_fewshot_trainset(
        num_shots=num_shots, 
        image_root=image_root, 
        cached_features_path=cached_features_path, 
        save_path=save_path, 
        filter_rule=filter_rule,
        batch_size=5000, 
    )
    print(len(new_annotations))
    
    
def main_convert_vqav2_okvqa_fewshot_trainset(num_shots=2):
    """legacy"""
    image_root = "playground/data/"
    save_path = f"./llava_{num_shots}shots_vqav2okvqa92k.json"
    cached_features_path="./cached_features_4vqa.pkl"
    LLAVA_VQAv2_OKVQA_TAIL_PROMPT = LLAVA_PROMPT["vqa"]
    filter_rule = lambda d: (d["conversations"][0]["value"].endswith(LLAVA_VQAv2_OKVQA_TAIL_PROMPT)) and (not d["image"].startswith("ocr")) and (not d["image"].startswith("gqa"))
    new_annotations = convert_task_prompt_based_fewshot_trainset(
        num_shots=num_shots, 
        image_root=image_root, 
        cached_features_path=cached_features_path, 
        save_path=save_path, 
        filter_rule=filter_rule,
    )
    print(len(new_annotations))
    
    
def main_convert_2grounding_fewshot_trainset(num_shots=2):
    """legacy"""
    image_root = "playground/data/"
    save_path = f"./llava_{num_shots}shots_2grounding135k.json"
    cached_features_path="./cached_features_2grounding.pkl"
    LLAVA_2GROUNDING_PROMPT_1 = LLAVA_PROMPT["grouonding1"]
    LLAVA_2GROUNDING_PROMPT_2 = LLAVA_PROMPT["grouonding2"]
    filter_rule = lambda d: ((d["conversations"][0]["value"].startswith(LLAVA_2GROUNDING_PROMPT_1)) or (d["conversations"][0]["value"].startswith(LLAVA_2GROUNDING_PROMPT_2)))
    new_annotations = convert_task_prompt_based_fewshot_trainset(
        num_shots=num_shots, 
        image_root=image_root, 
        cached_features_path=cached_features_path, 
        save_path=save_path, 
        filter_rule=filter_rule,
        batch_size=5000,  
    )
    print(len(new_annotations))
    
    
def main_convert_vg_fewshot_trainset(num_shots=2):
    """legacy"""
    image_root = "playground/data/"
    save_path = f"./llava_{num_shots}shots_vg86k.json"
    cached_features_path="./cached_features_2grounding.pkl"
    LLAVA_2GROUNDING_PROMPT_1 = LLAVA_PROMPT["grouonding1"]
    LLAVA_2GROUNDING_PROMPT_2 = LLAVA_PROMPT["grouonding2"]
    filter_rule = lambda d: ((d["conversations"][0]["value"].startswith(LLAVA_2GROUNDING_PROMPT_1)) or (d["conversations"][0]["value"].startswith(LLAVA_2GROUNDING_PROMPT_2))) and (d["image"].startswith("vg"))
    new_annotations = convert_task_prompt_based_fewshot_trainset(
        num_shots=num_shots, 
        image_root=image_root, 
        cached_features_path=cached_features_path, 
        save_path=save_path, 
        filter_rule=filter_rule, 
    )
    print(len(new_annotations))
    
    
def main_convert_refcoco_fewshot_trainset(num_shots=2):
    """legacy"""
    image_root = "playground/data/"
    save_path = f"./llava_{num_shots}shots_refcoco48k.json"
    cached_features_path="./cached_features_2grounding.pkl"
    LLAVA_2GROUNDING_PROMPT_1 = LLAVA_PROMPT["grouonding1"]
    LLAVA_2GROUNDING_PROMPT_2 = LLAVA_PROMPT["grouonding2"]
    filter_rule = lambda d: ((d["conversations"][0]["value"].startswith(LLAVA_2GROUNDING_PROMPT_1)) or (d["conversations"][0]["value"].startswith(LLAVA_2GROUNDING_PROMPT_2))) and (d["image"].startswith("coco"))
    new_annotations = convert_task_prompt_based_fewshot_trainset(
        num_shots=num_shots, 
        image_root=image_root, 
        cached_features_path=cached_features_path, 
        save_path=save_path, 
        filter_rule=filter_rule,
    )
    print(len(new_annotations))
    
    
def main_convert_aokvqa_fewshot_trainset(num_shots=2, disassemble_conversations=True):
    """
    /mnt/petrelfs/share_data/liqingyun/lmm_baseline_llava/sft_data/llava_xshots_aokvqa66k/llava_2shots_aokvqa66k.json
    A-OKVQA 提供了一个 question, 多个choice, 和答案index
    Llava 在处理 A-OKVQA 的时候，进行了 1.数据增广, 把abcd进行位置替换  2.组织prompt, 包括构造ABCD的形式和在结尾加上任务相关提示
    Llava 对 A-OKVQA 的尾部 prompt 是唯一的, 所以可以通过这个直接过滤出 A-OKVQA 的样本
    注意: 由于数据增广, Llava数据形式里面的id就是图像id, 也就是说会重复, 非唯一; 并且数据的重复内容比较多, 所以在使用时可以把不带上下文的样本滤掉
    """
    image_root = "playground/data/"
    save_path = f"./llava_{num_shots}shots_aokvqa66k.json" if not disassemble_conversations else f"./llava_{num_shots}shots_aokvqa17k.json"
    cached_features_path="./cached_features_aokvqa.pkl"
    LLAVA_AOKVQA_TAIL_PROMPT = LLAVA_PROMPT["aokvqa"]
    filter_rule = lambda d: d["conversations"][0]["value"].endswith(LLAVA_AOKVQA_TAIL_PROMPT)
    new_annotations = convert_task_prompt_based_fewshot_trainset(
        num_shots=num_shots, 
        image_root=image_root, 
        cached_features_path=cached_features_path, 
        save_path=save_path, 
        filter_rule=filter_rule,
        disassemble_conversations=disassemble_conversations, 
        dataset_name="aokvqa",
    )
    print(len(new_annotations))
    
    
def main_convert_gqa_fewshot_trainset(num_shots=2, disassemble_conversations=True):
    image_root = "playground/data/"
    save_path = f"./llava_{num_shots}shots_gqa72k.json"
    cached_features_path="./cached_features_4vqa.pkl"
    LLAVA_GQA_TAIL_PROMPT = LLAVA_PROMPT["vqa"]
    filter_rule = lambda d: (d["conversations"][0]["value"].endswith(LLAVA_GQA_TAIL_PROMPT)) and (d["image"].startswith("gqa"))
    new_annotations = convert_task_prompt_based_fewshot_trainset(
        num_shots=num_shots, 
        image_root=image_root, 
        cached_features_path=cached_features_path, 
        save_path=save_path, 
        filter_rule=filter_rule,
        disassemble_conversations=disassemble_conversations, 
        dataset_name="gqa", 
    )
    print(len(new_annotations))
    
    
def main_convert_ocrvqa_fewshot_trainset(num_shots=2, disassemble_conversations=True):
    image_root = "playground/data/"
    save_path = f"./llava_{num_shots}shots_ocrvqa80k.json"
    cached_features_path="./cached_features_4vqa.pkl"
    LLAVA_OCRVQA_TAIL_PROMPT = LLAVA_PROMPT["vqa"]
    filter_rule = lambda d: (d["conversations"][0]["value"].endswith(LLAVA_OCRVQA_TAIL_PROMPT)) and (d["image"].startswith("ocr"))
    new_annotations = convert_task_prompt_based_fewshot_trainset(
        num_shots=num_shots, 
        image_root=image_root, 
        cached_features_path=cached_features_path, 
        save_path=save_path, 
        filter_rule=filter_rule,
        disassemble_conversations=disassemble_conversations, 
        dataset_name="ocrvqa", 
    )
    print(len(new_annotations))
    
    
def main_convert_vqav2_fewshot_trainset(num_shots=2, disassemble_conversations=True):
    image_root = "playground/data/"
    save_path = f"./llava_{num_shots}shots_vqavii83k.json"
    cached_features_path="./cached_features_4vqa.pkl"
    LLAVA_VQA_TAIL_PROMPT = LLAVA_PROMPT["vqa"]
    filter_rule = lambda d: (
        (d["conversations"][0]["value"].endswith(LLAVA_VQA_TAIL_PROMPT)) and 
        (not d["image"].startswith("ocr")) and 
        (not d["image"].startswith("gqa")) and 
        (len(d["conversations"]) > 4)
    )  # VQAv2 always has 3-275 qa-pairs, while OKVQA always has 1-2 qa-pairs
    new_annotations = convert_task_prompt_based_fewshot_trainset(
        num_shots=num_shots, 
        image_root=image_root, 
        cached_features_path=cached_features_path, 
        save_path=save_path, 
        filter_rule=filter_rule,
        disassemble_conversations=disassemble_conversations, 
        dataset_name="vqav2", 
    )
    print(len(new_annotations))
    
    
def main_convert_okvqa_fewshot_trainset(num_shots=2, disassemble_conversations=True):
    image_root = "playground/data/"
    save_path = f"./llava_{num_shots}shots_okvqa9k.json"
    cached_features_path="./cached_features_4vqa.pkl"
    LLAVA_VQA_TAIL_PROMPT = LLAVA_PROMPT["vqa"]
    filter_rule = lambda d: (
        (d["conversations"][0]["value"].endswith(LLAVA_VQA_TAIL_PROMPT)) and 
        (not d["image"].startswith("ocr")) and 
        (not d["image"].startswith("gqa")) and 
        (len(d["conversations"]) <= 4)
    )  # VQAv2 always has 3-275 qa-pairs, while OKVQA always has 1-2 qa-pairs
    new_annotations = convert_task_prompt_based_fewshot_trainset(
        num_shots=num_shots, 
        image_root=image_root, 
        cached_features_path=cached_features_path, 
        save_path=save_path, 
        filter_rule=filter_rule,
        disassemble_conversations=disassemble_conversations,
        dataset_name="okvqa",  
    )
    print(len(new_annotations))
    
    
def main_convert_textcaps_fewshot_trainset(num_shots=2, disassemble_conversations=True):
    image_root = "playground/data/"
    save_path = f"./llava_{num_shots}shots_textcaps22k.json"
    cached_features_path="./cached_features_textcaps.pkl"
    LLAVA_TEXTCAPS_PROMPT = LLAVA_PROMPT["textcap"]
    filter_rule = lambda d: LLAVA_TEXTCAPS_PROMPT in d["conversations"][0]["value"]
    new_annotations = convert_task_prompt_based_fewshot_trainset(
        num_shots=num_shots, 
        image_root=image_root, 
        cached_features_path=cached_features_path, 
        save_path=save_path, 
        filter_rule=filter_rule,
        disassemble_conversations=disassemble_conversations, 
        dataset_name="textcaps",
    )
    print(len(new_annotations))
    
    
def main_convert_ocrvqa_new_fewshot_trainset(num_shots=2, disassemble_conversations=True):
    image_root = "/mnt/petrelfs/share_data/wangweiyun/datasets/OCR-VQA/images"
    save_path = f"/mnt/petrelfs/share_data/liqingyun/lmm_baseline_llava/sft_data/llava_xshots_ocrvqa/llava_{num_shots}shots_ocrvqa166k.json"
    cached_features_path="./cached_features_ocrvqa.pkl"
    new_annotations = convert_task_prompt_based_fewshot_trainset(
        llava_data_or_path="/mnt/petrelfs/share_data/liqingyun/lmm_baseline_llava/sft_data/llava_xshots_ocrvqa/llava_ocrvqa_verbose_wtoken_data_166k.json",
        num_shots=num_shots, 
        image_root=image_root, 
        cached_features_path=cached_features_path, 
        save_path=save_path, 
        disassemble_conversations=disassemble_conversations, 
        dataset_name="textcaps",
        batch_size=5000, 
    )
    print(len(new_annotations))


def main_merge_few_shot_sft_v4(num_shots=5):
    data_root = "/mnt/petrelfs/share_data/liqingyun/lmm_baseline_llava/sft_data"
    os.makedirs("/mnt/petrelfs/liqingyun/share_data/lmm_baseline_llava/sft_data/llava_xshots_v4mix260k/", exist_ok=True)
    save_path = None
    save_path = f"/mnt/petrelfs/liqingyun/share_data/lmm_baseline_llava/sft_data/llava_xshots_v4mix260k/llava_{num_shots}shots_v4mix260k.json"
    merged_annotations = []
    for annotation_path in [
        f"{data_root}/llava_xshots_aokvqa17k/llava_{num_shots}shots_aokvqa17k", 
        f"{data_root}/llava_xshots_gqa72k/llava_{num_shots}shots_gqa72k", 
        f"{data_root}/llava_xshots_ocrvqa80k/llava_{num_shots}shots_ocrvqa80k", 
        f"{data_root}/llava_xshots_okvqa9k/llava_{num_shots}shots_okvqa9k", 
        # f"{data_root}/llava_xshots_textcaps22k/llava_{num_shots}shots_textcaps22k", 
        f"{data_root}/llava_xshots_vqavii83k/llava_{num_shots}shots_vqavii83k", 
    ]:
        merged_annotations.extend(json.load(open(f"{annotation_path}.json", "r")))
            
    if save_path is not None:
        with open(save_path, "w") as f:
            json.dump(merged_annotations, f)
    print(len(merged_annotations))
    return merged_annotations
        
        
def main_filter_out_v4mix260k_data():
    ann_path = f"/mnt/petrelfs/share_data/liqingyun/lmm_baseline_llava/sft_data/llava_v1_5_mix/llava_v1_5_mix665k.json"
    out_path = f"/mnt/petrelfs/share_data/liqingyun/lmm_baseline_llava/sft_data/llava_v1_5_mix/llava_v1_5_filteroutv4mix260k_mix355k.json"
    
    llava_data = json.load(open(ann_path, "r"))
    aokvqa_filter_rule = lambda d: d["conversations"][0]["value"].endswith(LLAVA_PROMPT["aokvqa"])
    vqa_filter_rule = lambda d: LLAVA_PROMPT["vqa"] in d["conversations"][0]["value"]
    filter_rule = lambda d: not (aokvqa_filter_rule(d) or vqa_filter_rule(d))
    
    new_llava_data = [d for d in llava_data if filter_rule(d)]
    print(len(new_llava_data))
    
    with open(out_path, "w") as f:
        json.dump(new_llava_data, f)


if __name__ == "__main__":
    for main in (
        main_convert_aokvqa_fewshot_trainset, 
        main_convert_gqa_fewshot_trainset, 
        main_convert_ocrvqa_fewshot_trainset, 
        main_convert_okvqa_fewshot_trainset,
        main_convert_vqav2_fewshot_trainset,
        main_convert_textcaps_fewshot_trainset, 
        main_merge_few_shot_sft_v4,
        main_convert_ocrvqa_new_fewshot_trainset,
    ):
        for num_shots in (
            2, 
            3, 
            4, 
            5,
        ):
            print(f"{main.__name__} | num_shots={num_shots}")
            main(num_shots=num_shots)
            
    main_filter_out_v4mix260k_data()
    print(f"main_filter_out_v4mix260k_data | num_shots={num_shots}")
