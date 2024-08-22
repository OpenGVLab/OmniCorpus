"""
Run `pip install nltk scikit-learn inflection`

And you may need to download for nltk with:
```Python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

NOTE you may need to delete spice in pycocoevalcap if is useless 
`$CONDA_HOME/envs/xxx/lib/pythonX.X/site-packages/pycocoevalcap/eval.py`
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse
import json
import uuid
import random
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import utils
import math

import torch

from coco_metric import compute_cider, postprocess_captioning_generation
from eval_datasets import (
    CaptionDataset,
    VQADataset,
    ImageNetDataset,
    HatefulMemesDataset,
)
from rices import RICES
from tqdm import tqdm


from classification_utils import (
    IMAGENET_CLASSNAMES,
    HM_CLASSNAMES,
)

from eval_models import OpenFlamingoEvalModel, InternVLChatEvalModel
from ok_vqa_utils import postprocess_ok_vqa_generation
from vqa_metric import compute_vqa_accuracy, postprocess_vqa_generation
from utils import world_info_from_env, init_distributed_device


# InternVL Special Tokens
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'

ds_collections = {
    'flickr': {
        'image_dir_path': 'data/flickr30k/flickr30k-images',
        'karpathy_json_path': 'data/flickr30k/dataset_flickr30k.json',
        'annotations_json_path': 'data/flickr30k/dataset_flickr30k_coco_style.json',
    },
    'coco': {
        'train_image_dir_path': 'datasets/coco/train2014',
        'val_image_dir_path': 'datasets/coco/val2014',
        'karpathy_json_path': 'datasets/coco/annotations/karpathy_caption/dataset_coco.json',
        'annotations_json_path': 'datasets/coco/annotations/captions_val2014.json',
    },
    'vqav2': {
        'train_image_dir_path': 'datasets/coco/train2014',
        'train_questions_json_path': 'datasets/VQAv2/v2_OpenEnded_mscoco_train2014_questions.json',
        'train_annotations_json_path': 'datasets/VQAv2/v2_mscoco_train2014_annotations.json',
        'test_image_dir_path': 'datasets/coco/test2015',
        'test_questions_json_path': 'datasets/VQAv2/v2_OpenEnded_mscoco_test-dev2015_questions.json',
        'test_annotations_json_path': None,
        'final_test_questions_json_path': 'datasets/VQAv2/v2_OpenEnded_mscoco_test2015_questions.json',
    },
    'ok_vqa':{
        'train_image_dir_path': 'datasets/coco/train2014',
        'train_questions_json_path': 'datasets/OK-VQA/OpenEnded_mscoco_train2014_questions.json',
        'train_annotations_json_path': 'datasets/OK-VQA/mscoco_train2014_annotations.json',
        'test_image_dir_path': 'datasets/coco/val2014',
        'test_questions_json_path': 'datasets/OK-VQA/OpenEnded_mscoco_val2014_questions.json',
        'test_annotations_json_path': 'datasets/OK-VQA/mscoco_val2014_annotations.json',
    },
    'vizwiz':{
        'train_image_dir_path': 'datasets/VizWiz-VQA/train',
        'test_image_dir_path': 'datasets/VizWiz-VQA/test',
        'train_questions_json_path': 'datasets/VizWiz-VQA/openflamingo_eval_ann/train_questions_vqa_format.json',
        'train_annotations_json_path': 'datasets/VizWiz-VQA/openflamingo_eval_ann/train_annotations_vqa_format.json',
        'test_questions_json_path': 'datasets/VizWiz-VQA/openflamingo_eval_ann/test_questions_vqa_format.json',
        'test_annotations_json_path': None,
    },
    'textvqa': {
        'image_dir_path': 'datasets/TextVQA/train_images',
        'train_questions_json_path': 'datasets/TextVQA/openflamingo_eval_ann/train_questions_vqa_format.json',
        'train_annotations_json_path': 'datasets/TextVQA/openflamingo_eval_ann/train_annotations_vqa_format.json',
        'test_questions_json_path': 'datasets/TextVQA/openflamingo_eval_ann/val_questions_vqa_format.json',
        'test_annotations_json_path': 'datasets/TextVQA/openflamingo_eval_ann/val_annotations_vqa_format.json',
    },
    'imagenet': {
        'root': None,
    },
    'hateful_memes': {
        'image_dir_path': None,
        'train_annotations_json_path': None,
        'test_annotations_json_path': None,
    },
    'rices': {
        'vision_encoder_path': 'ViT-L-14',
        'vision_encoder_pretrained': 'openai',
        'cached_demonstration_features': 'rices_cached_feature',
    },
}
    
    
all_ds = [_ for _ in ds_collections.keys() if _ != "rices"]


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--datasets', nargs="+", type=str, default=all_ds, choices=all_ds, help='Datasets to evaluate on')
    parser.add_argument("--model", type=str, help="Model name", default="open_flamingo")
    parser.add_argument("--results_file", type=str, default=None, help="JSON file to save results")

    # Trial arguments
    parser.add_argument("--shots", nargs="+", default=[0, 4, 8, 16, 32], type=int)
    parser.add_argument("--num_trials", type=int, default=1, 
                        help="Number of trials to run for each shot using different demonstrations")
    parser.add_argument("--trial_seeds", nargs="+", type=int, default=[42], 
                        help="Seeds to use for each trial for picking demonstrations and eval sets")
    parser.add_argument("--num_samples", type=int, default=-1, 
                        help="Number of samples to evaluate on. -1 for all samples.")
    parser.add_argument("--query_set_size", type=int, default=2048, help="Size of demonstration query set")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--rices", action="store_true", 
                        help="Whether to use RICES for evaluation. If False, uses random demonstrations.")
    parser.add_argument(
        "--no_caching_for_classification",
        action="store_true",
        help="Whether to skip using key-value caching for classification evals, which usually speeds it up.",
    )
    parser.add_argument(
        "--classification_prompt_ensembling",
        action="store_true",
        help="Whether to use prompt ensembling (average log-likelihoods over permutations of in-context examples)",
    )

    # Distributed evaluation
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
    )
    parser.add_argument(
        "--zero-shot-add-text-shots",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--chat-few-shot-style",
        default="multi",
        choices=["single", "single_round_conversation", "multi", "multi_rounds_conversation", "direct"], 
        type=str,
    )
    parser.add_argument(
        "--verbose",
        default=False,
        action="store_true",
    )
    
    return parser


def parse_ds_collections(args):
    for ds_name, ds_args in ds_collections.items():
        for arg_name in ds_args.keys():
            if getattr(args, "rank", 0) == 0:
                print(f"Setting args.{ds_name}_{arg_name}={ds_args[arg_name]}")
            setattr(args, f"{ds_name}_{arg_name}", ds_args[arg_name])
            
    for ds in all_ds:
        setattr(args, f"eval_{ds}", ds in args.datasets)


def main():
    args, leftovers = get_parser().parse_known_args()
    
    model_args = {
        leftovers[i].lstrip("-"): leftovers[i + 1] for i in range(0, len(leftovers), 2)
    }
    if args.model == "open_flamingo":
        eval_model = OpenFlamingoEvalModel(model_args)
    elif args.model == "internvl_chat":
        eval_model = InternVLChatEvalModel(model_args)
    else:
        raise NotImplementedError(f"Model {args.model} not implemented")

    # set up distributed evaluation
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    device_id = init_distributed_device(args)
    eval_model.set_device(device_id)
    eval_model.init_distributed()
    
    # add options to args
    parse_ds_collections(args)

    # if args.model != "open_flamingo" and args.shots != [0]:
    #     raise ValueError("Only 0 shot eval is supported for non-open_flamingo models")

    if len(args.trial_seeds) != args.num_trials:
        raise ValueError("Number of trial seeds must be == number of trials.")

    results = defaultdict(list)

    if args.eval_flickr:
        if args.rank == 0:
            print("Evaluating on Flickr30k...")

        # load cached demonstration features for RICES
        if args.rices_cached_demonstration_features is not None and args.rices:
            cached_features = torch.load(
                f"{args.rices_cached_demonstration_features}/flickr30.pkl", map_location="cpu"
            )
        else:
            cached_features = None

        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                cider_score = evaluate_captioning(
                    args,
                    eval_model=eval_model,
                    num_shots=shot,
                    seed=seed,
                    dataset_name="flickr",
                    cached_features=cached_features,
                )
                if args.rank == 0:
                    print(f"Shots {shot} Trial {trial} CIDEr score: {cider_score}")
                    scores.append(cider_score)

            if args.rank == 0:
                print(f"Shots {shot} Mean CIDEr score: {np.nanmean(scores)}")
                results["flickr30"].append(
                    {
                        "shots": shot,
                        "trials": scores,
                        "mean": np.nanmean(scores),
                        "stddev": np.nanstd(scores),
                    }
                )

    if args.eval_coco:
        if args.rank == 0:
            print("Evaluating on COCO...")

        # load cached demonstration features for RICES
        if args.rices_cached_demonstration_features is not None and args.rices:
            cached_features = torch.load(
                f"{args.rices_cached_demonstration_features}/coco.pkl", map_location="cpu"
            )
        else:
            cached_features = None

        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                cider_score = evaluate_captioning(
                    args,
                    eval_model=eval_model,
                    num_shots=shot,
                    seed=seed,
                    dataset_name="coco",
                    cached_features=cached_features,
                )
                if args.rank == 0:
                    print(f"Shots {shot} Trial {trial} CIDEr score: {cider_score}")
                    scores.append(cider_score)

            if args.rank == 0:
                print(f"Shots {shot} Mean CIDEr score: {np.nanmean(scores)}")
                results["coco"].append(
                    {
                        "shots": shot,
                        "trials": scores,
                        "mean": np.nanmean(scores),
                        "stddev": np.nanstd(scores),
                    }
                )

    if args.eval_ok_vqa:
        if args.rank == 0:
            print("Evaluating on OK-VQA...")

        # load cached demonstration features for RICES
        if args.rices_cached_demonstration_features is not None and args.rices:
            cached_features = torch.load(
                f"{args.rices_cached_demonstration_features}/ok_vqa.pkl", map_location="cpu"
            )
        else:
            cached_features = None

        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                ok_vqa_score = evaluate_vqa(
                    args=args,
                    eval_model=eval_model,
                    num_shots=shot,
                    seed=seed,
                    dataset_name="ok_vqa",
                    cached_features=cached_features,
                )
                if args.rank == 0:
                    print(f"Shots {shot} Trial {trial} OK-VQA score: {ok_vqa_score}")
                    scores.append(ok_vqa_score)

            if args.rank == 0:
                print(f"Shots {shot} Mean OK-VQA score: {np.nanmean(scores)}")
                results["ok_vqa"].append(
                    {
                        "shots": shot,
                        "trials": scores,
                        "mean": np.nanmean(scores),
                        "stddev": np.nanstd(scores),
                    }
                )

    if args.eval_vqav2:
        if args.rank == 0:
            print("Evaluating on VQAv2...")

        # load cached demonstration features for RICES
        if args.rices_cached_demonstration_features is not None and args.rices:
            cached_features = torch.load(
                f"{args.rices_cached_demonstration_features}/vqav2.pkl", map_location="cpu"
            )
        else:
            cached_features = None

        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                vqa_score = evaluate_vqa(
                    args=args,
                    eval_model=eval_model,
                    num_shots=shot,
                    seed=seed,
                    dataset_name="vqav2",
                    cached_features=cached_features,
                )
                if args.rank == 0 and vqa_score is not None:
                    print(f"Shots {shot} Trial {trial} VQA score: {vqa_score}")
                    scores.append(vqa_score)

            if args.rank == 0 and len(scores) > 0:
                print(f"Shots {shot} Mean VQA score: {np.nanmean(scores)}")
                results["vqav2"].append(
                    {
                        "shots": shot,
                        "trials": scores,
                        "mean": np.nanmean(scores),
                        "stddev": np.nanstd(scores),
                    }
                )

    if args.eval_vizwiz:
        if args.rank == 0:
            print("Evaluating on VizWiz...")

        # load cached demonstration features for RICES
        if args.rices_cached_demonstration_features is not None and args.rices:
            cached_features = torch.load(
                f"{args.rices_cached_demonstration_features}/vizwiz.pkl", map_location="cpu"
            )
        else:
            cached_features = None

        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                vizwiz_score = evaluate_vqa(
                    args=args,
                    eval_model=eval_model,
                    num_shots=shot,
                    seed=seed,
                    dataset_name="vizwiz",
                    cached_features=cached_features,
                )
                if args.rank == 0 and vizwiz_score is not None:
                    print(f"Shots {shot} Trial {trial} VizWiz score: {vizwiz_score}")
                    scores.append(vizwiz_score)

            if args.rank == 0 and len(scores) > 0:
                print(f"Shots {shot} Mean VizWiz score: {np.nanmean(scores)}")
                results["vizwiz"].append(
                    {
                        "shots": shot,
                        "trials": scores,
                        "mean": np.nanmean(scores),
                        "stddev": np.nanstd(scores),
                    }
                )

    if args.eval_textvqa:
        if args.rank == 0:
            print("Evaluating on TextVQA...")

        # load cached demonstration features for RICES
        if args.rices_cached_demonstration_features is not None and args.rices:
            cached_features = torch.load(
                f"{args.rices_cached_demonstration_features}/textvqa.pkl", map_location="cpu"
            )
        else:
            cached_features = None

        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                textvqa_score = evaluate_vqa(
                    args=args,
                    eval_model=eval_model,
                    num_shots=shot,
                    seed=seed,
                    dataset_name="textvqa",
                    max_generation_length=10,
                    cached_features=cached_features,
                )
                if args.rank == 0:
                    print(f"Shots {shot} Trial {trial} TextVQA score: {textvqa_score}")
                    scores.append(textvqa_score)

            if args.rank == 0:
                print(f"Shots {shot} Mean TextVQA score: {np.nanmean(scores)}")
                results["textvqa"].append(
                    {
                        "shots": shot,
                        "trials": scores,
                        "mean": np.nanmean(scores),
                        "stddev": np.nanstd(scores),
                    }
                )

    if args.eval_imagenet:
        if args.rank == 0:
            print("Evaluating on ImageNet...")

        # load cached demonstration features for RICES
        if args.rices_cached_demonstration_features is not None and args.rices:
            cached_features = torch.load(
                f"{args.rices_cached_demonstration_features}/imagenet.pkl", map_location="cpu"
            )
        else:
            cached_features = None

        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                imagenet_score = evaluate_classification(
                    args,
                    eval_model=eval_model,
                    num_shots=shot,
                    seed=seed,
                    no_kv_caching=args.no_caching_for_classification,
                    dataset_name="imagenet",
                    cached_features=cached_features,
                    use_prompt_ensembling=args.classification_prompt_ensembling,
                )
                if args.rank == 0:
                    print(
                        f"Shots {shot} Trial {trial} "
                        f"ImageNet score: {imagenet_score}"
                    )
                    scores.append(imagenet_score)

            if args.rank == 0:
                print(f"Shots {shot} Mean ImageNet score: {np.nanmean(scores)}")
                results["imagenet"].append(
                    {
                        "shots": shot,
                        "trials": scores,
                        "mean": np.nanmean(scores),
                        "stddev": np.nanstd(scores),
                    }
                )

    if args.eval_hateful_memes:
        if args.rank == 0:
            print("Evaluating on Hateful Memes...")

        # load cached demonstration features for RICES
        if args.rices_cached_demonstration_features is not None and args.rices:
            cached_features = torch.load(
                f"{args.rices_cached_demonstration_features}/hateful_memes.pkl",
                map_location="cpu",
            )
        else:
            cached_features = None

        for shot in args.shots:
            scores = []
            for seed, trial in zip(args.trial_seeds, range(args.num_trials)):
                hateful_memes_score = evaluate_classification(
                    args,
                    eval_model=eval_model,
                    num_shots=shot,
                    seed=seed,
                    no_kv_caching=args.no_caching_for_classification,
                    dataset_name="hateful_memes",
                    cached_features=cached_features,
                )
                if args.rank == 0:
                    print(
                        f"Shots {shot} Trial {trial} "
                        f"Hateful Memes score: {hateful_memes_score}"
                    )
                    scores.append(hateful_memes_score)

            if args.rank == 0:
                print(f"Shots {shot} Mean Hateful Memes score: {np.nanmean(scores)}")
                results["hateful_memes"].append(
                    {
                        "shots": shot,
                        "trials": scores,
                        "mean": np.nanmean(scores),
                        "stddev": np.nanstd(scores),
                    }
                )

    if args.rank == 0 and args.results_file is not None:
        with open(args.results_file, "w") as f:
            json.dump(results, f)


def evaluate_captioning(
    args: argparse.Namespace,
    eval_model,
    seed: int = 42,
    min_generation_length: int = 0,
    max_generation_length: int = 20,
    num_beams: int = 3,
    length_penalty: float = 0.0,
    num_shots: int = 8,
    dataset_name: str = "coco",
    cached_features=None,
):
    """Evaluate a model on COCO dataset.

    Args:
        args (argparse.Namespace): arguments
        eval_model (XXXEvalModel): model to evaluate
        seed (int, optional): seed for random number generator. Defaults to 42.
        max_generation_length (int, optional): maximum length of the generated caption. Defaults to 20.
        num_beams (int, optional): number of beams to use for beam search. Defaults to 3.
        length_penalty (float, optional): length penalty for beam search. Defaults to -2.0.
        num_shots (int, optional): number of in-context samples to use. Defaults to 8.
        dataset_name (str, optional): dataset to evaluate on. Can be "coco" or "flickr". Defaults to "coco".
        cached_features (tensor, optional): cached demonstration features for RICES. Defaults to None.
    Returns:
        float: CIDEr score

    """

    if dataset_name == "coco":
        image_train_dir_path = args.coco_train_image_dir_path
        image_val_dir_path = args.coco_val_image_dir_path
        annotations_path = args.coco_karpathy_json_path
    elif dataset_name == "flickr":
        image_train_dir_path = (
            args.flickr_image_dir_path
        )  # Note: calling this "train" for consistency with COCO but Flickr only has one split for images
        image_val_dir_path = None
        annotations_path = args.flickr_karpathy_json_path
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_dataset = CaptionDataset(
        image_train_dir_path=image_train_dir_path,
        image_val_dir_path=image_val_dir_path,
        annotations_path=annotations_path,
        is_train=True,
        dataset_name=dataset_name if dataset_name != "nocaps" else "coco",
    )

    test_dataset = CaptionDataset(
        image_train_dir_path=image_train_dir_path,
        image_val_dir_path=image_val_dir_path,
        annotations_path=annotations_path,
        is_train=False,
        dataset_name=dataset_name,
    )

    effective_num_shots = utils.compute_effective_num_shots(num_shots, zero_shot_add_text_shots=args.zero_shot_add_text_shots)

    np.random.seed(seed)
    test_dataloader = utils.prepare_eval_samples(
        test_dataset,
        args.num_samples if args.num_samples > 0 else len(test_dataset),
        args.batch_size,
    )

    if args.rices:
        rices_dataset = RICES(
            train_dataset,
            eval_model.device,
            args.batch_size,
            cached_features=cached_features,
            vision_encoder_path=args.rices_vision_encoder_path,
            vision_encoder_pretrained=args.rices_vision_encoder_pretrained,
        )
    else:
        # subset of the training set to sample context images from
        query_set = utils.get_query_set(train_dataset, args.query_set_size)

    get_outputs_kwargs = {}
    utils.random_seed(seed, args.rank)
    predictions = defaultdict()
    for batch in tqdm(
        test_dataloader,
        desc=f"Running inference {dataset_name.upper()}",
        disable=args.rank != 0,
    ):
        if args.rices:
            batch_demo_samples = rices_dataset.find(batch["image"], effective_num_shots)
        else:
            batch_demo_samples = utils.sample_batch_demos_from_query_set(
                query_set, effective_num_shots, len(batch["image"])
            )
            
        if isinstance(eval_model, OpenFlamingoEvalModel) and not isinstance(eval_model, InternVLChatEvalModel):
            batch_images, batch_text = [], []
            for i in range(len(batch["image"])):
                if num_shots > 0:
                    context_images = [x["image"] for x in batch_demo_samples[i]]
                else:
                    context_images = []
                batch_images.append(context_images + [batch["image"][i]])

                context_text = "".join(
                    [
                        eval_model.get_caption_prompt(caption=x["caption"].strip()) + "\n"
                        for x in batch_demo_samples[i]
                    ]
                )

                # Keep the text but remove the image tags for the zero-shot case
                if num_shots == 0:
                    context_text = context_text.replace("<image>", "")

                batch_text.append(context_text + eval_model.get_caption_prompt())
        elif isinstance(eval_model, InternVLChatEvalModel):
            get_outputs_kwargs["IMG_CONTEXT_TOKEN"] = IMG_CONTEXT_TOKEN
            batch_images, batch_text = [], []
            
            from internvl.conversation import get_conv_template
            template = get_conv_template(eval_model.get_model().template)
            get_outputs_kwargs["eos_token"] = template.sep
            
            for i in range(len(batch["image"])):
                if num_shots > 0:
                    context_images = [x["image"] for x in batch_demo_samples[i]]
                else:
                    context_images = []
                # batch_images.append(context_images + [batch["image"][i]])
                
                pixel_values_all = [eval_model.load_image(image) for image in context_images + [batch["image"][i]]]
                patches_num = [len(_) for _ in pixel_values_all]
                pixel_values = torch.cat(pixel_values_all, dim=0)
                
                batch_images.append(pixel_values)
                    
                if args.chat_few_shot_style in ["multi", "multi_rounds_conversation"]:
                    # incontext samples
                    for xid, x in enumerate(batch_demo_samples[i]):
                        caption=x["caption"].strip()
                        # Keep the text but remove the image tags for the zero-shot case
                        if num_shots == 0:
                            image_tokens = ""
                        else:
                            image_tokens = \
                                IMG_START_TOKEN + \
                                IMG_CONTEXT_TOKEN * eval_model.get_model().num_image_token * patches_num[xid] \
                                + IMG_END_TOKEN + '\n'
                        question = image_tokens + 'Provide a one-sentence caption for the provided image.'
                        
                        template.append_message(template.roles[0], question)
                        template.append_message(template.roles[1], caption)
                    
                    # this sample
                    image_tokens = \
                        IMG_START_TOKEN + \
                        IMG_CONTEXT_TOKEN * eval_model.get_model().num_image_token * patches_num[-1] \
                        + IMG_END_TOKEN
                    question = image_tokens + '\n' + 'Provide a one-sentence caption for the provided image.'
                    
                    template.append_message(template.roles[0], question)
                    template.append_message(template.roles[1], None)
                    
                    batch_text.append(template.get_prompt())
                elif args.chat_few_shot_style in ["single", "single_round_conversation", "direct"]:
                    prompt = ""
                    
                    # incontext samples
                    for xid, x in enumerate(batch_demo_samples[i]):
                        if xid == 0:
                            prompt += "Here are some caption examples:\n"
                        
                        caption=x["caption"].strip()
                        # Keep the text but remove the image tags for the zero-shot case
                        if num_shots == 0:
                            image_tokens = ""
                        else:
                            image_tokens = \
                                IMG_START_TOKEN + \
                                IMG_CONTEXT_TOKEN * eval_model.get_model().num_image_token * patches_num[xid] \
                                + IMG_END_TOKEN + "\n"
                        prompt += f"Example {xid+1}:\n" + image_tokens
                        prompt += f"Caption: {caption}\n"
                        
                        if xid == len(batch_demo_samples[i])-1:
                            prompt += "Refer to these examples to perform the following task:\n"
                        
                    # this sample
                    image_tokens = \
                        IMG_START_TOKEN + \
                        IMG_CONTEXT_TOKEN * eval_model.get_model().num_image_token * patches_num[-1] \
                        + IMG_END_TOKEN
                    prompt += image_tokens + '\nProvide a one-sentence caption for the provided image.'
                    
                    if args.chat_few_shot_style == "direct":
                        batch_text.append(prompt + "Caption: ")
                    else:
                        template.append_message(template.roles[0], prompt)
                        template.append_message(template.roles[1], None)
                        batch_text.append(template.get_prompt())
                else:
                    raise ValueError(f"Unsupported chat_few_shot_style: {args.chat_few_shot_style}")
        else:
            raise NotImplementedError(f"Model {args.model} not implemented")
        
        if args.rank == 0 and args.verbose:
            print(batch_text[0])
        
        outputs = eval_model.get_outputs(
            batch_images=batch_images,
            batch_text=batch_text,
            min_generation_length=min_generation_length,
            max_generation_length=max_generation_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            **get_outputs_kwargs,
        )
        
        if args.rank == 0 and args.verbose:
            print(outputs[0])

        new_predictions = [
            postprocess_captioning_generation(out).replace('"', "") for out in outputs
        ]

        for i, sample_id in enumerate(batch["image_id"]):
            predictions[sample_id] = {
                "caption": new_predictions[i],
            }

    # all gather
    all_predictions = [None for _ in range(args.world_size)]
    torch.distributed.all_gather_object(all_predictions, predictions)  # list of dicts

    if args.rank != 0:
        return None

    all_predictions = {
        k: v for d in all_predictions for k, v in d.items()
    }  # merge dicts

    # save the predictions to a temporary file
    results_path = f"{dataset_name}results_{uuid.uuid4()}.json"

    with open(results_path, "w") as f:
        f.write(
            json.dumps(
                [
                    {"image_id": k, "caption": all_predictions[k]["caption"]}
                    for k in all_predictions
                ],
                indent=4,
            )
        )

    metrics = compute_cider(
        result_path=results_path,
        annotations_path=args.coco_annotations_json_path
        if dataset_name == "coco"
        else args.flickr_annotations_json_path,
    )

    # delete the temporary file
    os.remove(results_path)

    return metrics["CIDEr"] * 100.0


def evaluate_vqa(
    args: argparse.Namespace,
    eval_model,
    seed: int = 42,
    min_generation_length: int = 0,
    max_generation_length: int = 5,
    num_beams: int = 3,
    length_penalty: float = 0.0,
    num_shots: int = 8,
    dataset_name: str = "vqav2",
    cached_features=None,
):
    """
    Evaluate a model on VQA datasets. Currently supports VQA v2.0, OK-VQA, VizWiz and TextVQA.

    Args:
        args (argparse.Namespace): arguments
        eval_model (XXXEvalModel): model to evaluate
        seed (int, optional): random seed. Defaults to 42.
        max_generation_length (int, optional): max generation length. Defaults to 5.
        num_beams (int, optional): number of beams to use for beam search. Defaults to 3.
        length_penalty (float, optional): length penalty for beam search. Defaults to -2.0.
        num_shots (int, optional): number of shots to use. Defaults to 8.
        dataset_name (string): type of vqa dataset: currently supports vqav2, ok_vqa. Defaults to vqav2.
        cached_features (tensor, optional): cached demonstration features for RICES. Defaults to None.
    Returns:
        float: accuracy score
    """

    if dataset_name == "ok_vqa":
        train_image_dir_path = args.ok_vqa_train_image_dir_path
        train_questions_json_path = args.ok_vqa_train_questions_json_path
        train_annotations_json_path = args.ok_vqa_train_annotations_json_path
        test_image_dir_path = args.ok_vqa_test_image_dir_path
        test_questions_json_path = args.ok_vqa_test_questions_json_path
        test_annotations_json_path = args.ok_vqa_test_annotations_json_path
    elif dataset_name == "vqav2":
        train_image_dir_path = args.vqav2_train_image_dir_path
        train_questions_json_path = args.vqav2_train_questions_json_path
        train_annotations_json_path = args.vqav2_train_annotations_json_path
        test_image_dir_path = args.vqav2_test_image_dir_path
        test_questions_json_path = args.vqav2_test_questions_json_path
        test_annotations_json_path = args.vqav2_test_annotations_json_path
    elif dataset_name == "vizwiz":
        train_image_dir_path = args.vizwiz_train_image_dir_path
        train_questions_json_path = args.vizwiz_train_questions_json_path
        train_annotations_json_path = args.vizwiz_train_annotations_json_path
        test_image_dir_path = args.vizwiz_test_image_dir_path
        test_questions_json_path = args.vizwiz_test_questions_json_path
        test_annotations_json_path = args.vizwiz_test_annotations_json_path
    elif dataset_name == "textvqa":
        train_image_dir_path = args.textvqa_image_dir_path
        train_questions_json_path = args.textvqa_train_questions_json_path
        train_annotations_json_path = args.textvqa_train_annotations_json_path
        test_image_dir_path = args.textvqa_image_dir_path
        test_questions_json_path = args.textvqa_test_questions_json_path
        test_annotations_json_path = args.textvqa_test_annotations_json_path
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_dataset = VQADataset(
        image_dir_path=train_image_dir_path,
        question_path=train_questions_json_path,
        annotations_path=train_annotations_json_path,
        is_train=True,
        dataset_name=dataset_name,
    )

    test_dataset = VQADataset(
        image_dir_path=test_image_dir_path,
        question_path=test_questions_json_path,
        annotations_path=test_annotations_json_path,
        is_train=False,
        dataset_name=dataset_name,
    )

    effective_num_shots = utils.compute_effective_num_shots(num_shots, zero_shot_add_text_shots=args.zero_shot_add_text_shots)

    np.random.seed(seed)
    test_dataloader = utils.prepare_eval_samples(
        test_dataset,
        args.num_samples if args.num_samples > 0 else len(test_dataset),
        args.batch_size,
    )

    if args.rices:
        rices_dataset = RICES(
            train_dataset,
            eval_model.device,
            args.batch_size,
            cached_features=cached_features,
            vision_encoder_path=args.rices_vision_encoder_path,
            vision_encoder_pretrained=args.rices_vision_encoder_pretrained,
        )
    else:
        query_set = utils.get_query_set(train_dataset, args.query_set_size)

    get_outputs_kwargs = {}
    utils.random_seed(seed, args.rank)
    predictions = []
    for batch in tqdm(
        test_dataloader,
        desc=f"Running inference {dataset_name}",
        disable=args.rank != 0,
    ):
        if args.rices:
            batch_demo_samples = rices_dataset.find(batch["image"], effective_num_shots)
        else:
            batch_demo_samples = utils.sample_batch_demos_from_query_set(
                query_set, effective_num_shots, len(batch["image"])
            )

        if isinstance(eval_model, OpenFlamingoEvalModel) and not isinstance(eval_model, InternVLChatEvalModel):
            batch_images, batch_text = [], []
            for i in range(len(batch["image"])):
                if num_shots > 0:
                    context_images = [x["image"] for x in batch_demo_samples[i]]
                else:
                    context_images = []
                batch_images.append(context_images + [batch["image"][i]])

                context_text = "".join(
                    [
                        eval_model.get_vqa_prompt(
                            question=x["question"], answer=x["answers"][0]
                        )
                        + "\n"
                        for x in batch_demo_samples[i]
                    ]
                )

                # Keep the text but remove the image tags for the zero-shot case
                if num_shots == 0:
                    context_text = context_text.replace("<image>", "")

                batch_text.append(
                    context_text + eval_model.get_vqa_prompt(question=batch["question"][i])
                )
        elif isinstance(eval_model, InternVLChatEvalModel):
            base_prompt = 'Answer the question using a single word or phrase.'
            vizwiz_prompt = "When the provided information is insufficient, respond with 'Unanswerable'. "
            # infovqa_prompt = 'Answer the question directly.'
            infovqa_prompt = 'Answer the question using a single word or phrase.'
            ai2d_prompt = ''

            if 'vizwiz' in dataset_name:
                input_prompt = vizwiz_prompt + base_prompt
            elif 'ai2d' in dataset_name:
                input_prompt = ai2d_prompt
            elif 'infographicsvqa' in dataset_name:
                input_prompt = infovqa_prompt
            else:
                input_prompt = base_prompt
            
            get_outputs_kwargs["IMG_CONTEXT_TOKEN"] = IMG_CONTEXT_TOKEN
            batch_images, batch_text = [], []
            
            from internvl.conversation import get_conv_template
            template = get_conv_template(eval_model.get_model().template)
            get_outputs_kwargs["eos_token"] = template.sep
            
            for i in range(len(batch["image"])):
                if num_shots > 0:
                    context_images = [x["image"] for x in batch_demo_samples[i]]
                else:
                    context_images = []
                # batch_images.append(context_images + [batch["image"][i]])
                
                pixel_values_all = [eval_model.load_image(image) for image in context_images + [batch["image"][i]]]
                patches_num = [len(_) for _ in pixel_values_all]
                pixel_values = torch.cat(pixel_values_all, dim=0)
                
                batch_images.append(pixel_values)
                
                if args.chat_few_shot_style in ["multi", "multi_rounds_conversation"]:
                    # incontext samples
                    for xid, x in enumerate(batch_demo_samples[i]):
                        question, answer=x["question"], x["answers"][0]
                        # Keep the text but remove the image tags for the zero-shot case
                        if num_shots == 0:
                            image_tokens = ""
                        else:
                            image_tokens = \
                                IMG_START_TOKEN + \
                                IMG_CONTEXT_TOKEN * eval_model.get_model().num_image_token * patches_num[xid] \
                                + IMG_END_TOKEN + '\n'
                        question = image_tokens + question + '\n' + input_prompt
                        
                        template.append_message(template.roles[0], question)
                        template.append_message(template.roles[1], answer)
                    
                    # this sample
                    question = batch["question"][i]
                    image_tokens = \
                        IMG_START_TOKEN + \
                        IMG_CONTEXT_TOKEN * eval_model.get_model().num_image_token * patches_num[-1] \
                        + IMG_END_TOKEN
                    question = image_tokens + '\n' + question + '\n' + input_prompt
                    
                    template.append_message(template.roles[0], question)
                    template.append_message(template.roles[1], None)
                    
                    batch_text.append(template.get_prompt())
                elif args.chat_few_shot_style in ["single", "single_round_conversation", "direct"]:
                    prompt = ""
                    
                    # incontext samples
                    for xid, x in enumerate(batch_demo_samples[i]):
                        if xid == 0:
                            prompt += "Here is a collation of question and answer examples:\n"
                            
                        question, answer=x["question"], x["answers"][0]
                        # Keep the text but remove the image tags for the zero-shot case
                        if num_shots == 0:
                            image_tokens = ""
                        else:
                            image_tokens = \
                                IMG_START_TOKEN + \
                                IMG_CONTEXT_TOKEN * eval_model.get_model().num_image_token * patches_num[xid] \
                                + IMG_END_TOKEN + "\n"
                        prompt += f"Example {xid+1}:\n" + image_tokens
                        prompt += f"Question: {question}\nAnswer: {answer}\n"
                        
                        if xid == len(batch_demo_samples[i])-1:
                            prompt += "Refer to these examples to perform the following task:\n"
                    
                    # this sample
                    question = batch["question"][i]
                    image_tokens = \
                        IMG_START_TOKEN + \
                        IMG_CONTEXT_TOKEN * eval_model.get_model().num_image_token * patches_num[-1] \
                        + IMG_END_TOKEN
                    prompt += image_tokens + '\n' + f"Question: {question}\n" + input_prompt
                    
                    if args.chat_few_shot_style == "direct":
                        batch_text.append(prompt + "Answer: ")
                    else:
                        template.append_message(template.roles[0], prompt)
                        template.append_message(template.roles[1], None)
                        batch_text.append(template.get_prompt())
                else:
                    raise ValueError(f"Unsupported chat_few_shot_style: {args.chat_few_shot_style}")
        else:
            raise NotImplementedError(f"Model {args.model} not implemented")
        
        if args.rank == 0 and args.verbose:
            print(batch_text[0])

        outputs = eval_model.get_outputs(
            batch_images=batch_images,
            batch_text=batch_text,
            min_generation_length=min_generation_length,
            max_generation_length=max_generation_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            **get_outputs_kwargs,
        )

        process_function = (
            postprocess_ok_vqa_generation
            if dataset_name == "ok_vqa"
            else postprocess_vqa_generation
        )

        new_predictions = map(process_function, outputs)

        for new_prediction, sample_id in zip(new_predictions, batch["question_id"]):
            predictions.append({"answer": new_prediction, "question_id": sample_id})

    # all gather
    all_predictions = [None for _ in range(args.world_size)]
    torch.distributed.all_gather_object(all_predictions, predictions)  # list of lists

    if args.rank != 0:
        return None

    all_predictions = [
        item for sublist in all_predictions for item in sublist
    ]  # flatten

    # save the predictions to a temporary file
    random_uuid = str(uuid.uuid4())
    with open(f"{dataset_name}results_{random_uuid}.json", "w") as f:
        f.write(json.dumps(all_predictions, indent=4))

    if test_annotations_json_path is not None:
        acc = compute_vqa_accuracy(
            f"{dataset_name}results_{random_uuid}.json",
            test_questions_json_path,
            test_annotations_json_path,
        )
        # delete the temporary file
        os.remove(f"{dataset_name}results_{random_uuid}.json")

    else:
        print("No annotations provided, skipping accuracy computation.")
        acc = None
        if dataset_name == "vqav2":
            from fill_vqa_testdev_results import (
                fill_vqav2_test_json,
            )

            fill_fn = fill_vqav2_test_json
        elif dataset_name == "vizwiz":
            from fill_vqa_testdev_results import (
                fill_vizwiz_test_json,
            )

            fill_fn = fill_vizwiz_test_json
        else:
            print(
                "Temporary file saved to ", f"{dataset_name}results_{random_uuid}.json"
            )
            return

        fill_fn(
            f"{dataset_name}results_{random_uuid}.json",
            f"{dataset_name}-testdev_{eval_model.lm_name}_{num_shots}_{'rices' if args.rices else 'random'}_{seed}{'_dynamic' if eval_model.dynamic else ''}{'_2txtshots' if args.zero_shot_add_text_shots else ''}.json",
            args.vqav2_final_test_questions_json_path
            if dataset_name == "vqav2"
            else args.vizwiz_test_questions_json_path,
        )
        print(
            "Test-dev results saved to ",
            f"{dataset_name}-testdev_{eval_model.lm_name}_{num_shots}_{'rices' if args.rices else 'random'}_{seed}{'_dynamic' if eval_model.dynamic else ''}{'_2txtshots' if args.zero_shot_add_text_shots else ''}.json",
        )
        os.remove(f"{dataset_name}results_{random_uuid}.json")

    return acc


def evaluate_classification(
    args: argparse.Namespace,
    eval_model,
    seed: int = 42,
    num_shots: int = 8,
    dataset_name: str = "imagenet",
    cached_features=None,
    no_kv_caching=False,
    use_prompt_ensembling: bool = False,
):
    """
    Evaluate a model on classification dataset.

    Args:
        eval_model (XXXEvalModel): model to evaluate
        seed (int, optional): random seed. Defaults to 42.
        num_shots (int, optional): number of shots to use. Defaults to 8.
        no_kv_caching (bool): whether to disable key-value caching
        dataset_name (str, optional): dataset name. Defaults to "imagenet".
        cached_features (tensor, optional): cached demonstration features for RICES. Defaults to None.

    Returns:
        float: accuracy score
    """
    if args.model != "open_flamingo":
        raise NotImplementedError(
            "evaluate_classification is currently only supported for OpenFlamingo"
        )

    if dataset_name == "imagenet":
        train_dataset = ImageNetDataset(os.path.join(args.imagenet_root, "train"))
        test_dataset = ImageNetDataset(os.path.join(args.imagenet_root, "val"))
        prompt_fn = lambda x: eval_model.get_imagenet_prompt(label=x["class_name"])
        all_class_names = IMAGENET_CLASSNAMES
        k = 5
    elif dataset_name == "hateful_memes":
        train_dataset = HatefulMemesDataset(
            args.hateful_memes_image_dir_path,
            args.hateful_memes_train_annotations_json_path,
        )
        test_dataset = HatefulMemesDataset(
            args.hateful_memes_image_dir_path,
            args.hateful_memes_test_annotations_json_path,
        )
        prompt_fn = lambda x: eval_model.get_hateful_memes_prompt(
            text=x["ocr"], label=x["class_name"]
        )
        all_class_names = HM_CLASSNAMES
        k = 1
    else:
        raise ValueError(f"Unsupported dataset {dataset_name}")

    class_id_to_name = dict(zip(range(len(all_class_names)), all_class_names))

    effective_num_shots = utils.compute_effective_num_shots(num_shots, zero_shot_add_text_shots=args.zero_shot_add_text_shots)

    np.random.seed(seed)
    test_dataloader = utils.prepare_eval_samples(
        test_dataset,
        args.num_samples if args.num_samples > 0 else len(test_dataset),
        args.batch_size,
    )

    if args.rices:
        rices_dataset = RICES(
            train_dataset,
            eval_model.device,
            args.batch_size,
            cached_features=cached_features,
            vision_encoder_path=args.rices_vision_encoder_path,
            vision_encoder_pretrained=args.rices_vision_encoder_pretrained,
        )
    else:
        # subset of the training set to sample context images from
        query_set = utils.get_query_set(train_dataset, args.query_set_size)

    utils.random_seed(seed, args.rank)
    predictions = []
    for batch_idx, batch in tqdm(
        enumerate(test_dataloader),
        desc=f"Running inference {dataset_name}",
        disable=args.rank != 0,
    ):
        import ipdb; ipdb.set_trace
        if args.rices:
            batch_demo_samples = rices_dataset.find(batch["image"], effective_num_shots)
        else:
            batch_demo_samples = utils.sample_batch_demos_from_query_set(
                query_set, effective_num_shots, len(batch["image"])
            )

        # set up prompt ensembling
        num_permutations = (
            min(6, math.factorial(effective_num_shots)) if use_prompt_ensembling else 1
        )
        logprobs = []
        for _ in range(num_permutations):
            batch_images, batch_text = [], []
            for i in range(len(batch["image"])):
                if use_prompt_ensembling:
                    random.shuffle(batch_demo_samples[i])

                if effective_num_shots > 0:
                    context_images = [x["image"] for x in batch_demo_samples[i]]
                else:
                    context_images = []
                batch_images.append(context_images + [batch["image"][i]])

                context_text = "".join([prompt_fn(x) for x in batch_demo_samples[i]])

                # Keep the text but remove the image tags for the zero-shot case
                if num_shots == 0:
                    context_text = context_text.replace("<image>", "")

                batch_text.append(
                    context_text
                    + prompt_fn({"ocr": batch["ocr"][i], "class_name": None})
                )

            # get predicted class names
            logprobs.append(
                eval_model.get_rank_classifications(
                    batch_text,
                    batch_images,
                    all_class_names,
                    use_cache=(not no_kv_caching),
                    normalize_length=True,
                )
            )

        # ensemble logprobs together
        logprobs = torch.mean(torch.stack(logprobs, dim=-1), dim=-1)

        predicted_classnames, predicted_logprobs = utils.get_predicted_classnames(
            logprobs,
            k,
            class_id_to_name,
        )

        # compute accuracy
        for i, topk in enumerate(predicted_classnames):
            y_i = batch["class_name"][i]
            score = torch.exp(
                predicted_logprobs[i][0] - torch.logsumexp(logprobs[i], dim=0)
            ).item()
            predictions.append(
                {
                    "id": batch["id"][i],
                    "gt_label": y_i,
                    "pred_label": topk[0],
                    "pred_score": score,
                }
            )

    # all gather
    all_predictions = [None for _ in range(args.world_size)]
    torch.distributed.all_gather_object(all_predictions, predictions)  # list of lists
    if args.rank != 0:
        return

    all_predictions = [
        item for sublist in all_predictions for item in sublist
    ]  # flatten

    if dataset_name == "hateful_memes":
        # return ROC-AUC score
        greater_label = max(all_class_names)
        gts = [pred["gt_label"] for pred in all_predictions]
        pred_scores = [
            pred["pred_score"]
            if pred["pred_label"] == greater_label
            else 1 - pred["pred_score"]
            for pred in all_predictions
        ]
        return roc_auc_score(gts, pred_scores)
    else:
        # return top-1 accuracy
        acc1 = sum(
            int(pred["gt_label"] == pred["pred_label"]) for pred in all_predictions
        )
        return float(acc1) / len(all_predictions)


if __name__ == "__main__":
    main()
