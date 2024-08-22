"""
Preprocess and load datasets for training.
"""

import functools
import io
import json
import math
import re
import random
import hashlib
import argparse
from copy import deepcopy
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info
import torchvision
import webdataset as wds
from PIL import Image
import base64
import pillow_avif
from scipy.optimize import linear_sum_assignment
import transformers
import petrel_client
from open_flamingo.train.data_utils import *

Image.MAX_IMAGE_PIXELS = 1000000000
N_CHANNELS = 3
MIN_KB = 10
_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None
    
import warnings
warnings.filterwarnings("ignore", message="Palette images with Transparency expressed in bytes should be converted to RGBA images")


def preprocess_image(sample, image_processor):
    """
    Convert images to tensors for training.
    Augmentations: random horizontal flip.
    Normalization handled by wds.
    """
    image = [image_processor(s).unsqueeze(0) for s in sample]
    image = torch.cat(image, dim=0)
    image = torchvision.transforms.RandomHorizontalFlip(p=0.5)(image)
    return image


def filter_no_caption_or_no_image(sample):
    """
    Filter out LAION samples with no caption or no image.
    """
    return ("txt" in sample) and (
        "png" in sample or "jpg" in sample or "jpeg" in sample
    )


def preprocess_laion_text(sample, tokenizer, max_tokens=32, original=True):
    """
    Preprocess text for LAION.
    Captions are truncated to 32 tokens by default.
    """
    tokenizer.padding_side = "right"
    sample = [
        (f"<image>{s.strip()}<|endofchunk|>{tokenizer.eos_token}") for s in sample
    ]
    if original:
        text = tokenizer(
            sample,
            max_length=max_tokens,
            padding="longest",
            truncation="only_first",
            return_tensors="pt",
        )
    else:
        text = tokenizer(
            sample,
            max_length=max_tokens,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
    return text["input_ids"], text["attention_mask"]


def preprocess_gpt_interleaved(
    info, tokenizer, clip_processor, min_num_images, max_num_images, max_tokens=256
):
    """
    Preprocess a ChatGPT-generated image-text sequence.
    """
    text = info["example"]
    text = re.sub(r"_!_IMAGE\d+_!_", "<|endofchunk|><image>", text)

    # convert images from base64 to PIL
    images = []
    for image_key in range(1, len(info["image_map"]) + 1):
        image_base64 = info["image_map"][f"_!_IMAGE{image_key}_!_"]["base64_image"]
        rawbytes = base64.b64decode(image_base64)
        images.append(Image.open(io.BytesIO(rawbytes)).convert("RGB"))

    # preprocess and pad images
    images_tensors = preprocess_image(images, clip_processor)
    keep_ixs = range(min(len(images_tensors), max_num_images))
    images_tensors = images_tensors[keep_ixs]
    if len(images_tensors) < max_num_images:
        zero_padding = torch.zeros(
            (max_num_images - len(images_tensors), 3, 224, 224), dtype=torch.float
        )
        images_tensors = torch.cat((images_tensors, zero_padding), dim=0)

    # preprocess and tokenize text
    text = text.replace("<|endofchunk|>", "", 1)  # but remove first eoc
    # whitespace cleanup
    text = (
        text.replace(" <|endofchunk|>", "<|endofchunk|>")
        .replace("<image> ", "<image>")
        .replace(" <image>", "<image>")
    )

    indices = [m.start() for m in re.finditer("<image>", text)]
    if len(indices) > max_num_images:
        start_index = indices[max_num_images - 1]
        text = text[:start_index]

    text = f"{text}<|endofchunk|>{tokenizer.eos_token}"
    tokenizer.padding_side = "right"
    text_tensor = tokenizer(
        text,
        max_length=max_tokens,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    # reject sequences with too few images after truncation
    num_images = torch.count_nonzero(
        text_tensor["input_ids"]
        == tokenizer.additional_special_tokens_ids[
            tokenizer.additional_special_tokens.index("<image>")
        ]
    )
    if num_images < min_num_images:
        raise ValueError(f"Fewer than {min_num_images} images in sample")

    return (images_tensors, (text_tensor["input_ids"], text_tensor["attention_mask"]))


def preprocess_interleaved(
    sample,
    tokenizer,
    clip_processor,
    sim_threshold,
    min_num_images,
    max_num_images,
    img_url_prefix, 
    max_tokens=256,
):
    """
    Preprocess an interleaved image-text sequence, either by calling preprocess_gpt_interleaved (if the sequence
    is ChatGPT-generated) or by preprocessing in this function (if the sequences is from MMC4).
    """
    info = json.loads(sample[0])
    if "is_gpt" in info:
        return preprocess_gpt_interleaved(
            info, tokenizer, clip_processor, min_num_images, max_num_images, max_tokens
        )

    sentences = info["text_list"]
    sim_matrix = info["similarity_matrix"]

    # load images first to find which ones are valid
    valid_images, valid_image_indices = [], []
    for i, sample_image in enumerate(info["image_info"]):
        image_name = sample_image["image_name"]
        image_url = os.path.join(img_url_prefix, image_name)
        
        rawbytes = client.get(image_url)
        
        # filter to images >= 10KB
        if len(rawbytes) // 1000 <= MIN_KB:
            continue

        image = Image.open(io.BytesIO(rawbytes)).convert("RGB")
        valid_images.append(image)
        valid_image_indices.append(i)
        
    if len(valid_image_indices) == 0:
        raise ValueError("No images in sample")

    sim_matrix = np.array(sim_matrix)  # of shape images x sentences
    sim_matrix = sim_matrix[valid_image_indices]
    
    # negate the similarities to turn then into costs
    cost_matrix = -sim_matrix
    # find one to one assignements
    image_indices, sentence_indices = linear_sum_assignment(cost_matrix)

    images, sentence_ixs = [], []
    for i, sim_ix in zip(image_indices, sentence_indices):
        sim_score = sim_matrix[i][sim_ix]

        if sim_score < sim_threshold:
            continue

        images.append(valid_images[i])
        sentence_ixs.append(sim_ix)
        
    if len(images) == 0:
        raise ValueError("No images in sample")

    # preprocess and pad images
    images_tensors = preprocess_image(images, clip_processor)
    keep_ixs = range(min(len(images_tensors), max_num_images))
    images_tensors = images_tensors[keep_ixs]
    sentence_ixs = [sentence_ixs[ix] for ix in keep_ixs]
    if len(images_tensors) < max_num_images:
        zero_padding = torch.zeros(
            (
                max_num_images - len(images_tensors),
                N_CHANNELS,
                images_tensors[0].shape[1],
                images_tensors[0].shape[2],
            ),
            dtype=torch.float,
        )
        images_tensors = torch.cat((images_tensors, zero_padding), dim=0)

    # preprocess and tokenize text
    # add in <image> and <eoc> tokens
    for ix in sentence_ixs:
        sentences[ix] = f"<|endofchunk|><image>{sentences[ix]}"
    text = " ".join(sentences)
    text = text.replace("<|endofchunk|>", "", 1)  # but remove first eoc
    # whitespace cleanup
    text = (
        text.replace(" <|endofchunk|>", "<|endofchunk|>")
        .replace("<image> ", "<image>")
        .replace(" <image>", "<image>")
    )
    text = f"{text}<|endofchunk|>{tokenizer.eos_token}"
    tokenizer.padding_side = "right"
    text_tensor = tokenizer(
        text,
        max_length=max_tokens,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    # reject sequences with too few images (after truncation)
    num_images = torch.count_nonzero(
        text_tensor["input_ids"]
        == tokenizer.additional_special_tokens_ids[
            tokenizer.additional_special_tokens.index("<image>")
        ]
    )
    if num_images < min_num_images:
        raise ValueError(f"Fewer than {min_num_images} images in sample")
    elif (
        num_images == 1 and random.random() <= 0.5
    ):  # 50% chance of keeping single image samples
        raise ValueError("Only one image in sample")

    # avoid the situation where there's one <image> token and it's at the end
    if (
        num_images == 1
        and text_tensor["input_ids"][:, -1]
        == tokenizer.additional_special_tokens_ids[
            tokenizer.additional_special_tokens.index("<image>")
        ]
    ):
        raise ValueError(
            "Only one image at the end of sample, so labels will all be -100"
        )

    return (
        images_tensors,
        (text_tensor["input_ids"], text_tensor["attention_mask"]),
    )
    
    
def preprocess_interleaved_omnicorpus_w_similarities(
    sample,
    tokenizer,
    clip_processor,
    sim_threshold,
    min_num_images,
    max_num_images,
    img_url_prefix, 
    max_tokens=256,
):
    info = sample[0]
    if "is_gpt" in info:
        return preprocess_gpt_interleaved(
            info, tokenizer, clip_processor, min_num_images, max_num_images, max_tokens
        )

    sentences = [_[0] for _ in info["sentences"]]
    sim_matrix = [_[0] for _ in info["matrix"]]
    
    # load images first to find which ones are valid
    valid_images, valid_image_indices = [], []
    for i, sample_image in enumerate(info["images_info"]):
        image_name = encode_hash_sha256(sample_image["name"][0])
        image_url = os.path.join(img_url_prefix, image_name)
        
        rawbytes = client.get(image_url)
        
        # filter to images >= 10KB
        if len(rawbytes) // 1000 <= MIN_KB:
            continue

        image = Image.open(io.BytesIO(rawbytes)).convert("RGB")
        valid_images.append(image)
        valid_image_indices.append(i)
        
    if len(valid_image_indices) == 0:
        raise ValueError("No images in sample")

    sim_matrix = np.array(sim_matrix)  # of shape images x sentences
    sim_matrix = torch.from_numpy(sim_matrix).softmax(dim=-1).numpy()
    sim_matrix = sim_matrix[valid_image_indices]
    
    # negate the similarities to turn then into costs
    cost_matrix = -sim_matrix
    # find one to one assignements
    image_indices, sentence_indices = linear_sum_assignment(cost_matrix)

    images, sentence_ixs = [], []
    for i, sim_ix in zip(image_indices, sentence_indices):
        sim_score = sim_matrix[i][sim_ix]

        if sim_score < sim_threshold:
            continue

        images.append(valid_images[i])
        sentence_ixs.append(sim_ix)
        
    if len(images) == 0:
        raise ValueError("No images in sample")

    # preprocess and pad images
    images_tensors = preprocess_image(images, clip_processor)
    keep_ixs = range(min(len(images_tensors), max_num_images))
    images_tensors = images_tensors[keep_ixs]
    sentence_ixs = [sentence_ixs[ix] for ix in keep_ixs]
    if len(images_tensors) < max_num_images:
        zero_padding = torch.zeros(
            (
                max_num_images - len(images_tensors),
                N_CHANNELS,
                images_tensors[0].shape[1],
                images_tensors[0].shape[2],
            ),
            dtype=torch.float,
        )
        images_tensors = torch.cat((images_tensors, zero_padding), dim=0)

    # preprocess and tokenize text
    # add in <image> and <eoc> tokens
    for ix in sentence_ixs:
        sentences[ix] = f"<|endofchunk|><image>{sentences[ix]}"
    text = " ".join(sentences)
    text = text.replace("<|endofchunk|>", "", 1)  # but remove first eoc
    # whitespace cleanup
    text = (
        text.replace(" <|endofchunk|>", "<|endofchunk|>")
        .replace("<image> ", "<image>")
        .replace(" <image>", "<image>")
    )
    text = f"{text}<|endofchunk|>{tokenizer.eos_token}"
    tokenizer.padding_side = "right"
    text_tensor = tokenizer(
        text,
        max_length=max_tokens,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    # reject sequences with too few images (after truncation)
    num_images = torch.count_nonzero(
        text_tensor["input_ids"]
        == tokenizer.additional_special_tokens_ids[
            tokenizer.additional_special_tokens.index("<image>")
        ]
    )
    if num_images < min_num_images:
        raise ValueError(f"Fewer than {min_num_images} images in sample")
    elif (
        num_images == 1 and random.random() <= 0.5
    ):  # 50% chance of keeping single image samples
        raise ValueError("Only one image in sample")

    # avoid the situation where there's one <image> token and it's at the end
    if (
        num_images == 1
        and text_tensor["input_ids"][:, -1]
        == tokenizer.additional_special_tokens_ids[
            tokenizer.additional_special_tokens.index("<image>")
        ]
    ):
        raise ValueError(
            "Only one image at the end of sample, so labels will all be -100"
        )

    return (
        images_tensors,
        (text_tensor["input_ids"], text_tensor["attention_mask"]),
    )
    
    
def get_jsonl_zip_url_list(data_url_prefix):
    data_bucket = re.findall(r"(^.*?:?s3://.+?/)", data_url_prefix)[0]
    jsonl_zip_url_surfix_list = client.list(data_url_prefix)
    jsonl_zip_url_list = [os.path.join(data_url_prefix, s) for s in jsonl_zip_url_surfix_list if s.endswith(".jsonl.zip")]
    return jsonl_zip_url_list


def get_mmc4_dataset(args, image_processor, tokenizer, epoch=0, floor=False):
    """
    Initialize webdataset for MMC4 / ChatGPT sequences
    """
    resampled = getattr(args, "dataset_resampled", False)
    
    data_url_prefix = "s3://public-dataset/mmc4/ai2-jackh-mmc4-gated-public-41423/data/"
    img_url_prefix = "s3://public-dataset/mmc4/ai2-jackh-mmc4-gated-public-41423/images/"
    
    # num_samples, num_shards = get_dataset_size(input_shards)
    url_list = get_jsonl_zip_url_list(data_url_prefix)
    num_samples = None
    num_shards = len(url_list)

    num_samples = None
    if not num_samples:
        num_samples = args.train_num_samples_interleaved
        if not num_samples:
            raise RuntimeError(
                "Currently, number of dataset samples must be specified for training dataset. "
                "Please specify via `--train-num-samples` if no dataset length info present."
            )

    # create a shared epoch store to sync epoch to dataloader worker proc
    shared_epoch = SharedEpoch(epoch=epoch)
    if resampled:
        pipeline = [
            ResampledShards2(url_list, deterministic=True, epoch=shared_epoch)
        ]
    else:
        pipeline = [wds.SimpleShardList(url_list)]

    preprocess_fn = functools.partial(
        preprocess_interleaved,
        clip_processor=image_processor,
        tokenizer=tokenizer,
        sim_threshold=args.mmc4_textsim_threshold,
        min_num_images=args.mmc4_min_num_images,
        max_num_images=args.mmc4_max_num_images,
        img_url_prefix=img_url_prefix, 
    )
    
    # at this point we have an iterator over all the shards
    if not resampled:
        pipeline.extend(
            [
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ]
        )
    pipeline.extend(
        [
            # at this point, we have an iterator over the shards assigned to each worker at each node
            # wds.tarfile_to_samples(handler=log_and_continue),
            # tarfile_to_samples_nothrow,
            jsonlzip_to_samples_nothrow, 
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ]
    )

    pipeline.extend(
        [
            wds.to_tuple("json", handler=log_and_continue),
            wds.map(preprocess_fn, handler=log_and_continue),
            wds.batched(args.batch_size_interleaved, partial=False),
        ]
    )

    dataset = wds.DataPipeline(*pipeline)
    if not resampled:
        assert (
            num_shards >= args.workers * args.world_size
        ), "number of shards must be >= total workers"
    # roll over and repeat a few samples to get same number of full batches on each node
    round_fn = math.floor if floor else math.ceil
    global_batch_size = args.batch_size_interleaved * args.world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, args.workers)
    num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size
    # each worker is iterating over this
    dataset = dataset.with_epoch(num_worker_batches)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_omnicorpus_w_similarities_dataset(args, image_processor, tokenizer, epoch=0, floor=False):
    """
    Initialize webdataset for MMC4 / ChatGPT sequences
    """
    resampled = getattr(args, "dataset_resampled", False)
    data_url_prefix = "langchao:s3://liqingyun/projects/lmm_interleaved/omnicorpus_similarity"
    img_url_prefix = "wwhnew_pssd:s3://mllm-cc/raw-images/"
    
    # num_samples, num_shards = get_dataset_size(input_shards)
    url_list = [
        os.path.join(data_url_prefix, folder, fname)
        for folder in client.list(data_url_prefix)
        if folder.startswith("CC-MAIN-")
        for fname in client.list(os.path.join(data_url_prefix, folder))
        if fname.endswith(".jsonl")
    ]  # 1737
    num_samples = None
    num_shards = len(url_list)

    num_samples = None
    if not num_samples:
        num_samples = args.train_num_samples_interleaved
        if not num_samples:
            raise RuntimeError(
                "Currently, number of dataset samples must be specified for training dataset. "
                "Please specify via `--train-num-samples` if no dataset length info present."
            )

    # create a shared epoch store to sync epoch to dataloader worker proc
    shared_epoch = SharedEpoch(epoch=epoch)
    if resampled:
        pipeline = [
            ResampledShards2(url_list, deterministic=True, epoch=shared_epoch)
        ]
    else:
        pipeline = [wds.SimpleShardList(url_list)]

    preprocess_fn = functools.partial(
        preprocess_interleaved_omnicorpus_w_similarities,
        clip_processor=image_processor,
        tokenizer=tokenizer,
        sim_threshold=args.mmc4_textsim_threshold,
        min_num_images=args.mmc4_min_num_images,
        max_num_images=args.mmc4_max_num_images,
        img_url_prefix=img_url_prefix, 
    )
    
    # at this point we have an iterator over all the shards
    if not resampled:
        pipeline.extend(
            [
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ]
        )
    pipeline.extend(
        [
            # at this point, we have an iterator over the shards assigned to each worker at each node
            # wds.tarfile_to_samples(handler=log_and_continue),
            # tarfile_to_samples_nothrow,
            jsonl_to_samples_nothrow, 
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ]
    )

    pipeline.extend(
        [
            wds.to_tuple("json", handler=log_and_continue),
            wds.map(preprocess_fn, handler=log_and_continue),
            wds.batched(args.batch_size_interleaved, partial=False),
        ]
    )

    dataset = wds.DataPipeline(*pipeline)
    if not resampled:
        assert (
            num_shards >= args.workers * args.world_size
        ), "number of shards must be >= total workers"
    # roll over and repeat a few samples to get same number of full batches on each node
    round_fn = math.floor if floor else math.ceil
    global_batch_size = args.batch_size_interleaved * args.world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, args.workers)
    num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size
    # each worker is iterating over this
    dataset = dataset.with_epoch(num_worker_batches)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_laion_dataset(args, image_processor, tokenizer, epoch=0, floor=False):
    """
    Initialize webdataset for LAION data
    """
    input_shards = args.laion_shards
    assert input_shards is not None
    resampled = getattr(args, "dataset_resampled", False)

    num_samples, num_shards = get_dataset_size(input_shards)
    num_samples = None
    if not num_samples:
        num_samples = args.train_num_samples_laion
        if not num_samples:
            raise RuntimeError(
                "Currently, number of dataset samples must be specified for training dataset. "
                "Please specify via `--train-num-samples` if no dataset length info present."
            )

    # create a shared epoch store to sync epoch to dataloader worker proc
    shared_epoch = SharedEpoch(epoch=epoch)
    if resampled:
        pipeline = [
            ResampledShards2(input_shards, deterministic=True, epoch=shared_epoch)
        ]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # create two preprocess functions that take in the passed in image_processor and tokenizer
    preprocess_image_fn = functools.partial(
        preprocess_image, image_processor=image_processor
    )
    preprocess_text_fn = functools.partial(preprocess_laion_text, tokenizer=tokenizer)

    # at this point we have an iterator over all the shards
    if not resampled:
        pipeline.extend(
            [
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ]
        )
    pipeline.extend(
        [
            # at this point, we have an iterator over the shards assigned to each worker at each node
            # wds.tarfile_to_samples(handler=log_and_continue),
            tarfile_to_samples_nothrow,
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ]
    )

    pipeline.extend(
        [
            wds.select(filter_no_caption_or_no_image),
            wds.decode("pilrgb", handler=log_and_continue),
            wds.to_tuple("jpg;png;jpeg", "txt", handler=log_and_continue),
            wds.batched(args.batch_size_laion, partial=False),
            wds.map_tuple(
                preprocess_image_fn, preprocess_text_fn, handler=log_and_continue
            ),
        ]
    )

    dataset = wds.DataPipeline(*pipeline)
    if not resampled:
        assert (
            num_shards >= args.workers * args.world_size
        ), "number of shards must be >= total workers"
    # roll over and repeat a few samples to get same number of full batches on each node
    round_fn = math.floor if floor else math.ceil
    global_batch_size = args.batch_size_laion * args.world_size
    num_batches = round_fn(num_samples / global_batch_size)
    num_workers = max(1, args.workers)
    num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
    num_batches = num_worker_batches * num_workers
    num_samples = num_batches * global_batch_size
    # each worker is iterating over this
    dataset = dataset.with_epoch(num_worker_batches)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=True,
    )

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


class OBELICSDataset(Dataset):
    
    def __init__(self, 
                 data_args: argparse.Namespace,
                 tokenizer: transformers.PreTrainedTokenizer,
                 image_processor: transformers.CLIPImageProcessor):
        
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        
        self.data_path = data_args.obelics_data_path
        self.image_path = data_args.obelics_image_path
        
        self.max_num_images = getattr(self.data_args, "mmc4_max_num_images", 6)
        self.train_num_samples = getattr(self.data_args, "train_num_samples_interleaved", None)
        self.dataset_resampled = getattr(self.data_args, "dataset_resampled", True)
        
        self.safe_loading = True
        self.shard_mode = True
        self.verbose = False
        self.max_tokens = getattr(self.data_args, "max_tokens_interleaved", 256)  # 1024
        self.num_samples_each_shard = 70300  # even if the actual num is more
        self._length = self.num_samples_each_shard * 1999  # 981 is not avaliable
        
        self.seed = getattr(data_args, "seed", 42)
        self.random = random.Random(self.seed)
        self._shard_initialized = False
        if not getattr(data_args, "use_iterable_dataset", False):
            self.initialize_shard()
        
        # self.check_shard_id_range(self.shard_id_range, len(self))
        self.media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
        
        self.successful_loads = 0
        self.failure_loads = 0
        
    def initialize_shard(self):
        shard_order = [i for i in range(2000) if i != 981]
        data_args = self.data_args
        if getattr(data_args, "use_iterable_dataset", False):
            worker_info = get_worker_info()
            # self.print(f"worker_info: {worker_info}")
            shard_id = data_args.rank * max(1, worker_info.num_workers) + worker_info.id
            shard_num = max(1, worker_info.num_workers) * data_args.world_size
            shard_order = partition_for_rank(shard_order, shard_id, shard_num)
        else:
            _len = len(shard_order) // data_args.world_size * data_args.world_size
            if _len < len(shard_order):
                shard_order = self.random.sample(shard_order, _len)
            shard_order = partition_for_rank(shard_order, data_args.rank, data_args.world_size)
        if self.dataset_resampled:
            self.random.shuffle(shard_order)
        self.shard_order = shard_order
        
        # hard code a shard_id_range
        self.shard_id_range = {
            f"part-{shard_order[i]:06d}-1694cb95.jsonl": (
                self.num_samples_each_shard * i, 
                self.num_samples_each_shard * (i + 1) - 1
            ) 
            for i in range(len(shard_order))
        }
        
        self.current_shard_name = f"part-{shard_order[0]:06d}-1694cb95.jsonl"
        print(f"Initialize shard file to {self.current_shard_name}")
        self.current_shard_data = load_jsonl(os.path.join(self.data_path, self.current_shard_name), client)
        self.random.shuffle(self.current_shard_data)
        self._shard_initialized = True
        
    def print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def load_ann_file(self, file_path):
        if file_path.endswith(".json"):
            return load_json(file_path, client)
        elif file_path.endswith(".jsonl"):
            return load_jsonl(file_path, client)
        else:
            raise NotImplementedError(f"Unsupported annotation file format: {file_path}")
        
    def __len__(self):
        if getattr(self.data_args, "use_iterable_dataset"):
            shard_num = self.data_args.world_size * max(1, self.data_args.workers)
        else:
            shard_num = self.data_args.world_size
        if self.train_num_samples is not None:
            return min(self.train_num_samples, self._length) // shard_num
        return self._length // shard_num
    
    @staticmethod
    def check_shard_id_range(shard_id_range, length):
        ids = []
        for start, end in shard_id_range.values():
            ids.extend(range(start, end))
        assert sorted(ids)[:length] == list(range(0, length))
    
    def load_data(self, index):
        assert self.shard_mode
        if index >= self._length:
            index = index % self._length
        start, end = self.shard_id_range[self.current_shard_name]
        if start <= index <= end:
            return deepcopy(self.current_shard_data[index - start])
        else:
            for shard_name, (start, end) in self.shard_id_range.items():
                if start <= index <= end:
                    self.current_shard_name = shard_name
                    self.current_shard_data = self.load_ann_file(
                        os.path.join(self.data_path, shard_name))
                    self.random.shuffle(self.current_shard_data)
                    self.print(f"Change shard file to {self.current_shard_name}")
                    return deepcopy(self.current_shard_data[index - start])
                    
    def get_img_filename(self, web_url, imgmeta):
        return self.encode_hash_sha256(web_url)
        
    @staticmethod
    def encode_hash_sha256(web_url):
        hash_object = hashlib.sha256(web_url.encode())
        hex_dig = hash_object.hexdigest()
        return hex_dig
    
    def load_image(self, image_path_or_url):
        try:
            if "s3://" in self.image_path:
                # load from aws ceph
                # print(f"Successfully loading image: {image_path_or_url}")  # debug
                return Image.open(io.BytesIO(client.get(image_path_or_url))).convert('RGB')
            else:
                # load from local (or s3mount node)
                return Image.open(image_path_or_url).convert("RGB")
        except Exception as err:
            self.print(f"Error loading image: {image_path_or_url}: {err}")
            return None
        
    def parse_sample(self, sample):
        images = sample["images"]
        texts = sample["texts"]
        metadata = sample.get(
            "metadata",
            [
                {"filename": self.encode_hash_sha256(web_url)} 
                if web_url is not None else None
                for web_url in images
            ]
        )
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        assert isinstance(metadata, list), metadata
        valid_image = sample.get("valid_image", [True] * sum(img is not None for img in images))
        assert len(images) == len(texts)
        assert sum(img is not None for img in images) == sum(txt is None for txt in texts) == len(valid_image), (sum(img is not None for img in images), sum(txt in ["<image>", None] for txt in texts), len(valid_image), sample)
        for _img, _imgmeta in zip(images, metadata):
            assert( _img is None) == (_imgmeta is None), sample
        return images, texts, metadata, valid_image
    
    def __getitem__(self, index):
        # print(index)
        if not self._shard_initialized:
            self.initialize_shard()
        
        try:
            item = self.getitem(index)
            self.successful_loads += 1
        except Exception as err:
            if self.safe_loading:
                self.print(err)
                index = (index + 1) % len(self)
                self.print(f"Try to load next index (obelics): {index}")
                item = self.__getitem__(index)
            else:
                raise RuntimeError(f"Error at index {index}") from err
            self.failure_loads += 1
        return item
        
    def getitem(self, index):
        
        # print(f"1: {index}")
        
        # 'images', 'metadata', 'general_metadata', 'texts', 'doc_loc', 'valid_image'
        sample = self.load_data(index)
        
        # parse sample and check
        images, texts, metadata, valid_image = self.parse_sample(sample)
        
        # get valid images
        images = [os.path.join(self.image_path, self.get_img_filename(img, imgmeta)) for img, imgmeta in zip(images, metadata) if img is not None]
        
        loaded_images = []
        valid_count = 0
        for idx, (img, valid) in enumerate(zip(images, valid_image)):
            if valid:
                if valid_count >= self.max_num_images:
                    valid_image[idx] = False
                else:
                    _image = self.load_image(img)
                    if _image is not None:
                        loaded_images.append(_image)
                        valid_count += 1
                    else:
                        valid_image[idx] = False
        images = loaded_images
        
        # print(f"2: {len(images)}")
        
        assert len(images) > 0 and sum(valid_image)
                    
        # preprocess and pad images
        images_tensors = preprocess_image(images, self.image_processor)
        if len(images_tensors) < self.max_num_images:
            zero_padding = torch.zeros(
                (
                    self.max_num_images - len(images_tensors),
                    N_CHANNELS,
                    images_tensors[0].shape[1],
                    images_tensors[0].shape[2],
                ),
                dtype=torch.float,
            )
            images_tensors = torch.cat((images_tensors, zero_padding), dim=0)
        
        # preprocess and tokenize text
        # add in <image> and <eoc> tokens
        image_idx = 0
        for i in range(len(texts)):
            if texts[i] is None:
                if valid_image[image_idx]:
                    texts[i] = "<|endofchunk|><image>"
                image_idx += 1
        text = " ".join([_ for _ in texts if _])
        text = text.replace("<|endofchunk|>", "", 1)  # but remove first eoc
        # whitespace cleanup
        text = (
            text.replace(" <|endofchunk|>", "<|endofchunk|>")
            .replace("<image> ", "<image>")
            .replace(" <image>", "<image>")
        )
        
        # print(f"3: {text[:20]}")
        
        # the end
        text = f"{text}<|endofchunk|>{self.tokenizer.eos_token}"
        # # use white space to split contigious <image> tokens
        # repl = lambda match: ' '.join('<image>' for _ in range(match.group(0).count('<image>')))
        # text = re.sub(r'(<image>)+', repl, text)
        self.tokenizer.padding_side = "right"
        text_tensor = self.tokenizer(
            text,
            max_length=self.max_tokens,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        if self.media_token_id not in text_tensor["input_ids"]:
            raise ValueError("No <image> token in text")
        
        # print(f"4: {index}")
        
        return (
            images_tensors,
            (text_tensor["input_ids"], text_tensor["attention_mask"]),
        )
        

class OmniCorpusDataset(OBELICSDataset):
    
    def __init__(self, 
                 data_args: argparse.Namespace,
                 tokenizer: transformers.PreTrainedTokenizer,
                 image_processor: transformers.CLIPImageProcessor):
        
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        
        self.data_path = "omnicorpus_shuffled/"
        self.image_path = "s3://omnicorpus/raw-images/"
        
        self.max_num_images = getattr(self.data_args, "mmc4_max_num_images", 6)
        self.train_num_samples = getattr(self.data_args, "train_num_samples_interleaved", None)
        self.dataset_resampled = getattr(self.data_args, "dataset_resampled", True)
        
        # 0-6143 each 34194/34195/34196 samples
        self.safe_loading = True
        self.shard_mode = True
        self.verbose = False
        self.max_tokens = getattr(self.data_args, "max_tokens_interleaved", 256)  # 1024
        self.num_samples_each_shard = 34190  # even if the actual num is more
        self._length = self.num_samples_each_shard * 6144
        
        self.seed = getattr(data_args, "seed", 42)
        self.random = random.Random(self.seed)
        self._shard_initialized = False
        if not getattr(data_args, "use_iterable_dataset", False):
            self.initialize_shard()
        
        # self.check_shard_id_range(self.shard_id_range, len(self))
        self.media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
        
        self.successful_loads = 0
        self.failure_loads = 0
        
    def initialize_shard(self):
        shard_order = list(range(6144))
        data_args = self.data_args
        if getattr(data_args, "use_iterable_dataset", False):
            worker_info = get_worker_info()
            # self.print(f"worker_info: {worker_info}")
            shard_id = data_args.rank * max(1, worker_info.num_workers) + worker_info.id
            shard_num = max(1, worker_info.num_workers) * data_args.world_size
            shard_order = partition_for_rank(shard_order, shard_id, shard_num)
        else:
            shard_order = partition_for_rank(shard_order, data_args.rank, data_args.world_size)
        if self.dataset_resampled:
            self.random.shuffle(shard_order)
        self.shard_order = shard_order
        
        # hard code a shard_id_range
        self.shard_id_range = {
            f"omnicorpus_shuffled_shard_{shard_order[i]}.jsonl": (
                self.num_samples_each_shard * i, 
                self.num_samples_each_shard * (i + 1) - 1
            ) 
            for i in range(len(shard_order))
        }
        
        self.current_shard_name = f"omnicorpus_shuffled_shard_{shard_order[0]}.jsonl"
        self.print(f"Initialize shard file to {self.current_shard_name}")
        self.current_shard_data = load_jsonl(os.path.join(self.data_path, self.current_shard_name), client)
        self.random.shuffle(self.current_shard_data)
        self._shard_initialized = True
        
        

class LAIONEn2BDataset(Dataset):
    def __init__(self, 
                 data_args: argparse.Namespace,
                 tokenizer: transformers.PreTrainedTokenizer,
                 image_processor: transformers.CLIPImageProcessor):
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        
        # self.data_path = data_args.laionen2b_data_path
        # self.image_path = data_args.laionen2b_image_path
        data_path = "LaionEn"
        self.data_path = data_path
        self.image_path = "s3://LAION-5B/"
        
        self.train_num_samples = getattr(self.data_args, "train_num_samples_laion", None)
        self.dataset_resampled = getattr(self.data_args, "dataset_resampled", True)
        
        self.safe_loading = True
        self.shard_mode = True
        self.verbose = False
        self.max_tokens = getattr(self.data_args, "max_tokens_laion", 32)  # 128
        self.num_samples_each_shard = 10000
        self._length = 2007095753 
        
        self.seed = getattr(data_args, "seed", 42)
        self.random = random.Random(self.seed)
        self._shard_initialized = False
        if not getattr(data_args, "use_iterable_dataset", False):
            self.initialize_shard()
        
        self.successful_loads = 0
        self.failure_loads = 0
        
    def initialize_shard(self):
        data_args = self.data_args
        if getattr(data_args, "use_iterable_dataset", False):
            worker_info = get_worker_info()
            # self.print(f"worker_info: {worker_info}")
            shard_id = data_args.rank * max(1, worker_info.num_workers) + worker_info.id
            shard_num = max(1, worker_info.num_workers) * data_args.world_size
            shard_order = partition_for_rank(list(range(200709)), shard_id, shard_num)
        else:
            shard_order = partition_for_rank(list(range(200709 // data_args.world_size * data_args.world_size)), data_args.rank, data_args.world_size)
        if self.dataset_resampled:
            self.random.shuffle(shard_order)
        self.shard_order = shard_order
        
        # hard code a shard_id_range
        self.shard_id_range = {
            f"{shard_order[i]:07d}.txt": (
                self.num_samples_each_shard * i, 
                self.num_samples_each_shard * (i + 1) - 1
            ) 
            for i in range(len(shard_order))
        }
        
        self.current_shard_name = f"{shard_order[0]:07d}.txt"
        self.print(f"Initialize shard file to {self.current_shard_name}")
        self.current_shard_data = load_jsonl(os.path.join(self.data_path, self.current_shard_name), client)
        self.random.shuffle(self.current_shard_data)
        self._shard_initialized = True
        
    def print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)
        
    def __len__(self):
        if getattr(self.data_args, "use_iterable_dataset"):
            shard_num = self.data_args.world_size * max(1, self.data_args.workers)
        else:
            shard_num = self.data_args.world_size
        if self.train_num_samples is not None:
            return min(self.train_num_samples, self._length) // shard_num
        return self._length // shard_num
    
    def load_data(self, index):
        assert self.shard_mode
        if index >= self._length:
            index = index % self._length
        start, end = self.shard_id_range[self.current_shard_name]
        if start <= index <= end:
            return deepcopy(self.current_shard_data[index - start])
        else:
            for shard_name, (start, end) in self.shard_id_range.items():
                if start <= index <= end:
                    self.current_shard_name = shard_name
                    self.current_shard_data = load_jsonl(os.path.join(self.data_path, shard_name), client)
                    self.random.shuffle(self.current_shard_data)
                    self.print(f"Change shard file to {self.current_shard_name}")
                    return deepcopy(self.current_shard_data[index - start])
        raise IndexError(f"Index {index} out of range")

    def load_image(self, image_path_or_url):
        try:
            if "s3://" in self.image_path:
                # load from aws ceph
                # print(f"Successfully loading image: {image_path_or_url}")  # debug
                return Image.open(io.BytesIO(client.get(image_path_or_url))).convert('RGB')
            else:
                # load from local (or s3mount node)
                return Image.open(image_path_or_url).convert("RGB")
        except Exception as err:
            raise type(err)(f"Error loading image: {image_path_or_url}") from err
    
    def __getitem__(self, index):
        if not self._shard_initialized:
            self.initialize_shard()
            
        try:
            item = self.getitem(index)
            self.successful_loads += 1
        except Exception as err:
            if self.safe_loading:
                self.print(err)
                index = (index + 1) % len(self)
                self.print(f"Try to load next index (laionen2b): {index}")
                item = self.__getitem__(index)
            else:
                raise RuntimeError(f"Error at index {index}") from err
            self.failure_loads += 1
        return item
        
    def getitem(self, index):
        sample = self.load_data(index)
        image_surfix, caption = sample["image"], sample["caption"]
        image_url = os.path.join(self.image_path, image_surfix)
        image = self.load_image(image_url)
        images_tensors = preprocess_image([image], self.image_processor)
        input_ids, attention_mask = preprocess_laion_text([caption], self.tokenizer, self.max_tokens, original=False)
        
        return (
            images_tensors[0],
            (input_ids[0], attention_mask[0]),
        )
        
        
class SequentialIterableDatasetConvertor(IterableDataset):
    def __init__(self, dataset, data_args):
        print("Note: SequentialIterableDatasetConvertor is used")
        self.dataset = dataset
        self.data_args = data_args
        self.num_samples =self.data_args.train_num_samples
        
        self.verbose = True
        self.safe_loading = True
        assert self.dataset.safe_loading == self.safe_loading
        
        assert hasattr(self.dataset, "getitem")
        if not self.dataset.train_num_samples is None:
            self.dataset.train_num_samples = None
            print("Warning: train_num_samples is set to None")
    
    def __len__(self):
        return self.num_samples // getattr(self.data_args, "world_size", 1)
    
    def print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)
    
    @property
    def dataset_length(self):
        return len(self.dataset)
    
    def __iter__(self):
        if not self.dataset._shard_initialized:
            self.dataset.initialize_shard()  # we can also put this into worker_init_fn
            
        worker_info = get_worker_info()
        worker_id = worker_info.id
        num_workers = worker_info.num_workers
            
        self.error_num = 0
        self.current_index = 0
        while self.current_index < (self.num_samples // getattr(self.data_args, "world_size", 1) // num_workers):
            index = self.current_index % self.dataset_length
            try:
                yield self.dataset.getitem(index)
            except Exception as err:
                self.error_num += 1
                if self.safe_loading:
                    self.print(f"[{self.dataset.__class__}] Passing error when loading {self.current_index} ({index}) ({err}) (has passed {self.error_num})")
                else:
                    raise RuntimeError(f"Error at index {self.current_index} ({index})") from err
            self.current_index += 1
        
        
def get_mapdataset_datainfo(dataset, args, epoch, batch_size, sampler_type="distributed"):
    shared_epoch = SharedEpoch(epoch=epoch)  # NOTE useless here
    if sampler_type == "distributed":
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=args.world_size, rank=args.rank, 
            shuffle=getattr(args, "dataset_resampled", False), 
            seed=args.seed, drop_last=False
        )
    elif sampler_type == "sequential":
        sampler = JumpableSequentialSampler(dataset)
    else:
        raise NotImplementedError(f"Unsupported sampler type: {sampler_type}")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size, sampler=sampler, 
        num_workers=args.workers, pin_memory=True, 
        collate_fn=wds.filters.default_collation_fn, 
    )
    dataloader.num_batches = len(dataloader)
    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_iterabledataset_datainfo(dataset, args, epoch, batch_size):
    shared_epoch = SharedEpoch(epoch=epoch)  # NOTE useless here
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size, num_workers=args.workers, pin_memory=True, 
        collate_fn=wds.filters.default_collation_fn, 
    )
    dataloader.num_batches = len(dataloader)
    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


class ARGSNoneTrainNumSamples:
    def __init__(self, args, name="train_num_samples_interleaved"):
        self.args = args
        self.name = name
        self.train_num_samples = getattr(self.args, self.name, None)
        
    def __getattr__(self, name):
        if name == self.name:
            return None
        return getattr(self.args, name)


def get_obelics_dataset(args, image_processor, epoch, tokenizer):
    """
    Initialize Mapdataset for Obelics-format data
    """
    if getattr(args, "use_iterable_dataset", False):
        data_args = ARGSNoneTrainNumSamples(args, "train_num_samples_interleaved")
        dataset = OBELICSDataset(data_args, tokenizer, image_processor)
        dataset = SequentialIterableDatasetConvertor(dataset, data_args=data_args)
        return get_iterabledataset_datainfo(dataset, args, epoch, batch_size=args.batch_size_interleaved)
    else:
        dataset = OBELICSDataset(args, tokenizer, image_processor)
        return get_mapdataset_datainfo(dataset, args, epoch, batch_size=args.batch_size_interleaved, sampler_type="sequential")


def get_laionen2b_dataset(args, image_processor, epoch, tokenizer):
    if getattr(args, "use_iterable_dataset", False):
        data_args = ARGSNoneTrainNumSamples(args, "train_num_samples_laion")
        dataset = LAIONEn2BDataset(data_args, tokenizer, image_processor)
        dataset = SequentialIterableDatasetConvertor(dataset, data_args=data_args)
        return get_iterabledataset_datainfo(dataset, args, epoch, batch_size=args.batch_size_laion)
    else:
        dataset = LAIONEn2BDataset(args, tokenizer, image_processor)
        return get_mapdataset_datainfo(dataset, args, epoch, batch_size=args.batch_size_laion, sampler_type="sequential")


def get_omnicorpus_dataset(args, image_processor, epoch, tokenizer):
    if getattr(args, "use_iterable_dataset", False):
        data_args = ARGSNoneTrainNumSamples(args, "train_num_samples_interleaved")
        dataset = OmniCorpusDataset(data_args, tokenizer, image_processor)
        dataset = SequentialIterableDatasetConvertor(dataset, data_args=data_args)
        return get_iterabledataset_datainfo(dataset, args, epoch, batch_size=args.batch_size_interleaved)
    else:
        dataset = OmniCorpusDataset(args, tokenizer, image_processor)
        return get_mapdataset_datainfo(dataset, args, epoch, batch_size=args.batch_size_interleaved, sampler_type="sequential")


def get_dataset_fn(dataset_type):
    """
    Helper function to get the dataset function based on the dataset type
    """
    if dataset_type == "image_text":
        # return get_laion_dataset
        return get_laionen2b_dataset
    if dataset_type == "mmc4":
        return get_mmc4_dataset
    if dataset_type == "obelics":
        return get_obelics_dataset
    if dataset_type == "laionen2b":
        return get_laionen2b_dataset
    if dataset_type == "omnicorpus":
        return get_omnicorpus_dataset
    if dataset_type == "omnicorpus_w_similarities":
        return get_omnicorpus_w_similarities_dataset
    raise ValueError(f"Unsupported dataset type: {dataset_type}")


def get_data(args, image_processor, tokenizer, dataset_type, epoch=0):
    """
    Interface for getting the webdatasets
    """
    return get_dataset_fn(dataset_type)(
        args, image_processor=image_processor, epoch=epoch, tokenizer=tokenizer
    )
