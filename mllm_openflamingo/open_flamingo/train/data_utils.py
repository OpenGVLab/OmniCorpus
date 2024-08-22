"""
Util functions for initializing webdataset objects
"""

import ast
import json
import logging
import io
import os
import time
import base64
import random
import hashlib
import zipfile
import sys
from dataclasses import dataclass
from multiprocessing import Value

import braceexpand
import numpy as np
import webdataset as wds
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset, get_worker_info, Sampler
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import (
    base_plus_ext,
    tar_file_expander,
    url_opener,
    valid_sample,
)

from petrel_client.client import Client as PetrelClient
from petrel_client.version import version as petrel_client_version

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value("i", epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def get_dataset_size(shards):
    shards_list = list(braceexpand.braceexpand(shards))
    dir_path = os.path.dirname(shards[0])
    sizes_filename = os.path.join(dir_path, "sizes.json")
    len_filename = os.path.join(dir_path, "__len__")
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, "r"))
        total_size = sum(
            [
                int(sizes[os.path.basename(shard)])
                if os.path.basename(shard) in sizes
                else 0
                for shard in shards_list
            ]
        )
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, "r").read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


def group_by_keys_nothrow(
    data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None
):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if (
            current_sample is None
            or prefix != current_sample["__key__"]
            or suffix in current_sample
        ):
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


class detshuffle2(wds.PipelineStage):
    def __init__(
        self,
        bufsize=1000,
        initial=100,
        seed=0,
        epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.
        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls = wds.shardlists.expand_urls(urls)
        self.urls = urls
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch

        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            yield dict(url=self.rng.choice(self.urls))
            
            
def load_image(image_url_or_path, _client=None, convert_rgb: bool = True):
    if "s3://" in image_url_or_path:
        image = Image.open(io.BytesIO(_client.get(image_url_or_path)))
    else:
        image = Image.open(image_url_or_path)
    return image.convert("RGB") if convert_rgb else image


def load_jsonl_zip(jsonl_zip_url_or_path, _client=None):
    jsonl_fname = os.path.basename(jsonl_zip_url_or_path).replace(".zip", "")
    zipfile_input = io.BytesIO(_client.get(jsonl_zip_url_or_path)) if "s3://" in jsonl_zip_url_or_path else jsonl_zip_url_or_path
    with zipfile.ZipFile(zipfile_input) as zip_file:
        with zip_file.open(jsonl_fname, "r") as jsonl_file:
            jsonl_lines = jsonl_file.readlines()
    return jsonl_lines


def load_json(json_url_or_path, _client=None):
    if "s3://" in json_url_or_path:
        try_times = 0
        bytes = None
        while try_times < 10:
            try:
                bytes = _client.get(json_url_or_path)
                break
            except Exception as e:
                print(f"Failed to get {json_url_or_path}, retry {try_times}")
                try_times += 1
                time.sleep(1)
        return json.load(io.BytesIO(bytes))
    else:
        return json.load(open(json_url_or_path, "r"))
    

def load_json_line(line_str, try_times=20):
    meta_line_str = line_str
    _try_times = 0
    while _try_times < try_times:
        try:
            data = json.loads(line_str)
            break
        except Exception as e:
            data = None
            print(f"Failed to load line, retry {_try_times}")
            _try_times += 1
            line_str = line_str[:-1]
    if data is None:
        raise Exception(f"Failed to get {meta_line_str}")
    return data


def load_jsonl(jsonl_url_or_path, _client=None, return_lines=False):
    if "s3://" in jsonl_url_or_path:
        try_times = 0
        while try_times < 10:
            try:
                # bytes = _client.get(jsonl_url_or_path)
                # bytes = bytes.replace(b"\x00", b"\n")
                data_str = _client.get(jsonl_url_or_path).decode("utf-8")
                break
            except Exception as e:
                print(f"Failed to get {jsonl_url_or_path}, retry {try_times}")
                try_times += 1
                time.sleep(1)
        # lines = io.BytesIO(bytes).readlines()
        lines = io.StringIO(data_str).readlines()
    else:
        lines = open(jsonl_url_or_path, "r").readlines()
    
    if return_lines:
        return lines
        
    data = []
    for line in lines:
        if len(line.strip()) > 2:
            try:
                sample = load_json_line(line.strip())
                data.append(sample)
            except Exception as e:
                raise ValueError(f"Failed to load line: {jsonl_url_or_path} {line.strip()}") from e
    return data


def save_jsonl(data_list, jsonl_url_or_path, _client=None):
    jsonl_lines = [json.dumps(data, ensure_ascii=False) for data in data_list]
    if "s3://" in jsonl_url_or_path:
        try_times = 0
        while try_times < 10:
            try:
                with io.BytesIO("\n".join(jsonl_lines).encode("utf-8")) as f:
                    _client.put(jsonl_url_or_path, f)
                break
            except Exception as e:
                print(f"Failed to get {jsonl_url_or_path}, retry {try_times}")
                try_times += 1
                time.sleep(1)
    else:
        with open(jsonl_url_or_path, "w") as f:
            for line in jsonl_lines:
                f.write(line + '\n')


def encode_hash_sha256(txt):
    hash_object = hashlib.sha256(txt.encode())
    hex_dig = hash_object.hexdigest()
    return hex_dig


def load_image_from_base64(base64_str):
    img_data = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_data))
    return img


class Client(PetrelClient):
    @property
    def version(self):
        return petrel_client_version
    
    @property
    def is_v2_version(self):
        return petrel_client_version.startswith("v2")
    
    def list(self, *args, **kwargs):
        if self.is_v2_version:
            # convert v2 return type to v1
            for item in super().list(*args, **kwargs):
                yield os.path.basename(item.strip("/"))
        else:
            return super().list(*args, **kwargs)

client = Client("~/ceph_config.conf")


def partition_for_rank(all_rank_item_list: list, rank: int, world_num: int) -> list:
    this_rank_item_list = []
    this_rank_index = range(rank, len(all_rank_item_list), world_num)
    for idx in this_rank_index:
        this_rank_item_list.append(all_rank_item_list[idx])
    return this_rank_item_list


def get_slurm_rank_info():
    world_num = int(os.environ.get('SLURM_NTASKS', 1))
    rank = int(os.environ.get('SLURM_PROCID', 0))
    return rank, world_num


def jsonl_zip_url_opener(data, handler):
    for sample in data:
        assert isinstance(sample, dict), sample
        assert "url" in sample
        url = sample["url"]
        try:
            jsonl_lines = load_jsonl_zip(url, client)
            sample.update(stream=jsonl_lines)
            yield sample
        except Exception as exn:
            exn.args = exn.args + (url,)
            if handler(exn):
                continue
            else:
                break
            
            
def jsonl_url_opener(data, handler):
    for sample in data:
        assert isinstance(sample, dict), sample
        assert "url" in sample
        url = sample["url"]
        try:
            jsonl_lines = load_jsonl(url, client)
            sample.update(stream=jsonl_lines)
            yield sample
        except Exception as exn:
            exn.args = exn.args + (url,)
            if handler(exn):
                continue
            else:
                break
            
            
def jsonl_expander(data, handler):
    for source in data:
        url = source["url"]
        try:
            assert isinstance(source, dict)
            assert "stream" in source
            for line in source["stream"]:
                sample = dict(data=line)
                assert (
                    isinstance(sample, dict) and "data" in sample
                )
                sample["__url__"] = url
                yield sample
        except Exception as exn:
            exn.args = exn.args + (source.get("stream"), source.get("url"))
            if handler(exn):
                continue
            else:
                break
            
            
def group_by_keys_nothrow2(data, handler=None):
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        # fname, value = filesample["fname"], filesample["data"]
        # prefix, suffix = base_plus_ext(fname)
        # if prefix is None:
            # continue
        # suffix = suffix.lower()  # lcase=True
        suffix = "json"
        if valid_sample(current_sample):
            yield current_sample
        current_sample = dict(__url__=filesample["__url__"])
        current_sample[suffix] = filesample["data"]
    if valid_sample(current_sample):
        yield current_sample


def jsonlzip_to_samples_nothrow(src, handler=log_and_continue):
    streams = jsonl_zip_url_opener(src, handler=handler)
    files = jsonl_expander(streams, handler=handler)
    samples = group_by_keys_nothrow2(files, handler=handler)
    return samples


def jsonl_to_samples_nothrow(src, handler=log_and_continue):
    streams = jsonl_url_opener(src, handler=handler)
    files = jsonl_expander(streams, handler=handler)
    samples = group_by_keys_nothrow2(files, handler=handler)
    return samples


class JumpableSequentialSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        self.jump_steps = 0

    def jump(self, n_steps):
        """设置跳过的steps数"""
        print(f"Jumping {n_steps} steps")
        self.jump_steps = n_steps

    def __iter__(self):
        # 从跳过的steps后开始迭代，但对外表现如同从0开始
        start = self.jump_steps
        self.jump_steps = 0
        return iter(range(start, start + len(self.data_source)))

    def __len__(self):
        return len(self.data_source)
    