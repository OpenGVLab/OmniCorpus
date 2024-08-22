import io
import os
import re
import json
import hashlib
import numpy as np
from PIL import Image
from copy import deepcopy
import pillow_avif  # pip install pillow-avif-plugin, NOTE importing this can make .avif readable
from scipy.optimize import linear_sum_assignment

import torch
import torch.distributed as dist
from torch.utils.data import Dataset
import transformers

from llava.constants import IMAGE_TOKEN_INDEX
from llava.mm_utils import expand2square

N_CHANNELS = 3


def preprocess_image(images, image_processor, image_aspect_ratio="pad"):
    out_images = []
    for image in images:
        if image_aspect_ratio == 'pad':
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
        try:
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        except Exception as err:
            print(image)
            print(image.shape)
            raise err
        
        image = image.unsqueeze(0)
        out_images.append(image)
    image = torch.cat(out_images, dim=0)
    # image = torchvision.transforms.RandomHorizontalFlip(p=0.5)(image)
    return image


class MMC4FormatDataset(Dataset):
    
    def __init__(self, 
                 data_path: str,
                 image_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args, 
                 safe_loading: bool = True):
        self.data_path = data_path
        self.image_path = image_path
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.max_num_images = getattr(self.data_args, "mmc4_max_num_images", 6)
        self.clip_processor = self.data_args.image_processor
        self.image_aspect_ratio = self.data_args.image_aspect_ratio
        self.max_tokens = tokenizer.model_max_length
        self.safe_loading = safe_loading
        
        self.internlm2_chat_style = getattr(self.data_args, "internlm2_chat_style", False)
        if self.internlm2_chat_style:
            print("Using internlm2_chat_style for formating the documents. (using <|im_start|> and <|im_end|>)")
        
        if not os.path.isdir(data_path):
            # not shard mode
            self.shard_mode = False
            self.data = json.load(open(data_path, "r"))
            self._length = len(self.data)
        else:
            self.shard_mode = True
            self.shards_length = json.load(open(os.path.join(data_path, "length.json"), "r"))
            self.shard_id_range = json.load(open(os.path.join(data_path, "shard_id_range.json"), "r"))
            self._length = sum(self.shards_length.values())
            self.check_shard_id_range(self.shard_id_range, self._length)
            first_shard_name = list(self.shards_length.keys())[0]
            self.current_shard_name = first_shard_name
            self.current_shard_data = json.load(open(os.path.join(data_path, first_shard_name), "r"))
        
    def __len__(self):
        return self._length
    
    @staticmethod
    def check_shard_id_range(shard_id_range, length):
        ids = []
        for start, end in shard_id_range.values():
            ids.extend(range(start, end))
        assert sorted(ids)[:length] == list(range(0, length))
        
    @staticmethod
    def encode_hash_sha256(web_url):
        hash_object = hashlib.sha256(web_url.encode())
        hex_dig = hash_object.hexdigest()
        return hex_dig
    
    def load_data(self, index):
        if not self.shard_mode:
            return self.data[index]
        else:
            start, end = self.shard_id_range[self.current_shard_name]
            if start <= index < end:  # NOTE
                return self.current_shard_data[index - start]
            else:
                for shard_name, (start, end) in self.shard_id_range.items():
                    if start <= index < end:  # NOTE
                        self.current_shard_name = shard_name
                        print(f"Change shard file to {self.current_shard_name}")
                        self.current_shard_data = json.load(open(os.path.join(self.data_path, shard_name), "r"))
                        return self.current_shard_data[index - start]
    
    def load_image(self, image_path_or_url):
        if "s3://" in self.image_path:
            # load from aws ceph
            if not hasattr(self, "ceph_client"):
                from petrel_client.client import Client
                self.ceph_client = Client("~/ceph_config.conf")
            return Image.open(io.BytesIO(self.ceph_client.get(image_path_or_url))).convert('RGB')
        else:
            # load from local (or s3mount node)
            return Image.open(image_path_or_url).convert("RGB")
        
    def __getitem__(self, index):
        try:
            item = self.getitem(index)
        except Exception as err:
            if self.safe_loading:
                print(err)
                index = (index + 1) % len(self)
                print(f"Try to load next index: {index}")
                item = self.__getitem__(index)
            else:
                raise RuntimeError(f"Error at index {index}") from err
        return item
    
    def getitem(self, index):
        # 'url', 'text_list', 'image_info', 'similarity_matrix', 'doc_loc', 'valid_image_ids'
        sample = self.load_data(index)
        
        sentences = sample["text_list"]
        image_info = sample["image_info"]
        sim_matrix = sample["similarity_matrix"]
        valid_image_ids = sample.get("valid_image_ids", list(range(len(image_info))))
        
        # filter with valid_image_ids
        valid_image_path, valid_sim_matrix = [], []
        for i in valid_image_ids:
            if "image_name" not in image_info[i]:
                assert "image_url" in image_info[i], image_info[i]
                image_name = self.encode_hash_sha256(image_info[i]["image_url"])
            else:
                image_name = image_info[i]["image_name"]
            valid_image_path.append(os.path.join(self.image_path, image_name))
            valid_sim_matrix.append(sim_matrix[i])
        valid_images = [self.load_image(p) for p in valid_image_path]
        sim_matrix = valid_sim_matrix
        
        sim_matrix = np.array(sim_matrix)  # of shape images x sentences
        image_indices, sentence_indices = linear_sum_assignment(-sim_matrix)
        
        images, sentence_ixs = [], []
        for i, sim_ix in zip(image_indices, sentence_indices):
            images.append(valid_images[i])
            sentence_ixs.append(sim_ix)

        # preprocess and pad images
        images_tensors = preprocess_image(images, self.clip_processor, self.image_aspect_ratio)
        keep_ixs = range(min(len(images_tensors), self.max_num_images))
        images_tensors = images_tensors[keep_ixs]
        sentence_ixs = [sentence_ixs[ix] for ix in keep_ixs]
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
        for ix in sentence_ixs:
            sentences[ix] = f"<image>{sentences[ix]}"
        text = " ".join(sentences)
        # the end
        if self.internlm2_chat_style:
            text = f"<|im_start|>{text}<|im_end|>"
        else:
            text = f"{text}{getattr(self, 'stop_word', self.tokenizer.eos_token)}"
        text = text.replace("<image> ", "<image>").replace(" <image>", "<image>")
        
        # text_tensor_ = tokenizer_image_token(text, tokenizer, return_tensors='pt')  # NOTE
        self.tokenizer.padding_side = "right"
        
        text_tensor = self.tokenizer(
            text,
            max_length=self.max_tokens,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        tokenized_image_id = self.tokenizer.additional_special_tokens_ids[
            self.tokenizer.additional_special_tokens.index("<image>")]
        # change the tokenized <image> id into IMAGE_TOKEN_INDEX of Llava
        text_tensor["input_ids"][text_tensor["input_ids"] == tokenized_image_id] = IMAGE_TOKEN_INDEX
        
        return (
            images_tensors,
            (text_tensor["input_ids"], text_tensor["attention_mask"]),
        )

    
class ObelicsFormatDataset(Dataset):
    
    def __init__(self, 
                 data_path: str,
                 image_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args, 
                 safe_loading: bool = True):
        self.data_path = data_path
        self.image_path = image_path
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.safe_loading = safe_loading
        self.max_num_images = getattr(self.data_args, "mmc4_max_num_images", 6)
        self.clip_processor = self.data_args.image_processor
        self.image_aspect_ratio = self.data_args.image_aspect_ratio
        self.max_tokens = tokenizer.model_max_length
        
        self.internlm2_chat_style = getattr(self.data_args, "internlm2_chat_style", False)
        if self.internlm2_chat_style:
            print("Using internlm2_chat_style for formating the documents. (using <|im_start|> and <|im_end|>)")
        
        if not os.path.isdir(data_path):
            # not shard mode
            self.shard_mode = False
            # self.data = json.load(open(data_path, "r"))
            if data_path.endswith(".json"):
                self.data = json.load(open(data_path, "r"))
            elif data_path.endswith(".jsonl"):
                with open(data_path, "r") as f:
                    self.data = [json.loads(line) for line in f]
            else:
                raise ValueError(f"Unknown file format: {first_shard_name}")
            self._length = len(self.data)
        else:
            self.shard_mode = True
            self.shards_length = json.load(open(os.path.join(data_path, "length.json"), "r"))
            self.shard_id_range = json.load(open(os.path.join(data_path, "shard_id_range.json"), "r"))
            self._length = sum(self.shards_length.values())
            self.check_shard_id_range(self.shard_id_range, self._length)
            for _shard_name, (start, end) in self.shard_id_range.items():
                if start == 0:
                    break
            # first_shard_name = list(self.shards_length.keys())[0]
            first_shard_name = _shard_name
            self.current_shard_name = first_shard_name
            print(f"Initialize shard file to {self.current_shard_name}")
            if first_shard_name.endswith(".json"):
                self.current_shard_data = json.load(open(os.path.join(self.data_path, first_shard_name), "r"))
            elif first_shard_name.endswith(".jsonl"):
                self.current_shard_data = [json.loads(line) for line in open(os.path.join(self.data_path, first_shard_name), "r")]
            else:
                raise ValueError(f"Unknown file format: {first_shard_name}")
        
    def __len__(self):
        return self._length
    
    @staticmethod
    def check_shard_id_range(shard_id_range, length):
        ids = []
        for start, end in shard_id_range.values():
            ids.extend(range(start, end))
        assert sorted(ids)[:length] == list(range(0, length))
    
    def load_data(self, index):
        if not self.shard_mode:
            return deepcopy(self.data[index])
        else:
            start, end = self.shard_id_range[self.current_shard_name]
            if start <= index < end:  # NOTE
                return deepcopy(self.current_shard_data[index - start])
            else:
                for shard_name, (start, end) in self.shard_id_range.items():
                    if start <= index < end:  # NOTE
                        self.current_shard_name = shard_name
                        if shard_name.endswith(".json"):
                            self.current_shard_data = json.load(open(os.path.join(self.data_path, shard_name), "r"))
                        elif shard_name.endswith(".jsonl"):
                            self.current_shard_data = [json.loads(line) for line in open(os.path.join(self.data_path, shard_name), "r")]
                        else:
                            raise ValueError(f"Unknown file format: {shard_name}")
                        print(f"Change shard file to {self.current_shard_name}")
                        return deepcopy(self.current_shard_data[index - start])
                    
    def get_img_filename(self, web_url, imgmeta):
        return self.encode_hash_sha256(web_url)
    
    def get_img_filepath(self, images, metadata, metasample=None):
        images = [os.path.join(self.image_path, self.get_img_filename(img, imgmeta)) 
                  for img, imgmeta in zip(images, metadata) if img is not None]
        return images
        
    @staticmethod
    def encode_hash_sha256(web_url):
        hash_object = hashlib.sha256(web_url.encode())
        hex_dig = hash_object.hexdigest()
        return hex_dig
    
    def load_image(self, image_path_or_url):
        try:
            if "s3://" in self.image_path:
                # load from aws ceph
                if not hasattr(self, "ceph_client"):
                    from petrel_client.client import Client
                    self.ceph_client = Client("~/ceph_config.conf")
                return Image.open(io.BytesIO(self.ceph_client.get(image_path_or_url))).convert('RGB')
            else:
                # load from local (or s3mount node)
                return Image.open(image_path_or_url).convert("RGB")
        except Exception as err:
            raise type(err)(f"Error loading image: {image_path_or_url}") from err
        
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
        try:
            item = self.getitem(index)
        except Exception as err:
            if self.safe_loading:
                print(err)
                index = (index + 1) % len(self)
                print(f"Try to load next index: {index}")
                item = self.__getitem__(index)
            else:
                raise RuntimeError(f"Error at index {index}") from err
        return item
        
    def getitem(self, index):
        # 'images', 'metadata', 'general_metadata', 'texts', 'doc_loc', 'valid_image'
        sample = self.load_data(index)
        
        # parse sample and check
        images, texts, metadata, valid_image = self.parse_sample(sample)
        
        # get valid images
        images = self.get_img_filepath(images, metadata, sample)
        if sum(valid_image) > self.max_num_images:
            true_count = 0
            for i in range(len(valid_image)):
                if valid_image[i] is True:
                    true_count += 1
                    if true_count > self.max_num_images:
                        valid_image[i] = False
        images = [self.load_image(img) for img, valid in zip(images, valid_image) if valid]
        assert len(images) > 0
                    
        # preprocess and pad images
        images_tensors = preprocess_image(images, self.clip_processor, self.image_aspect_ratio)
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
                    texts[i] = "<image>"
                image_idx += 1
        texts = [_ for _ in texts if _]
        text = " ".join(texts)
        # the end
        if self.internlm2_chat_style:
            text = f"<|im_start|>{text}<|im_end|>"
        else:
            text = f"{text}{getattr(self, 'stop_word', self.tokenizer.eos_token)}"
        text = text.replace("<image> ", "<image>").replace(" <image>", "<image>")
        repl = lambda match: ' '.join('<image>' for _ in range(match.group(0).count('<image>')))
        text = re.sub(r'(<image>)+', repl, text)
        
        self.tokenizer.padding_side = "right"
        text_tensor = self.tokenizer(
            text,
            max_length=self.max_tokens,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        tokenized_image_id = self.tokenizer.additional_special_tokens_ids[
            self.tokenizer.additional_special_tokens.index("<image>")]
        # change the tokenized <image> id into IMAGE_TOKEN_INDEX of Llava
        text_tensor["input_ids"][text_tensor["input_ids"] == tokenized_image_id] = IMAGE_TOKEN_INDEX
        
        return (
            images_tensors,
            (text_tensor["input_ids"], text_tensor["attention_mask"]),
        )
