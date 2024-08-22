import os
from tqdm import tqdm

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import open_clip
from open_flamingo.eval.utils import custom_collate_fn


class RICES:
    def __init__(
        self,
        dataset,
        device,
        batch_size,
        vision_encoder_path="ViT-B-32",
        vision_encoder_pretrained="openai",
        cached_features=None,
    ):
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size

        # Load the model and processor
        vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
            vision_encoder_path,
            pretrained=vision_encoder_pretrained,
        )
        self.model = vision_encoder.to(self.device)
        self.image_processor = image_processor

        # Precompute features
        if cached_features is None:
            self.features = self._precompute_features()
        else:
            self.features = cached_features

    def _precompute_features(self):
        features = []

        # Switch to evaluation mode
        self.model.eval()

        # Set up loader
        loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            collate_fn=custom_collate_fn,
        )

        with torch.no_grad():
            for batch in tqdm(
                loader,
                desc="Precomputing features for RICES",
            ):
                batch = batch["image"]
                inputs = torch.stack(
                    [self.image_processor(image) for image in batch]
                ).to(self.device)
                image_features = self.model.encode_image(inputs)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features.detach())

        features = torch.cat(features)
        return features

    def find(self, batch, num_examples):
        """
        Get the top num_examples most similar examples to the images.
        """
        # Switch to evaluation mode
        self.model.eval()

        with torch.no_grad():
            inputs = torch.stack([self.image_processor(image) for image in batch]).to(
                self.device
            )

            # Get the feature of the input image
            query_feature = self.model.encode_image(inputs)
            query_feature /= query_feature.norm(dim=-1, keepdim=True)
            query_feature = query_feature.detach().cpu()

            if query_feature.ndim == 1:
                query_feature = query_feature.unsqueeze(0)

            # Compute the similarity of the input image to the precomputed features
            similarity = (query_feature @ self.features.T).squeeze()

            if similarity.ndim == 1:
                similarity = similarity.unsqueeze(0)

            # Get the indices of the 'num_examples' most similar images
            # NOTE: `stable=True` is for samples in self.dataset with the same image.
            indices = similarity.argsort(dim=-1, descending=True, stable=True)[:, :num_examples]

        # Return with the most similar images last
        return [[self.dataset[i] for i in reversed(row)] for row in indices]
    

# >>> New functions added by LQY >>>
class CustomRICES:
    def __init__(
        self,
        cached_features_path=None,
        data_list=None, 
        batch_size=128,
        device="cuda",
        vision_encoder_path="ViT-B-32",
        vision_encoder_pretrained="openai",
    ):
        self.device = device
        self.batch_size = batch_size

        # Load the model and processor
        vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
            vision_encoder_path,
            pretrained=vision_encoder_pretrained,
        )
        self.model = vision_encoder.to(self.device)
        self.image_processor = image_processor
        
        self.cached_features_path = cached_features_path
        self.data_list = data_list
        
    def prepare_cached_feature(self):
        self.cached_features = None
        cached_features_path = self.cached_features_path
        if cached_features_path is not None and os.path.exists(cached_features_path):
            self.cached_features = torch.load(cached_features_path, map_location="cpu")
        else:
            if cached_features_path in [None, ""]:
                cached_features_path = "./cached_features.pkl"
            assert self.data_list is not None, "Please provide `data_list` if `cached_features` is not provided."
            self.cached_features = self._precompute_features(self.data_list)
            torch.save(self.cached_features, cached_features_path)
        return {item["image_id"]:item for item in self.cached_features}
            
    def _precompute_features(self, data_list):
        dataset = DatasetForCustomRICES(data_list)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=custom_collate_fn,
        )
        
        cached_features_custom_fmt = []
        
        with torch.no_grad():
            for batch in tqdm(
                loader,
                desc="Precomputing features for RICES",
            ):
                batch_image = batch["image"]
                batch_image_path = batch["image_path"]
                batch_image_id = batch["image_id"]
                
                inputs = torch.stack(
                    [self.image_processor(image) for image in batch_image]
                ).to(self.device)
                image_features = self.model.encode_image(inputs)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                for feat, img_path, img_id in zip(image_features, batch_image_path, batch_image_id):
                    cached_features_custom_fmt.append(
                        dict(
                            image_id=img_id,
                            image_path=img_path,
                            features=feat.detach().cpu()
                        )
                    )
        return cached_features_custom_fmt
        

class DatasetForCustomRICES:
    def __init__(self, data_list):
        for sample in data_list:
            assert "image_path" in sample and "image_id" in sample, sample
        self.data_list = data_list
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, i):
        data = self.data_list[i]
        return dict(**data, image=Image.open(data['image_path']).convert('RGB'))
