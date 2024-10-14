# OmniCorpus

[[Paper](https://arxiv.org/abs/2406.08418)]
[[OmniCorpus-CC-210M](https://huggingface.co/datasets/OpenGVLab/OmniCorpus-CC-210M)]
[[OmniCorpus-CC](https://huggingface.co/datasets/OpenGVLab/OmniCorpus-CC)]
[[OmniCorpus-YT](https://huggingface.co/datasets/OpenGVLab/OmniCorpus-YT)]
[[OmniCorpus-CW]()]
[[Model](https://huggingface.co/Qingyun/OmniCorpus-InternVL)]

## News🚀🚀🚀

- `2024/06/13`: 🚀We introduce OmniCorpus, a 10 billion-level image-text interleaved dataset. This dataset contains 8.6 billion images, 1,696 billion text tokens, and 2.2 billion documents!

## Schedule

- [X] Release OmniCorpus-CC-200M
- [ ] Release OmniCorpus-CC
- [x] Release OmniCorpus-YT
- [ ] Release OmniCorpus-CW
- [X] Release HTML extraction toolkit
- [X] Release Human-Feedback filtering functions
- [X] Release code for interleaved image-text pre-training and few-shot evaluation toolkit

## Introduction

OmniCorpus dataset is the largest multimodal dataset to date, which pushes the boundaries of scale and diversity by encompassing 8.6 billion images interleaved with 1,696 text tokens from diverse sources, significantly surpassing previous datasets.
This dataset demonstrates several advantages over its counterparts:

1. **Larger data scale:** Our dataset is 1.7 times larger in images and 12.5 times larger in texts compared to the previously largest multimodal dataset, LAION-5B, while maintaining excellent data quality.
2. **Richer data diversity:** Drawing from a broader range of data sources, our dataset is more diverse than other image-text interleaved datasets. It includes bilingual multimodal data in both Chinese and English, and encompasses text-centric and vision-centric documents extracted from common websites and video platforms.
3. **More flexible format:** The streaming data format of our dataset offers exceptional flexibility, allowing adaptation to various data structures, including pure text corpora, image-text pairs, and interleaved data formats.

<img width="578" alt="image" src="https://github.com/OpenGVLab/OmniCorpus/assets/47669167/641a6427-ba50-41e6-8634-8810113fd803">

Some examples:

<img width="719" alt="image" src="https://github.com/OpenGVLab/OmniCorpus/assets/47669167/17759b19-a494-4b2b-a10d-a73359999141">

<img width="372" alt="image" src="https://github.com/OpenGVLab/OmniCorpus/assets/47669167/a4f02355-8d67-45ab-a4b1-957112b4b721">

<img width="374" alt="image" src="https://github.com/OpenGVLab/OmniCorpus/assets/47669167/2f85506c-f4db-40c8-bf45-9ea9230f22e8">

## Data Pipeline

Our data pipeline consists of five key stages: main body extraction, preliminary text filtering, document deduplication, image downloading \& filtering, and detailed text filtering. Each stage efficiently reduces the dataset to retain only high-quality data.
Please refer to our paper for more details about the data pipeline.

<img width="723" alt="image" src="https://github.com/OpenGVLab/OmniCorpus/assets/47669167/a6de8928-58fb-4ff4-8ef9-4bd90e9ada5f">

## Experimental Results

We conduct a series of experiments to evaluate the effectiveness of OmniCorpus. As shown in the table below, model trained on our dataset demonstrates superior performance on academic caption and vqa benchmarks.
Please refer to our paper for more experimental results.

<img width="735" alt="image" src="https://github.com/OpenGVLab/OmniCorpus/assets/47669167/a008f2d2-a8f0-484b-9b8b-d9c47e1067b4">
