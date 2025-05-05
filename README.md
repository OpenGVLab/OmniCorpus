<p align="center">
  <h1 align="center">🐳 OmniCorpus: A Unified Multimodal Corpus of 10 Billion-Level Images Interleaved with Text</h1>
</p>

[Paper](https://openreview.net/pdf?id=kwqhn2VuG4) | 
[OmniCorpus-CC](https://huggingface.co/datasets/OpenGVLab/OmniCorpus-CC) | 
[OmniCorpus-YT](https://huggingface.co/datasets/OpenGVLab/OmniCorpus-YT) | 
[OmniCorpus-CW](https://openxlab.org.cn/datasets/Li-Qingyun/OmniCorpus-CW) | 
[OmniCorpus-CC-210M](https://huggingface.co/datasets/OpenGVLab/OmniCorpus-CC-210M) | 
[Model](https://huggingface.co/Qingyun/OmniCorpus-InternVL)

## News 🚀🚀🚀

- `2025/02/11`: 🎉 OmniCorpus is accepted by ICLR 2025!
- `2024/10/22`: We release all the processed 📄 documents on [🤗 Hugging Face](https://huggingface.co/collections/OpenGVLab/omnicorpus-6709d180dc8f500b508e195f) and [OpenDataLab](https://openxlab.org.cn/datasets/Li-Qingyun/OmniCorpus-CW) platforms.
- `2024/10/14`: We release a new 7B InternVL [model](https://huggingface.co/Qingyun/OmniCorpus-InternVL) pre-trained with OmniCorpus. See [here](https://github.com/OpenGVLab/OmniCorpus?tab=readme-ov-file#experimental-results) for updated results.
- `2024/08/30`: We release 210 million filtered documents with meta-annotations, i.e., [OmniCorpus-CC-210M](https://huggingface.co/datasets/OpenGVLab/OmniCorpus-CC-210M) on Hugging Face.
- `2024/08/23`: The code for interleaved image-text pre-training with OmniCorpus, along with scripts for few-shot evaluation, are available. The developed human-feedback filtering functions for English documents and enhanced mainbody extraction tools are also available.
- `2024/07/04`: 🔥 [InternVL2-Pro](https://internvl.github.io/blog/2024-07-02-InternVL-2.0/) is released. OmniCorpus provide the interleaved data used in Stage-1 training.
- `2024/06/13`: We introduce 🐳 OmniCorpus, a 10 billion-level image-text interleaved dataset. This dataset contains 8.6 billion images, 1,696 billion text tokens, and 2.2 billion documents!

## Introduction

OmniCorpus dataset is the largest multimodal dataset to date, which pushes the boundaries of scale and diversity by encompassing 8.6 billion images interleaved with 1,696 billion text tokens from diverse sources, significantly surpassing previous datasets.
This dataset demonstrates several advantages over its counterparts:

1. **Larger data scale:** Our dataset is 1.7 times larger in images and 12.5 times larger in texts compared to the previously largest multimodal dataset, LAION-5B, while maintaining excellent data quality.
2. **Richer data diversity:** Drawing from a broader range of data sources, our dataset is more diverse than other image-text interleaved datasets. It includes bilingual multimodal data in both Chinese and English, and encompasses text-centric and vision-centric documents extracted from common websites and video platforms.
3. **More flexible format:** The streaming data format of our dataset offers exceptional flexibility, allowing adaptation to various data structures, including pure text corpora, image-text pairs, and interleaved data formats.

<img width="578" alt="image" src="https://github.com/OpenGVLab/OmniCorpus/assets/47669167/641a6427-ba50-41e6-8634-8810113fd803">

The OmniCorpus contains three sections:

- **[OmniCorpus-CC](https://huggingface.co/datasets/OpenGVLab/OmniCorpus-CC)**: processed from dumps in Common Crawl from 2013 to Nov./Dec. 2023.
- **[OmniCorpus-CW](https://openxlab.org.cn/datasets/Li-Qingyun/OmniCorpus-CW)**: sourced from Chinese internet resources, will be availiable in OpenDataLab platform.
- **[OmniCorpus-YT](https://huggingface.co/datasets/OpenGVLab/OmniCorpus-YT)**: samples Youtube video frames as images and collects subtitles as texts.

The image-text interleaved documents are recommanded for the following usages:

- Pre-training multimodal large language model (MLLM): Recent MLLMs (such as Flamingo series, EMU series, IDEFICS series, MM1, Cambrian-1, and xGen-MM) have shown that image-text interleaved data aids multimodal in-context learning and maintains the capabilities of large language models during multimodal fine-tuning.
- Long text-image retrieval: We provide image-text similarities calculated with CLIP, which can convert the documents to image-text retrieval dataset with longer text. A retrieval model pre-trained on such data can retrieval images based on longer text, which can be used for multimodal RAG, converting pure text to multimodal sample, etc.
- Source for futher dataset research: Our data is large-scale, which can serve as the source for researches for data curation strategies. We provide many useful attributes as metadata for each document, which can enrich the filtering strategy and reduce the cost.
- ......

## Data Pipeline

Our data pipeline consists of five key stages: main body extraction, preliminary text filtering, document deduplication, image downloading \& filtering, and detailed text filtering. Each stage efficiently reduces the dataset to retain only high-quality data.
Please refer to our paper for more details about the data pipeline.

<img width="723" alt="image" src="https://github.com/OpenGVLab/OmniCorpus/assets/47669167/a6de8928-58fb-4ff4-8ef9-4bd90e9ada5f">

## Experimental Results

We conduct a series of experiments to evaluate the effectiveness of OmniCorpus. As shown in the table below, model trained on our dataset demonstrates superior performance on academic caption and vqa benchmarks.
Please refer to our paper for more experimental results.

<img width="735" alt="image" src="https://github.com/user-attachments/assets/d705f04e-1355-4a40-9b1a-9d6a9490f199">

## Contact

- Qingyun Li: 21b905003@stu.hit.edu.cn
- Zhe Chen: chenzhe98@smail.nju.edu.cn
- Weiyun Wang: wangweiyun@pjlab.org.cn
- Wenhai Wang: wangwenhai@pjlab.org.cn
- Shenglong Ye: yeshenglong@pjlab.org.cn

## License and Terms of Use
The OmniCorpus dataset is distributed under [the CC BY 4.0 License](https://creativecommons.org/licenses/by/4.0/). The open-source code is released under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). 

The Terms of Use (ToUs) have been developed based on widely accepted standards. By accessing or using this dataset, users acknowledge their responsibility to comply with all relevant legal, regulatory, and ethical standards.
- All users, whether from academia or industry, must comply with the ToUs outlined in the CC BY 4.0 License.
- Any derived datasets or models must acknowledge the use of the OmniCorpus dataset to maintain transparency.
- The OmniCorpus must not be used in any project involving sensitive content or harmful outcomes, including but not limited to political manipulation, hate speech generation, misinformation propagation, or tasks that perpetuate harmful stereotypes or biases. 
- The use of this dataset in any manner that violates rights, such as copyright infringement, privacy breaches, or misuse of sensitive information, is strictly prohibited.
- While we do not enforce jurisdiction-specific terms, we strongly recommend that users ensure compliance with applicable local laws and regulations.
- The use of specific subset must comply with the ToUs of the primary source. Specifically, the use of OmniCorpus-CC, OmniCorpus-CW, and OmniCorpus-YT must comply with [the Common Crawl ToUs](https://commoncrawl.org/terms-of-use), the [regulations](https://www.gov.cn/zhengce/content/202409/content\_6977766.htm) on the security management of Internet data in China, and [YouTube’s ToUs](https://www.youtube.com/terms), respectively.
- These ToUs do not supersede the ToUs of the original content sources. Users must ensure that any use of the dataset’s content complies with the original ToUs and the rights of the data subjects.

## Citation

```
@inproceedings{li2024omnicorpus,
  title={OmniCorpus: A Unified Multimodal Corpus of 10 Billion-Level Images Interleaved with Text},
  author={Li, Qingyun and Chen, Zhe and Wang, Weiyun and Wang, Wenhai and Ye, Shenglong and Jin, Zhenjiang and others},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025}
}
```
