import os
import io
import json
import copy
import time
import gradio as gr
import base64
from PIL import Image
from io import BytesIO
from argparse import Namespace
import llava.conversation as conversation_lib
from typing import Sequence
conversation_lib.default_conversation = conversation_lib.conv_templates["v1"]  # NOTE


data_root = "/mnt/petrelfs/share_data/liqingyun/lmm_baseline_llava/sft_data"
filepath = {
    "llava_v1_5_mix665k": [
        f"{data_root}/llava_v1_5_mix/llava_v1_5_mix665k.json", 
        f"{data_root}/llava_v1_5_mix/data_root",
    ],
    "llava_v1_5_nopuretext_mix625k": [
        f"{data_root}/llava_v1_5_mix/llava_v1_5_nopuretext_mix625k.json", 
        f"{data_root}/llava_v1_5_mix/data_root",
    ],
    "llava_2shots_okvqa9k": [
        f"{data_root}/llava_xshots_okvqa9k/llava_2shots_okvqa9k.json", 
        f"{data_root}/llava_xshots_okvqa9k/train2014", 
    ], 
    "llava_3shots_okvqa9k": [
        f"{data_root}/llava_xshots_okvqa9k/llava_3shots_okvqa9k.json", 
        f"{data_root}/llava_xshots_okvqa9k/train2014", 
    ],
    "llava_4shots_okvqa9k": [
        f"{data_root}/llava_xshots_okvqa9k/llava_4shots_okvqa9k.json", 
        f"{data_root}/llava_xshots_okvqa9k/train2014", 
    ], 
    "llava_5shots_aokvqa17k": [
        f"{data_root}/llava_xshots_aokvqa17k/llava_5shots_aokvqa17k.json", 
        f"{data_root}/llava_xshots_aokvqa17k/data_root"
    ],
    "llava_5shots_gqa72k": [
        f"{data_root}/llava_xshots_gqa72k/llava_5shots_gqa72k.json", 
        f"{data_root}/llava_xshots_gqa72k/data_root", 
    ],
    "llava_5shots_ocrvqa80k": [
        f"{data_root}/llava_xshots_ocrvqa80k/llava_5shots_ocrvqa80k.json", 
        f"{data_root}/llava_xshots_ocrvqa80k/data_root", 
    ],
    "llava_5shots_ocrvqa166k": [
        f"{data_root}/llava_xshots_ocrvqa/llava_5shots_ocrvqa166k.json", 
        f"{data_root}/llava_xshots_ocrvqa/images", 
    ],
    "llava_5shots_textcaps22k": [
        f"{data_root}/llava_xshots_textcaps22k/llava_5shots_textcaps22k.json", 
        f"{data_root}/llava_xshots_textcaps22k/data_root", 
    ],
    "llava_5shots_vqavii83k": [
        f"{data_root}/llava_xshots_vqavii83k/llava_5shots_vqavii83k.json", 
        f"{data_root}/llava_xshots_vqavii83k/data_root", 
    ],
}


def image_to_mdstring(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"![image](data:image/jpeg;base64,{img_str})"


class InContextTrainset:
    def __init__(self, ann_path, img_path):
        self.list_data_dict = json.load(open(ann_path, "r"))
        if img_path.endswith(".json"):
            self.incontext_image_path_fmt = "json"
            print(f"Loading images of in-context samples from {img_path}")
            _start_time = time.time()
            self.images = json.load(open(img_path, "r"))
            print(f"Successfully Loaded images of in-context samples from {img_path}, which takes {time.time() - _start_time}s")
        else:
            self.incontext_image_path_fmt = "folder"
            self.incontext_image_folder = img_path
            
    def preprocess_image(self, image):
        image = self.expand2square(image, tuple(int(x*255) for x in [0.48145466, 0.4578275, 0.40821073]))
        return image
            
    @staticmethod
    def expand2square(pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result
        
    def load_image(self, image):
        if self.incontext_image_path_fmt == "json":
            image_id = image
            base64_image = self.images[image_id]
            image_bytes = base64.b64decode(base64_image)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')  # may raise ValueError: Decompressed Data Too Large
        elif self.incontext_image_path_fmt == "folder":
            image_fname = image
            image_fpath = os.path.join(self.incontext_image_folder, image_fname)
            image = Image.open(image_fpath).convert('RGB')
        return image
            
    def __getitem__(self, i):
        try:
            sources = [self.list_data_dict[i]]
        except IndexError as err:
            print(i, len(self.list_data_dict))
            raise err
        if 'image' in sources[0]:
            all_images = []
            if 'incontext_images' in self.list_data_dict[i]:
                incontext_images = self.list_data_dict[i]['incontext_images']
                assert isinstance(incontext_images, Sequence) and not isinstance(incontext_images, str), "incontext_images should be `tuple` or `list` of images"
                all_images.extend(incontext_images)
            if not isinstance(self.list_data_dict[i]['image'], str):  # NOTE we allow both list of multiple images and single image str in this implementation
                all_images.extend(self.list_data_dict[i]['image'])
            else:
                all_images.append(self.list_data_dict[i]['image'])
            all_images = [self.preprocess_image(self.load_image(_img)) for _img in all_images]
            # >>> Check num images >>>
            num_images_1 = len(all_images)
            # <<< Check num images <<<
            if len(sources[0].get("incontext_conversations", [])) > 0:
                sources = copy.deepcopy(sources)
                incontext_conversations = sources[0].pop("incontext_conversations")
                conversations_without_incontext = sources[0].pop("conversations")
                sources[0]["conversations"] = incontext_conversations + conversations_without_incontext
                sources[0]["conversations_without_incontext"] = conversations_without_incontext
            
            _sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                Namespace(is_multimodal=True, mm_use_im_start_end=False))
        else:
            # NOTE: not support incontext for pure text data for now
            _sources = copy.deepcopy([e["conversations"] for e in sources])
            all_images = []
            
        conv = conversation_lib.default_conversation.copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        
        # Apply prompt templates
        conversations = []
        for i, source in enumerate(_sources):
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human  # NOTE
                source = source[1:]
            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["value"])
            conversations.append(conv.get_prompt())
        
        data = sources[0]
        text = conversations[0]
        return text, all_images, data
    
    def __len__(self):
        return len(self.list_data_dict)
    
    
def test_incontext_trainset():
    dataset = InContextTrainset(*filepath["llava_2shots_okvqa9k.json"])
    sample = dataset[66]
    

# Model Constants
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
def preprocess_multimodal(sources, data_args):
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources
    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                num_imags = sentence['value'].count(DEFAULT_IMAGE_TOKEN)
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                for _ in range(num_imags):
                    sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
    return sources
    
    
def process_v1conv_item(text, all_images, meta_data):
    md_str = ""
    assert text.count("<image>") == len(all_images), (text, text.count("<image>"), len(all_images))
    split_components = text.split("<image>")
    for i, c in enumerate(split_components):
        if i > 0:
            md_str += "\n" + image_to_mdstring(all_images[i - 1]) + "\n"
        c = c.replace("\n", "\n### ")
        md_str += f"### {c}"
        
    md_str = md_str.replace("USER:", "\n### `USER:`")
    md_str = md_str.replace("ASSISTANT:", "\n### `ASSISTANT:`")
    
    prompt_text = text.replace("\n", "\\n")
    md_str += "\n".join([
        "\n\n## META DATA DICT: ",
        "```", 
        f"prompt_text = {prompt_text}", f"len(all_images) = {len(all_images)}", "", "", 
        f"meta_data = {json.dumps(meta_data, indent=4)}", "", "", 
        f"md_string = {md_str}", 
        "```",  
    ]) 
    return md_str
    

def gradio_app_vis_incontext_trainset(_filepath):
    data, loaded_obj = None, {}
    
    def load_and_collate_annotations(ann_filename):
        ann_path, img_path = _filepath[ann_filename]
        dataset = InContextTrainset(ann_path, img_path)
        return dataset
    
    def when_btn_submit_click(ann_filename, ann_id, md_annotation):
        if ann_filename is None:
            return when_ann_filename_change(ann_filename, ann_id, md_annotation)
        nonlocal data
        try:
            item = data[int(max(min(ann_id, len(data) - 1), 0))]
        except IndexError as err:
            print(ann_id, len(data), int(max(min(ann_id, len(data) - 1), 0)))
            raise err
        md_annotation = process_v1conv_item(*item)
        return ann_filename, int(max(min(ann_id, len(data) - 1), 0)), md_annotation
    
    def when_btn_next_click(ann_filename, ann_id, md_annotation):
        return when_btn_submit_click(ann_filename, ann_id + 1, md_annotation)
    
    def when_ann_filename_change(ann_filename, ann_id, annotation):
        nonlocal data, loaded_obj
        if ann_filename not in _filepath:
            return ann_filename, ann_id, annotation
        obj = loaded_obj.get(ann_filename, None) 
        if obj is None:
            obj = loaded_obj[ann_filename] = load_and_collate_annotations(ann_filename)
        data = obj
        return when_btn_submit_click(ann_filename, 0, annotation)
        
    with gr.Blocks() as app:
        ann_filename = gr.Radio(list(_filepath.keys()), value=None)
        with gr.Row():
            ann_id = gr.Number(0)
            btn_next = gr.Button("Next")
            btn_submit = gr.Button("id跳转")
        annotation = gr.Markdown()
        
        all_components = [ann_filename, ann_id, annotation]
        ann_filename.change(when_ann_filename_change, all_components, all_components)
        btn_submit.click(when_btn_submit_click, all_components, all_components)
        btn_next.click(when_btn_next_click, all_components, all_components)
        
    # app.launch()
    app.launch(share=True, server_port=10055)
        

if __name__ == "__main__":
    gradio_app_vis_incontext_trainset(filepath)
