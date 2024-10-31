from generate_lib.construct_prompt import construct_prompt
from generate_lib.constant import GENERATION_TEMPERATURE, GENERATION_TOP_P, SYSTEM_PROMPT, MAX_TOKENS, SYSTEM_PROMPT
import torch
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings("ignore")
from decord import VideoReader, cpu
from PIL import Image
import json
import logging
from tqdm import tqdm
import numpy as np
import os
import decord
from decord import VideoReader, cpu
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import PILToTensor
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
decord.bridge.set_bridge("torch")
import random
random.seed(42)

def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets

def load_video(video_path, num_segments=8, return_msg=False, resolution=224, hd_num=4, padding=False):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    num_frames = len(vr)
    frame_indices = get_index(num_frames, num_segments)
    
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.float().div(255.0)),
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.Normalize(mean, std)
    ])

    frames = vr.get_batch(frame_indices)
    frames = frames.permute(0, 3, 1, 2)
    frames = transform(frames)

    T_, C, H, W = frames.shape
        
    if return_msg:
        fps = float(vr.get_avg_fps())
        sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
        # " " should be added in the start and end
        msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
        return frames, msg
    else:
        return frames


def generate_response(model_name: str, 
                      queries: list,
                      total_frames: int,
                      output_dir: str):

    logging.info(f"Model: {model_name}")


    tokenizer =  AutoTokenizer.from_pretrained('./pretrained/InternVideo2-Chat-8B', trust_remote_code=True, use_fast=False)

    model = AutoModel.from_pretrained(
        './pretrained/InternVideo2-Chat-8B',
        torch_dtype=torch.bfloat16,
        trust_remote_code=True).cuda()

    sample_config = dict(do_sample='False')

    for query in tqdm(queries):
        id_ = query['id']
        video_path = os.path.join('videos', query['demonstration_type'], '.'.join([query['key'], 'mp4']))
        question = query['question']
        options = query['options']
        optionized_list = [f"{chr(65 + i)}. {option}" for i, option in enumerate(options)]
        gt = optionized_list[query['answer']]

        prompt, all_choices, index2ans = construct_prompt(question=question,
                                                          options=options,
                                                          num_frames=total_frames)

        video_tensor = load_video(video_path, num_segments=total_frames, return_msg=False)
        video_tensor = video_tensor.to(model.device)

        response = model.chat(tokenizer, 
                              SYSTEM_PROMPT, # sys prompt
                              prompt, # user prompt
                              media_type='video', 
                              media_tensor=video_tensor, 
                              generation_config=sample_config)

        with open(output_dir, "a") as f:
            f.write(json.dumps(
                {
                    "id": id_,
                    "question": question,
                    "response": response,
                    "all_choices": all_choices,
                    "index2ans": index2ans,
                    'gt': gt
                }
            ) + "\n")