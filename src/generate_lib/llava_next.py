from operator import attrgetter
from generate_lib.LLaVA_NeXT.llava.model.builder import load_pretrained_model
from generate_lib.LLaVA_NeXT.llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from generate_lib.LLaVA_NeXT.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from generate_lib.LLaVA_NeXT.llava.conversation import conv_templates, SeparatorStyle
from generate_lib.construct_prompt import construct_prompt
from generate_lib.constant import GENERATION_TEMPERATURE, GENERATION_TOP_P, SYSTEM_PROMPT, MAX_TOKENS
import torch
import cv2
import numpy as np
from PIL import Image
import requests
import copy
import warnings
from decord import VideoReader, cpu
import torch
from tqdm import tqdm
import os
import json
import logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

warnings.filterwarnings("ignore")
import random 
random.seed(42)


def load_video(video_path, max_frames_num):
    if type(video_path) == str:
        vr = VideoReader(video_path, ctx=cpu(0))
    else:
        vr = VideoReader(video_path[0], ctx=cpu(0))
    total_frame_num = len(vr)

    if max_frames_num == 1:
        uniform_sampled_frames = np.array([np.random.choice(range(total_frame_num))])
    else:
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)

    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames  # (frames, height, width, channels)


def generate_response(model_name: str, 
                      queries: list,
                      total_frames: int,
                      output_dir: str):
    
    logging.info(f"Model: {model_name}")

    pretrained = f"./pretrained/{model_name}"
    llm_name = "llava_qwen"
    device = "cuda"
    device_map = "auto"
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, llm_name, device_map=device_map, attn_implementation="sdpa")
    model.eval()


    for query in tqdm(queries):
        id_ = query['id']
        video_path = os.path.join('videos', query['demonstration_type'], '.'.join([query['key'], 'mp4']))
        question = query['question']
        options = query['options']
        optionized_list = [f"{chr(65 + i)}. {option}" for i, option in enumerate(options)]
        gt = optionized_list[query['answer']]


        video_frames = load_video(video_path=video_path, 
                                  max_frames_num=total_frames)

        image_tensors = []
        frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()
        image_tensors.append(frames)

        conv_template = "qwen_1_5"
        prompt, all_choices, index2ans = construct_prompt(question=question,
                                                          options=options,
                                                          num_frames=total_frames)


        prompt = f"{DEFAULT_IMAGE_TOKEN}\n{prompt}" 

        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], prompt) 
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_question, 
                                          tokenizer, 
                                          IMAGE_TOKEN_INDEX, 
                                          return_tensors="pt").unsqueeze(0).to(device)
        
        image_sizes = [frame.size for frame in video_frames]

        cont = model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,
            do_sample=False,
            max_new_tokens=MAX_TOKENS,
            modalities=["video"],
        )
        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
        response = text_outputs[0]
        
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
