import torch
from generate_lib.Video_CCAM.eval import load_decord
# from generate_lib.Video_CCAM.model import create_videoccam
from generate_lib.construct_prompt import construct_prompt
from generate_lib.constant import GENERATION_TEMPERATURE, GENERATION_TOP_P, SYSTEM_PROMPT, MAX_TOKENS
from tqdm import tqdm
from transformers import AutoModel
import json
import os
import logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def generate_response(model_name: str, 
                      queries: list,
                      total_frames: int,
                      output_dir: str):

    logging.info(f"Model: {model_name}")


    if model_name == "Video-CCAM-14B-v1.1":
        model = AutoModel.from_pretrained(
            './pretrained/Video-CCAM-14B-v1.1',
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            llm_name_or_path='./pretrained/Phi-3-medium-4k-instruct', 
            visual_encoder_name_or_path='./pretrained/siglip-so400m-patch14-384', 
        )
    elif model_name == "Video-CCAM-9B-v1.1":
        model = AutoModel.from_pretrained(
            './pretrained/Video-CCAM-9B-v1.1',
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            llm_name_or_path='./pretrained/Yi-1.5-9B-Chat', 
            visual_encoder_name_or_path='./pretrained/siglip-so400m-patch14-384', 
        )
    elif model_name == "Video-CCAM-4B-v1.1":
        model = AutoModel.from_pretrained(
            './pretrained/Video-CCAM-4B-v1.1',
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            llm_name_or_path='./pretrained/Phi-3-mini-4k-instruct', 
            visual_encoder_name_or_path='./pretrained/siglip-so400m-patch14-384', 
        )
    
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
        


        messages = [
            [
                {
                    'role': 'user',
                    'content': f'<video>\n{prompt}'
                }
            ]
        ]

        images = [
            load_decord(video_path, sample_type='uniform', num_frames=total_frames)
        ]

        response = model.chat(messages, images, max_new_tokens=MAX_TOKENS, do_sample=False)[0]


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
