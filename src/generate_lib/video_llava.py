import os
import sys
import json
import logging
from tqdm import tqdm

import torch

from generate_lib.construct_prompt import construct_prompt
from generate_lib.constant import GENERATION_TEMPERATURE, MAX_TOKENS


sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Video-LLaVA'))


from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria




def generate_response(model_name: str, 
                      queries: list,
                      total_frames: int,
                      output_dir: str):

    logging.info(f"Model: {model_name}")
    logging.info(f"Model {model_name} has a default total frame number of 8.")

    disable_torch_init()
    model_path = os.path.join("./pretrained", model_name)
    
    load_4bit, load_8bit = False, False      # need to fix llava_arch.py LlavaMetaForCausalLM encode_videos to use non-quantized/8 bit

    device = 'cuda'
    cache_dir = 'cache_dir'

    tokenizer, model, processor, _ = load_pretrained_model(
        model_path, 
        None, 
        model_name, 
        load_8bit, 
        load_4bit, 
        device=device, 
        cache_dir=cache_dir
    )
    video_processor = processor['video']
    conv_mode = "llava_v1"

    for query in tqdm(queries):
        id_ = query['id']
        video_path = os.path.join('videos', query['demonstration_type'], '.'.join([query['key'], 'mp4']))
        question = query['question']
        options = query['options']
        optionized_list = [f"{chr(65 + i)}. {option}" for i, option in enumerate(options)]
        gt = optionized_list[query['answer']]

        
        
        inp, all_choices, index2ans = construct_prompt(question=question,
                                                          options=options,
                                                          num_frames=total_frames)

        video_tensor = video_processor(video_path, return_tensors='pt')['pixel_values']
        if type(video_tensor) is list:
            tensor = [video.to(model.device, dtype=torch.float16) for video in video_tensor]
        else:
            tensor = video_tensor.to(model.device, dtype=torch.float16)

        # print(f"{roles[1]}: {inp}")
        conv = conv_templates[conv_mode].copy()
        roles = conv.roles
        inp = ' '.join([DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames) + '\n' + inp
        print(f"{roles[1]}: {inp}")
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=tensor,
                do_sample=False,
                temperature=GENERATION_TEMPERATURE,
                max_new_tokens= MAX_TOKENS,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        response = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        
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
