import os
import cv2
import json
import logging
import requests
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor 
from transformers import AutoModelForCausalLM

from generate_lib.construct_prompt import construct_prompt
from generate_lib.constant import GENERATION_TEMPERATURE, MAX_TOKENS


def load_video(video_path: str,
               num_frames: int):

    images = list()

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(total_frames // num_frames, 1)

    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_interval)
        ret, frame = cap.read()
        if not ret:
            break
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        images.append(image)

    return images
    

def generate_response(model_name: str, 
                      queries: list,
                      total_frames: int,
                      output_dir: str):

    logging.info(f"Model: {model_name}")

    # Note: set _attn_implementation='eager' if you don't have flash_attn installed
    model_path = os.path.join("./pretrained", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map="cuda", 
        trust_remote_code=True, 
        torch_dtype="auto", 
        _attn_implementation='flash_attention_2'
    )

    # for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
    processor = AutoProcessor.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        num_crops=4
    ) 
    
    for query in tqdm(queries):
        id_ = query['id']
        video_path = os.path.join('videos', query['demonstration_type'], '.'.join([query['key'], 'mp4']))
        question = query['question']
        # raw_question = question.copy()
        options = query['options']
        optionized_list = [f"{chr(65 + i)}. {option}" for i, option in enumerate(options)]
        gt = optionized_list[query['answer']]


       
        
        question_prompt, all_choices, index2ans = construct_prompt(question=question,
                                                          options=options,
                                                          num_frames=total_frames)


        images = load_video(video_path, total_frames)

        placeholder = ""
        for i in range(len(images)):
            placeholder += f"<|image_{i+1}|>\n"
        question = placeholder + question_prompt
        messages = [
            {"role": "user", "content": question},
        ]

        prompt = processor.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        inputs = processor(prompt, images, return_tensors="pt").to("cuda:0") 

        generation_args = { 
            "max_new_tokens": MAX_TOKENS, 
            "do_sample": False, 
        } 

        generate_ids = model.generate(**inputs, 
            eos_token_id=processor.tokenizer.eos_token_id, 
            **generation_args
        )

        # remove input tokens 
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = processor.batch_decode(generate_ids, 
                                          skip_special_tokens=True, 
                                          clean_up_tokenization_spaces=False)[0] 

        with open(output_dir, "a") as f:
            f.write(json.dumps(
                {
                    "id": id_,
                    # "question": raw_question,
                    "question": question,
                    "response": response,
                    "all_choices": all_choices,
                    "index2ans": index2ans,
                    'gt': gt
                }
            ) + "\n")
