from generate_lib.construct_prompt import construct_prompt, construct_prompt_random_guess
from generate_lib.constant import GENERATION_TEMPERATURE, MAX_TOKENS, SYSTEM_PROMPT, GENERATION_TOP_P
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import logging
from generate_lib.utils import read_video
import os
from tqdm import tqdm 
import torch
import json


def generate_response(model_name: str, 
                      queries: list,
                      total_frames: int,
                      output_dir: str):


    logging.info(f"Model: {model_name}")
    
    model_path = os.path.join("./pretrained", model_name)

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        attn_implementation="flash_attention_2"
    )

    processor = AutoProcessor.from_pretrained(model_path)



    for query in tqdm(queries):
        id_ = query['id']
        video_path = os.path.join('videos', query['demonstration_type'], '.'.join([query['key'], 'mp4']))
        question = query['question']
        options = query['options']
        optionized_list = [f"{chr(65 + i)}. {option}" for i, option in enumerate(options)]
        gt = optionized_list[query['answer']]

        if total_frames > 0:
            base64Frames, _ = read_video(video_path=video_path, 
                                        total_frames=total_frames)

            # convert to the QwenVL format
            base64Frames = [f"data:image;base64,{x}" for x in base64Frames]

            # question_prompt = construct_prompt(question=question, 
            #                                     options=options)
            
            question_prompt, all_choices, index2ans = construct_prompt(question=question,
                                                            options=options,
                                                            num_frames=total_frames)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": base64Frames,
                        },
                        {"type": "text", "text": question_prompt},
                    ],
                }
            ]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        else:
            print(f"[INFO] random guess")
            question_prompt, all_choices, index2ans = construct_prompt_random_guess(question=question,
                                                            options=options,
                                                            num_frames=total_frames)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": base64Frames,
                        },
                        {"type": "text", "text": question_prompt},
                    ],
                }
            ]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text],
                padding=True,
                return_tensors="pt",
            )


        inputs = inputs.to("cuda")

        # Inference
        generated_ids = model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = output_text[0]

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

        