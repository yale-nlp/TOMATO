from generate_lib.utils import read_video
import anthropic
import os
import time
import json 
from tqdm import tqdm
from generate_lib.constant import GENERATION_TEMPERATURE, GENERATION_TOP_P, SYSTEM_PROMPT, MAX_TOKENS
from generate_lib.construct_prompt import construct_prompt
from dotenv import load_dotenv
import logging
load_dotenv()

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def generate_response(model_name: str,
                      queries: list,
                      total_frames: int, 
                      output_dir: str):

    logging.info(f"Model: {model_name}")

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    for query in tqdm(queries):
        id_ = query['id']
        video_path = os.path.join('videos', query['demonstration_type'], '.'.join([query['key'], 'mp4']))
        question = query['question']
        options = query['options']
        optionized_list = [f"{chr(65 + i)}. {option}" for i, option in enumerate(options)]
        gt = optionized_list[query['answer']]


        base64Frames, _ = read_video(video_path=video_path,
                                     total_frames=total_frames)

        prompt, all_choices, index2ans = construct_prompt(question=question,
                                                          options=options,
                                                          num_frames=total_frames)

        prompt_message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ] + [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": frame,
                        },
                    } for frame in base64Frames
                ],
            },
        ]
        

        params = {
            "model": model_name,
            "system": SYSTEM_PROMPT,
            "messages": prompt_message,
            "max_tokens": MAX_TOKENS,
            "temperature": GENERATION_TEMPERATURE,
            "top_p": GENERATION_TOP_P
        }
        # if model_name == "claude-3-opus-20240229":
        #     time.sleep(2)
        response = client.messages.create(**params)
        response = response.content[0].text


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

        time.sleep(30)
