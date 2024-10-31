from generate_lib.utils import read_video
from openai import OpenAI
import os
import json
from tqdm import tqdm
from generate_lib.constant import GENERATION_TEMPERATURE, GENERATION_TOP_P, SYSTEM_PROMPT, MAX_TOKENS, GENERATION_SEED
from generate_lib.construct_prompt import construct_prompt, construct_prompt_random_guess
import time
from dotenv import load_dotenv
load_dotenv()
import logging 
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("Assitant")

def generate_response(model_name: str, 
                      queries: list,
                      total_frames: int,
                      output_dir: str):

    logging.info(f"Model: {model_name}")


    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    for query in tqdm(queries):
        id_ = query['id']
        video_path = os.path.join('videos', query['demonstration_type'], '.'.join([query['key'], 'mp4']))
        question = query['question']
        options = query['options']
        optionized_list = [f"{chr(65 + i)}. {option}" for i, option in enumerate(options)]
        gt = optionized_list[query['answer']]

        if total_frames > 0:
            base64Frames, _ = read_video(video_path=video_path, total_frames=total_frames)

            prompt, all_choices, index2ans = construct_prompt(question=question,
                                                            options=options,
                                                            num_frames=total_frames)

            prompt_message = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                }, 
                {
                    "role": "user",
                    "content": [prompt, *map(lambda x: {"image": x, "resize": 100}, base64Frames),],
                },
            ]
        else:
            print(f"[INFO] Your are not using any frames, GPT-4o is doing random guess...")
            prompt, all_choices, index2ans = construct_prompt_random_guess(question=question,
                                                            options=options,
                                                            num_frames=total_frames)
            
            prompt_message = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                }, 
                {
                    "role": "user",
                    "content": prompt,
                },
            ]


        params = {
            "model": model_name,
            "messages": prompt_message,
            "max_tokens": MAX_TOKENS,
            "temperature": GENERATION_TEMPERATURE,
            "top_p": GENERATION_TOP_P,
            "seed": GENERATION_SEED
        }


        response = client.chat.completions.create(**params)
        response = response.choices[0].message.content

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
        
        # time.sleep(1)