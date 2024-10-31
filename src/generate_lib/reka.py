import reka
import os
import json
from tqdm import tqdm
from generate_lib.constant import GENERATION_TEMPERATURE, GENERATION_TOP_P, SYSTEM_PROMPT, MAX_TOKENS
from generate_lib.construct_prompt import construct_prompt
from dotenv import load_dotenv
load_dotenv()
import logging 

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def generate_response(model_name: str, 
                      queries: list,
                      total_frames: int,
                      output_dir: str):
    
    logging.info(f"Model: {model_name}")
    reka.API_KEY = os.environ.get("REKA_API_KEY")

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


        response = reka.chat(
            prompt,
            media_filename=video_path,
            model_name=model_name,
            request_output_len=MAX_TOKENS,
            temperature=GENERATION_TEMPERATURE,
            runtime_top_p=GENERATION_TOP_P,
            random_seed=215
        )

        response = response['text']
        
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