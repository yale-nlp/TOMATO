import os
import google.generativeai as genai
import time
from tqdm import tqdm
import json
from generate_lib.constant import GENERATION_TEMPERATURE, GENERATION_TOP_P, SYSTEM_PROMPT, MAX_TOKENS
from generate_lib.construct_prompt import construct_prompt
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv
load_dotenv()
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def generate_response(model_name: str, 
                      queries: list, 
                      total_frames: int, 
                      output_dir: str):

    logging.info(f"Model: {model_name}")

    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

    generation_config = genai.types.GenerationConfig(
        candidate_count=1,
        max_output_tokens=MAX_TOKENS,
        temperature=GENERATION_TEMPERATURE,
        top_p=GENERATION_TOP_P)
    
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE}
    
    model = genai.GenerativeModel(model_name=model_name,
                                  system_instruction=SYSTEM_PROMPT)
    
    
    for query in tqdm(queries):
        time.sleep(12)
        id_ = query['id']
        video_path = os.path.join('videos', query['demonstration_type'], '.'.join([query['key'], 'mp4']))
        question = query['question']
        options = query['options']
        optionized_list = [f"{chr(65 + i)}. {option}" for i, option in enumerate(options)]
        gt = optionized_list[query['answer']]

        video_file = genai.upload_file(path=video_path)

        while video_file.state.name == "PROCESSING":
            # logging.info(".")
            time.sleep(2)
            video_file = genai.get_file(video_file.name)


        prompt, all_choices, index2ans = construct_prompt(question=question,
                                                          options=options,
                                                          num_frames=total_frames)

        response = model.generate_content(contents=[video_file, prompt],
                                          generation_config=generation_config,
                                          safety_settings=safety_settings)

        genai.delete_file(video_file.name) # clean up after generation
        response.resolve()
        response = response.text


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