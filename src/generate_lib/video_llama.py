import sys
sys.path.append('./')
import os
import json
from tqdm import tqdm
from generate_lib.constant import GENERATION_TEMPERATURE, GENERATION_TOP_P, SYSTEM_PROMPT, MAX_TOKENS, GENERATION_SEED
from generate_lib.construct_prompt import construct_prompt
from generate_lib.VideoLLaMA2.videollama2 import model_init, mm_infer
from generate_lib.VideoLLaMA2.videollama2.utils import disable_torch_init
import logging 
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')




def generate_response(model_name: str, 
                      queries: list,
                      total_frames: int,
                      output_dir: str):
    
    logging.info(f"Model: {model_name}")
    disable_torch_init()


    # model_path = 'DAMO-NLP-SG/VideoLLaMA2-8x7B'
    model_path = f"./pretrained/{model_name}"
    model, processor, tokenizer = model_init(model_path)

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
        modal = 'video'
        modal_path = video_path
        instruct = prompt
        

        output = mm_infer(processor[modal](modal_path), instruct, model=model, tokenizer=tokenizer, do_sample=False, modal=modal)

        with open(output_dir, "a") as f:
            f.write(json.dumps(
                {
                    "id": id_,
                    "question": question,
                    "response": output,
                    "all_choices": all_choices,
                    "index2ans": index2ans,
                    'gt': gt
                }
            ) + "\n")

