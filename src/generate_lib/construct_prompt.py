from typing import Tuple

def construct_prompt(question: str, options: list, num_frames: int) -> Tuple:    
    """
    Args:
        question (str): question in the dataset
        options (list): list of options 
        num_frames (int): number of frames extracted from the video 
    
    Returns:
        prompt (str): well-constructed prompt
        all_choices (list): list of options (A, B, C, ...)
        index2ans (dict): dictionary of option-answer mapping
    """

    all_choices = [f"{chr(65 + i)}" for i in range(len(options))]
    index2ans = {all_choices[i]: options[i] for i in range(len(options))}

    prompt = f"""You will be provided with {num_frames} separate frames uniformly sampled from a video, the frames are provided in chronological order of the video. Analyze these frames and provide the answer to the question about the video content. Answer the multiple-choice question about the video content. 

You must use these frames to answer the multiple-choice question; do not rely on any externel knowledge or commonsense. 

<question> 
{question} 
</question>

<options> 
{index2ans} 
</options>

Even if the information in these separate frames is not enough to answer the question, PLEASE TRY YOUR BEST TO GUESS AN ANSWER WHICH YOU THINK WOULD BE THE MOST POSSIBLE ONE BASED ON THE QUESTION. 

DO NOT GENERATE ANSWER SUCH AS 'NOT POSSIBLE TO DETERMINE.' 
"""

    return prompt, all_choices, index2ans


def construct_prompt_random_guess(question: str, options: list, num_frames: int) -> Tuple:    
    print(f"[INFO]: You are inputting 0 frames to the model.")

    all_choices = [f"{chr(65 + i)}" for i in range(len(options))]
    index2ans = {all_choices[i]: options[i] for i in range(len(options))}

    prompt = f"""Randomly guess a reasonable answer based on the question only.

<question> 
{question} 
</question>

<options> 
{index2ans} 
</options>
    
DO NOT GENERATE ANSWER SUCH AS 'NOT POSSIBLE TO DETERMINE.' 
"""

    return prompt, all_choices, index2ans