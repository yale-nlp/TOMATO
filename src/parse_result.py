from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
import os
import re
import random
import json
from tqdm import tqdm


# number of maximum re-try in parsing
MAX_ITER = 5


def gpt_parser(response, all_choices, index2ans):
    print("using gpt parser...")
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    prompt = f"""You are given a response, a list of multiple-choice options, and a index2answer mapping. You are required to extract the letter option from the GPT. 
    
    response: {response}

    all_choices: {all_choices}

    index2answer: {index2ans}

Only output the single parsed letter from the response. No other texts are needed. 

If you think no options can match the index2answer dictionary, randomly select one letter. 

Your extracted letter is: 
"""
    prompt_message = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
    
    params = {
            "model": "gpt-4o-mini",
            "messages": prompt_message,
            "max_tokens": 16,
            "temperature": 0.0,
        }
    response = client.chat.completions.create(**params)
    response = response.choices[0].message.content

    return response


def pre_parser(response, all_choices, index2ans):
    parsed_response = ""
    response = response.strip()

    # preprocess matches
    full_choices = [f'{k}: {v}' for k, v in index2ans.items()]
    pattern = r"^Answer is:?[\(]?([A-Fa-f])[\)]?$"
    match = re.match(pattern, response)

    # exact match single letter
    if len(response) == 1 and response.upper() in all_choices:
        parsed_response = response.upper()

    # exact match of the choice
    elif response.upper() in full_choices:
        parsed_response = response[0].upper()

    # regex match of "Answer is: A", "Answer is (A)", etc
    elif match:
        parsed_response = match.group(1).upper()

    return parsed_response


def parse_result(response_json: str):
    id_ = response_json['id']
    question = response_json['question']
    response = response_json['response']
    all_choices = response_json['all_choices']
    index2ans = response_json['index2ans']
    gt = response_json['gt']

    # pre parsing using regex
    parsed_response = pre_parser(response=response, 
                                 all_choices=all_choices, 
                                 index2ans=index2ans)
    
    # actual parsing using gpt
    if parsed_response not in all_choices:
        curr_iter = 0
        while curr_iter < MAX_ITER:
            response_candidate = gpt_parser(response=response, 
                                            all_choices=all_choices, 
                                            index2ans=index2ans)
            
            if response_candidate in all_choices:
                parsed_response = response_candidate
                break
            curr_iter += 1
        
    if parsed_response not in all_choices:
        parsed_response = random.choice(all_choices)

    # format parsed result
    parsed_result = {
        "id": id_,
        "question": question,
        "response": parsed_response,
        "gt": gt
    }

    return parsed_result


def get_result_file_dict(results_dir):
    # compute a nested dictionaries that maintains the directory hierarchy
    # also for visualization
    results_dict = dict()

    if not os.path.exists(results_dir):
        return results_dict
    
    type_dirs = os.listdir(results_dir)
    for type_dir in type_dirs:
        type_dir = os.path.join(results_dir, type_dir)
        frame_dirs = [os.path.join(type_dir, d) for d in os.listdir(type_dir)]
        results_dict[type_dir] = {d: os.listdir(d) for d in frame_dirs}
    
    return results_dict


def get_result_file_list(results_dir):
    # compute a list 
    result_file_dict = get_result_file_dict(results_dir)

    result_file_paths = list()
    for type_dir, frame_dirs in result_file_dict.items():
        for frame_dir, result_files in frame_dirs.items():
            for result_file in result_files:
                result_file_path = os.path.join(frame_dir, result_file)
                result_file_paths.append(result_file_path)
    return result_file_paths


def main():
    result_files = get_result_file_list('results')
    for result_file in result_files:
        print('checking ' + result_file)

        # open parsed result file if exists
        parsed_ids = set()
        parsed_result_file = re.sub('results', 'parsed_results', result_file)
        os.makedirs(os.path.dirname(parsed_result_file), exist_ok=True)

        if os.path.exists(parsed_result_file):
            with open(parsed_result_file, 'r') as f2:
                parsed_result = [json.loads(line) for line in f2]
                parsed_ids = {p['id'] for p in parsed_result}

        # open result file
        with open(result_file, 'r') as f1:
            result = [json.loads(line) for line in f1]

        # append result if not parsed before
        count = 0
        for r in result:
            if r['id'] in parsed_ids:
                continue
            print('question ' + r['id'])
            pr = parse_result(r)
            with open(parsed_result_file, 'a+') as f2:
                f2.write(json.dumps(pr) + '\n')
            count += 1
        
        print('newly added number of results ' + str(count) + '\n')


if __name__ == "__main__":
    main()
