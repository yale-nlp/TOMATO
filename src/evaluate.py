import os
import json
import argparse
import warnings

import importlib
from collections import defaultdict


def validate_choices(input_value, all_choices, input_name):
    if input_value == 'ALL':
        return all_choices
    else:
        selected_values = [item.strip() for item in input_value.split(",")]
        invalid_values = [item for item in selected_values if item not in all_choices]
        if invalid_values:
            raise ValueError(f"Invalid {input_name} type(s): {', '.join(invalid_values)}. "
                             f"Valid choices are: {', '.join(all_choices + ['ALL'])}")
        return selected_values


def main(model_name: str, model_families: dict, queries: dict, total_frames: int, shuffle: bool = False) -> None:
    for family, model_list in model_families.items():
        if model_name in model_list:

            assert model_name in model_list, f"Model version {model_name} not supported in the model family {family}"
            
            module = importlib.import_module(f"generate_lib.{family}")
            generate_response = getattr(module, "generate_response")
            for output_dir, qas in queries.items():
                print(f"Current output directory is {output_dir}")
                generate_response(
                    model_name=model_name, 
                    queries=qas, 
                    total_frames=total_frames, 
                    output_dir=output_dir,
                    shuffle=shuffle
                )
            return
    
    raise ValueError(f"Model family of the model {model_name} not supported")

    
if __name__ == "__main__":
    # load configs
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    model_families = config.get('models', {})
    video_models = config.get('video_models', [])
    model_choices = [item for sublist in model_families.values() for item in sublist]
    reasoning_type_choices = config.get('reasoning_types', [])
    demonstration_type_choices = config.get('demonstration_types', [])


    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=model_choices)
    parser.add_argument('--reasoning_type', type=str, default='ALL')
    parser.add_argument('--demonstration_type', type=str, default='ALL')
    parser.add_argument('--total_frames', type=int, default=16)
    args = parser.parse_args()


    # get model name and total frame
    model_name = args.model
    if model_name in video_models:
        warnings.warn(f"{model_name} processes videos directly and **total_frames** cannot be specified.", UserWarning)
        total_frames = -1
    else:
        total_frames = args.total_frames


    # check reasoning types and demonstration types validity
    reasoning_type = validate_choices(args.reasoning_type, reasoning_type_choices, 'reasoning')
    demonstration_type = validate_choices(args.demonstration_type, demonstration_type_choices, 'demonstration')
    
    
    # creat output directories & construct queries for each output path
    queries = defaultdict(list)
    existing_paths = list()
    for rt in reasoning_type:
        dataset_path = f"./data/{rt}.json"
        with open(dataset_path, "r") as f:
            qas = json.load(f)
        
        for dt in demonstration_type:
            # create output path
            output_subdir = '+'.join([rt, dt])
            output_path = f"./results/{output_subdir}/{total_frames}/{model_name}.jsonl"
            curr_results = set()
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

            elif os.path.exists(output_path):
                existing_paths.append(output_path)
                with open(output_path, 'r') as f:
                    for line in f:
                        curr_results.add(json.loads(line)['id'])
            
            # construct query dictionary & leave out existing results
            for id_, qa in qas.items():
                if qa['demonstration_type'] == dt:
                    if curr_results and id_ in curr_results:
                        continue
                    qa['id'] = id_
                    queries[output_path].append(qa)
    
    if existing_paths:
        warnings.warn(f"Result json(s) {', '.join(existing_paths)} already exist! Will append new results to existing files", UserWarning)

    # generate responses 
    print("Generating responses ...")
    main(model_name, model_families, queries, total_frames)
