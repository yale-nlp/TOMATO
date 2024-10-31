import numpy as np
import os
import json
from tqdm import tqdm
from parse_result import get_result_file_list

def get_score(parsed_result_file: str):
    overall_accuracies = list()

    with open(parsed_result_file, "r") as f:
        for l in f:
            data = json.loads(l)
            response = data['response']
            gt = data['gt'][0]
            overall_accuracies.append(response==gt)

    return float(np.mean(overall_accuracies))


def main():
    parsed_result_files = get_result_file_list('parsed_results')
    for parsed_result_file in parsed_result_files:
        score = get_score(parsed_result_file)
        print(parsed_result_file)
        print(score)
        print()


if __name__ == "__main__":
    main()
