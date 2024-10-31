import os
import re
import json
import pandas as pd


from parse_result import get_result_file_list


NUM_FRAME = '16'


def get_single_score(result):
    response = result['response']
    gt = result['gt'][0]
    return response == gt


def get_score(parsed_result_file: str):
    overall_accuracies = list()
    with open(parsed_result_file, "r") as f:
        for l in f:
            l = json.loads(l)
            overall_accuracies.append(get_single_score(l))

    return sum(overall_accuracies), len(overall_accuracies)


def main():
    
    # read in parsed results
    parsed_result_files = get_result_file_list('parsed_results')
    data = dict()
    for parsed_result_file in parsed_result_files:

        split_file_path = parsed_result_file.split(os.sep)
        reason_type, demo_type = re.match(r"shuffle_(.*)\+(.*)", split_file_path[1]).groups()
        num_frame = split_file_path[2]
        model = split_file_path[3].rsplit('.', 1)[0]
        score, count = get_score(parsed_result_file)

        data[(reason_type, demo_type, num_frame, model)] = (score, count)

    # turn dict into dataframe
    df = pd.DataFrame.from_dict(data, orient='index', columns=['Score', 'Count'])
    df.index = pd.MultiIndex.from_tuples(df.index, names=['ReasonType', 'DemoType', 'NumFrame', 'Model'])

    # filter result for main table
    print("filter result for main table")
    df_filtered = df[df.index.get_level_values('NumFrame') == NUM_FRAME]
    grouped = df_filtered.groupby(['ReasonType', 'Model']).agg({'Score': 'sum', 'Count': 'sum'})
    grouped[f'Score_per_Count_{NUM_FRAME}_Frames'] = grouped['Score'] / grouped['Count']
    print(grouped[[f'Score_per_Count_{NUM_FRAME}_Frames']])
    print()

    # filter result for demonstration types
    print("filter result for demonstration types")
    demo_grouped = df_filtered.groupby(['DemoType', 'Model']).agg({'Score': 'sum', 'Count': 'sum'})
    demo_grouped[f'Score_per_Count_{NUM_FRAME}_Frames'] = demo_grouped['Score'] / demo_grouped['Count']
    print(demo_grouped[[f'Score_per_Count_{NUM_FRAME}_Frames']])
    print()

    # filter result for models
    print("filter result for models")
    model_grouped = df_filtered.groupby(['Model']).agg({'Score': 'sum', 'Count': 'sum'})
    model_grouped[f'Score_per_Count_{NUM_FRAME}_Frames'] = model_grouped['Score'] / model_grouped['Count']
    print(model_grouped[[f'Score_per_Count_{NUM_FRAME}_Frames']])
    print()

if __name__ == "__main__":
    main()
