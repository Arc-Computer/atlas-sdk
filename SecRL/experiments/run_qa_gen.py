# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from secgym.myconfig import CONFIG_LIST
from secgym.qagen.qa_gen import QAGen
import os
import argparse
import json
import random

def get_args():
    parser = argparse.ArgumentParser(description="Run Alert QA Generation")
    parser.add_argument("--qa_path", "-q", type=str, default="./experiments/questions", help="Path to save the generated QA")
    parser.add_argument("--graph_path", "-g", type=str, default="secgym/qagen/graph_files", help="Path to the alert graph")
    parser.add_argument("--cache_seed", type=int, default=42, help="Seed for the cache")
    parser.add_argument("--include_entry", action="store_true", help="Include full alert entry in the question prompt")
    parser.add_argument("--model", "-m", type=str, default="o1", help="Model to use for QA generation")
    parser.add_argument("--solution_model", "-s", type=str, default="o1", help="Model to use for solution generation")
    parser.add_argument("--include_incident", action="store_false", help="Include incident context in the question prompt")
    parser.add_argument("--split", type=str, default="train", help="Split to generate QA for")
    parser.add_argument("--relevant_type", type=str, default="n/a", help="Relevance type for the QA")
    parser.add_argument("--train_cap", type=int, default=100, help="Cap for the number of training questions")
    args = parser.parse_args()
    return args

args = get_args()
graph_files = [
    'incident_34.graphml',
    'incident_166.graphml',
    'incident_55.graphml',
    'incident_5.graphml',
    'incident_38.graphml',
    'incident_134.graphml',
    'incident_39.graphml',
    'incident_322.graphml'
 ]


# form question file name
qa_file_suffix = "qa"
if args.include_entry:
    qa_file_suffix += "_entry"
if args.include_incident:
    qa_file_suffix += "_incident"
qa_file_suffix += f"_{args.model}"
if args.solution_model is not None and args.solution_model != args.model:
    qa_file_suffix += f"_{args.solution_model}"
qa_file_suffix += f"_c{args.cache_seed}.json"

set_split = args.split
relevance_type = args.relevant_type
train_cap = args.train_cap

if relevance_type == "low":
    saved_paths_path = "./experiments/split_files/low_split"
elif relevance_type == "high":
    saved_paths_path = "./experiments/split_files/high_split"
elif relevance_type == "median":
    saved_paths_path = "./experiments/split_files/median_split"
else:
    saved_paths_path = None

os.makedirs(args.qa_path, exist_ok=True)


t = 0
print(f"Generating QA for the {set_split} set with relevance type {relevance_type}")
for file in graph_files:
    skip_count = 0
    graph_file_path = os.path.join(args.graph_path, file)
    qagenena = QAGen(
        config_list=CONFIG_LIST, 
        graph_path=graph_file_path,
        qa_gen_model=args.model,
        solution_gen_model=args.solution_model,
        cache_seed=41
    )

    # 1
    qas = []
    existing_qa_map = {}

    # 1.2 resume from new qa file if it exists
    current_qa_path = f"{args.qa_path}/{file.split('.')[0]}_{qa_file_suffix}"
    if os.path.exists(current_qa_path):
        with open(current_qa_path, "r") as f:
            new_qas = json.load(f)
            for qa in new_qas:
                existing_qa_map[(qa["start_alert"], qa["end_alert"])] = qa

    # 2. open graph paths file json
    if saved_paths_path is None:
        qagenena.generate_qa(graph_path=graph_file_path, qa_path=current_qa_path)
    else:
        with open(os.path.join(saved_paths_path, file.split('.')[0] + ".json"), "r") as f:
            question_paths = json.load(f)
        path_from_split = question_paths[set_split]
        if set_split == 'train' and train_cap > 0:
            path_from_split = random.sample(path_from_split, min(train_cap, len(path_from_split)))
        print(f"Generating QA for {file}... for {len(path_from_split)} questions")

        all_questions = []
        for path_dict in path_from_split:
            start_alert = path_dict["start_alert"]
            end_alert = path_dict["end_alert"]
            if (start_alert, end_alert) in existing_qa_map:
                all_questions.append(existing_qa_map[(start_alert, end_alert)])
                print(f"Reusing question for {start_alert} -> {end_alert}")
            else:
                print(f"Generating question for {start_alert} -> {end_alert}")
                question = qagenena.generate_one_question(path_dict)
                all_questions.append(question)
            # save the questions
            # Note this is a new path
            with open(current_qa_path, "w") as f:
                json.dump(all_questions, f, indent=4)
        t+=len(path_from_split)

print(f"Total {set_split} questions: {t}")
