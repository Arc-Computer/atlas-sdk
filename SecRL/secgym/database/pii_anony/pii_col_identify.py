# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from secgym.utils.utils import LLM_call
from secgym.database.process_logs import SEPARATOR
from secgym.myconfig import config_list_4o
import os
import pandas as pd
import re
import json

import numpy as np

# iterate over all csvs in the directory
# for each csv, load the column names and some examples of the data
# call LLM to determine if it is a PII, or contains PII information
# collect and save repetitive column names


PII_identify_prompt = """Given a column name and some examples of the data, determine if it is a PII, or contains PII information.

PII stands for Personally Identifiable Information. It is information that can be used to identify an individual. Examples of PII include:
- Name
- IP address
- Email 

Please use JSON format for your response:
{
    "<column_name>": {
        "is_pii": <True/False>,
        "has_regex_pattern": <True/False>,
        "regex_pattern": "<regex_pattern>"
}

Notes:
- is_pii: True if the column contains PII information, False otherwise.
If is_pii is False, please skip the rest of the fields.
- has_regex_pattern: True if the PIIs can be identified using a regex pattern. For example, IP address can be identified using a regex pattern, but name cannot.
- regex_pattern: please provide the pattern if has_regex_pattern is True.
"""

str_matcher = {
    "IP": {
        "is_pii": True,
        "has_regex_pattern": True,
        "regex_pattern": r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
    },
}

def identify_pii(csv_file, save_file):
    df = pd.read_csv(csv_file, sep=SEPARATOR, encoding='utf-8', on_bad_lines='skip', engine='python')

    if os.path.exists(save_file):
        with open(save_file, "r") as f:
            examined_columns = json.load(f)
    else:
        examined_columns = {}

    for column in df.columns:
        # random get 5 examples
        examples = df[column].sample(5, replace=True).tolist()
        # convert to string
        examples = [str(e) for e in examples]

        if column in examined_columns and "examples" in examined_columns[column]:
            continue

        if len("".join(examples)) < 300:
            print("All examples are empty")
            examples = df[column].sample(10, replace=True).tolist()

        task = f"Column Name: {column}\nExamples: {examples}"
        if len(df[column].unique()) == 1:
            if isinstance(df[column].unique()[0], float) and np.isnan(df[column].unique()[0]):
                print("All examples are empty")
                examined_columns[column] = {
                    "is_pii": False,
                    "examples": examples,
                }
                continue
        
        if len(df[column].unique()) <= 5:
            examples = df[column].unique().tolist()

        # convert all nan to "nan"
        examples = [str(e) if not pd.isnull(e) else "nan" for e in examples]
            

        for i in range(5):
            print("-" * 10, "Input Prompt", "-" * 10)
            print(task)
            response = LLM_call(
                instruction=PII_identify_prompt,
                task=task,
                config_list=config_list_4o,
                response_format = {"type": "json_object"},
                cache_seed=41,
            )
            print("-" * 10, "Response", "-" * 10)
            print(response)

            try:
                response = json.loads(response)
            except:
                print("Error parsing response")
                response = {}
                continue
            if not (column in response or "is_pii" in response[column]):
                print("Invalid response")
                response = {}
                continue

            break
                
        if response != {}:
            response[column]['examples'] = examples
            examined_columns[column] = response[column]

        with open(save_file, "w") as f:
            json.dump(examined_columns, f)
    

# identify_pii(
#     csv_file="./data/incidents/incident_5/AADManagedIdentitySignInLogs.csv",
#     save_file="examined_columns.json"
# )

save_file = "examined_columns.json"
folder = "./data/incidents/incident_5"

# for file_name in os.listdir(folder):
#     print("-"*50)
#     print("Identifying PII for file: ", file_name)

#     if file_name.endswith(".csv"):
#         identify_pii(
#             csv_file=os.path.join(folder, file_name),
#             save_file=save_file
#         )
#     elif os.path.isdir(os.path.join(folder, file_name)):
#         # get the first csv file
#         csv_file = os.path.join(folder, file_name, f"{file_name}_0.csv")
#         identify_pii(
#             csv_file=csv_file,
#             save_file=save_file
#         )
        
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# Second filter

second_extraction = """Given a column name and some examples of the data, please determine the following:
1. If the column is a dict/list that contains several fields, or a single field.
2. If there are any PII information.
    - If the column name indicate it is PII, even if there are no value in the examples, please mark it as PII. For example, "ip_address", "email", "user_name".
    - If there are uuids or ip-address like strs, please check whether it is related to user/account or device. If it is related to device, you should NOT consider it as PII.
3. If it is a dict, and there are PII in the dict, please identify the PII fields.
    - Example: from a example, there are '... {"user": {"name":"Lynne Robbins"...} ...}', then the PII field is 'name', you only need to extract the innermost PII field.
    - There may be several fields like "latitude" and "longitude" in the dict.

Please use JSON format for your response:
{
    "<column_name>": {
        "is_dict": <True/False>,
        "is_pii": <True/False>,
        "pii_fields": ["<field_1>", "<field_2>"],
}
"""


save_file = "second_filter.json"
with open(save_file, "r") as f:
    columns = json.load(f)

has_pii = []
for k, val in columns.items():
    if val['is_pii']:
        # print(k, val)
        has_pii.append(k)
    # print("-"* 50)

for file_name in os.listdir(folder):
    if file_name.endswith(".csv"):
        df = pd.read_csv(os.path.join(folder, file_name), sep=SEPARATOR, encoding='utf-8', on_bad_lines='skip', engine='python')
    elif os.path.isdir(os.path.join(folder, file_name)):
        # get the first csv file
        csv_file = os.path.join(folder, file_name, f"{file_name}_0.csv")
        df = pd.read_csv(csv_file, sep=SEPARATOR, encoding='utf-8', on_bad_lines='skip', engine='python')
    else:
        continue

    print("Identifying PII for file: ", file_name)

    # iterated over columns of df to see if it is in has_pii
    for column in df.columns:
        if column in has_pii:
            print("Column: ", column)
            u = df[column].unique().tolist()

            for i in range(min(5, len(u))):
                print(u[i])
            print("-" * 50)

    

        
    