# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import re
import json
import pandas as pd
import time
import random
import uuid


SEPARATOR = "❖"

def generate_ip():
    return ".".join(str(random.randint(0, 255)) for _ in range(4))

def extract_ip_from_string(input) -> list:
    return re.findall(r'[0-9]+(?:\.[0-9]+){3}', input)

def extract_ipv6_from_string(input_string):
    # Updated regular expression for matching valid IPv6 addresses with at least 5 colons and handling "::"
    ipv6_pattern = r'\b(?:[0-9A-Fa-f]{1,4}:){2,7}[0-9A-Fa-f]{1,4}::?(?:[0-9A-Fa-f]{1,4})?\b'
    
    # Find all matches in the input string
    return re.findall(ipv6_pattern, input_string)

def generate_ipv6(original_ipv6):
    # random select 3 char and replace with random allow char
    new_ipv6 = original_ipv6

    for i in range(5):
        index = random.randint(0, len(new_ipv6)-1)
        if new_ipv6[index].isalnum():
            new_ipv6 = new_ipv6[:index] + random.choice('0123456789ABCDEF') + new_ipv6[index+1:]
    
    return new_ipv6

# ee17abf2-35a2-4a16-9850-89ebb4f499d0  generate this type
def generate_uuid():
    return str(uuid.uuid4())

def extract_uuid_from_string(input) -> list:
    if not isinstance(input, str):
        print("Convert to string:", input)
        input = str(input)
    return re.findall(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', input)

def extract_mac_address_from_string(input) -> list:
    return re.findall(r'(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}', input)

def generate_mac():
    return ':'.join([random.choice('0123456789ABCDEF') + random.choice('0123456789ABCDEF') for i in range(6)]) # correct

def extract_sid_from_string(input) -> list:
    return re.findall(r'S-[0-9-]+', input)

def generate_sid(original_sid):
    # random select 3 number and replace with random number
    new_sid = original_sid

    for i in range(10):
        index = random.randint(0, len(new_sid)-1)
        if new_sid[index].isdigit():
            new_sid = new_sid[:index] + str(random.randint(0, 9)) + new_sid[index+1:]

    return new_sid

def extract_sharepoint_url_account_from_string(input) -> list:
    matches = re.findall(r'sharepoint.com/personal/([^/]+)_vnevado_', input)
    if len(matches) == 0:
        matches = re.findall(r'sharepoint.com/personal/([^/]+)/', input)
    return matches

def match_latitute_longitude(input:str):
    lat =  re.findall(r'"latitude":([-+]?\d*\.?\d+),', input)
    lon =  re.findall(r'"longitude":([-+]?\d*\.?\d+)}', input)
    return lat+lon


# Helper functions to classify the type of string (IP, UUID, etc.)
def is_ip_address(s):
    ip_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
    return re.match(ip_pattern, s) is not None

def is_uuid(s):
    uuid_pattern = r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
    return re.match(uuid_pattern, s) is not None

def is_dict(s):
    if isinstance(s, str) and s.startswith("{") and s.endswith("}"):
        try:
            json.loads(s)
            return True
        except (ValueError, TypeError):
            return False
    return False

def classify_sample(sample_list):
    ip_count = sum([1 for s in sample_list if is_ip_address(str(s))])
    uuid_count = sum([1 for s in sample_list if is_uuid(str(s))])
    other_count = len(sample_list) - ip_count - uuid_count
    return ip_count, uuid_count, other_count

# Main function to process files using Pandas
def replace_keys_in_file_pandas(input_folder, output_folder, replace_dict, pii_columns):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        input_file_path = os.path.join(input_folder, filename)
        output_file_path = os.path.join(output_folder, filename)
        
        if os.path.isdir(input_file_path):
            replace_keys_in_file_pandas(os.path.join(input_folder, filename), os.path.join(output_folder, filename), replace_dict, pii_columns)
            continue

        if filename.endswith('.csv'):
            replace_pii_one_csv(filename, output_folder, input_folder, replace_dict, pii_columns)



def replace_pii_one_csv(filename:str, output_folder:str, input_folder, replace_dict, pii_columns):
    output_file_path = os.path.join(output_folder, filename)
    input_file_path = os.path.join(input_folder, filename)
    if os.path.exists(output_file_path):
        print(f"File {filename} already exists in {output_folder}, skipping...")
        return 

    print("-" * 50)
    print("-" * 50)
    print(f"Processing {filename}...")
    start_time = time.time()

    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_file_path, sep=SEPARATOR, encoding='utf-8', on_bad_lines='skip', engine='python')

    # Iterate over each column and classify unique values
    for column in df.columns:
        if column in ["TenantId", "TimeGenerated", "Timestamp"]:
            continue
        # print(f"{column},", end="")

        # Get unique values from the column (sample 100 or all if fewer)
        unique_values = df[column].dropna().unique()
        sample_size = min(100, len(unique_values))
        sample_list = unique_values[:sample_size]

        # Classify the sample
        ip_count, uuid_count = 0,0
        is_dict = False
        for s in sample_list:
            if not is_dict and  "{" in str(s) and "}" in str(s):
                try: 
                    json.loads(str(s))
                    is_dict = True
                except:
                    pass
                    
            if len(extract_ip_from_string(str(s))) > 0:
                ip_count += 1
            elif len(extract_uuid_from_string(str(s))) > 0:
                uuid_count += 1
        # for s in sample_list[:5]:
        #     print(s)

        # if 0.8 type is not str skip
        if sum([1 for s in sample_list if isinstance(s, str)]) < 0.8 * sample_size:
            # print(f"Skip {column}")
            continue

        replace_time = time.time()

        # Handle dict-type strings
        if is_dict:
            # print("Identified as dictionary")
            tmp_dict = replace_dict['other']
            if ip_count > 0:
                print("> add ip")
                tmp_dict.update(replace_dict['ip'])
            elif uuid_count > 0:
                print("> add uuid")
                tmp_dict.update(replace_dict['uuid'])

            # df[column] = df[column].str.replace(tmp_dict, regex=False)
            # Create a regular expression that matches any key in the dictionary
            pattern = re.compile('|'.join(re.escape(key) for key in tmp_dict.keys()))

            # Define a function that will replace matched patterns using the dictionary
            def multi_replace(match):
                a = tmp_dict[match.group(0)]
                # print(a)
                return a
            
            # Apply the multi_replace function to each cell in the column
            df[column] = df[column].apply(lambda x: pattern.sub(multi_replace, x) if isinstance(x, str) else x)

            print(f"> {column}: Dict replace time: {time.time() - replace_time:.2f} seconds")

        # Handle non-dict strings
        elif column in pii_columns:
            # must be in pii_columns
            if ip_count > 0.5 * sample_size:
                # print(f" {column} -> IP address")
                df[column] = df[column].replace(replace_dict['ip'])
            elif uuid_count > 0.5 * sample_size:
                # print(f" {column} -> UUID")
                df[column] = df[column].replace(replace_dict['uuid'])
                print(f"> {column}: UUID replace time: {time.time() - replace_time:.2f} seconds")
            else:
                # print(f" {column} -> Other")
                df[column] = df[column].replace(replace_dict['other'])
                print(f"> {column}: Other replace time: {time.time() - replace_time:.2f} seconds")
        else:
            print(f"> Skip {column}:", ", ".join(sample_list[:5]))

    # Save the modified DataFrame to the output folder
    df.to_csv(output_file_path, sep=SEPARATOR, index=False, encoding='utf-8')
    print()
    print(f"Processed and saved {filename} to {output_folder} in {time.time() - start_time:.2f} seconds")
    print("-" * 50) 

# Call the function
import json
with open("./pii/final_filter.json", "r") as f:
    pii_columns = json.load(f)

with open("./pii/classified_list.json", "r") as f:
    classified_list = json.load(f)


folders = [
    "./data/incidents/incident_5",
    "./data/incidents/incident_38",
    "./data/incidents/incident_34",
    "./data/incidents/incident_39",
    "./data/incidents/incident_55",
    "./data/incidents/incident_122",
    "./data/incidents/incident_134",
    "./data/incidents/incident_166",
    "./data/incidents/incident_322",
    "./data/alphineskihouse",
]


for f in folders:
    new_folder = f.replace("./data/", "./data_anonymized/")
    print("*" * 70)
    print("*" * 70)
    print(f"Processing {f}...")

    replace_keys_in_file_pandas(f, new_folder, classified_list, pii_columns)