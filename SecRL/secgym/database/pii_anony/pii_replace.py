# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import time
import json
import argparse
import random

def replace_keys_in_file(input_folder, output_folder, replace_dict):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # get all files
    files = os.listdir(input_folder)
    random.shuffle(files)
    # randomize order of files
    
    for filename in files:
        # if not any(  in filename): ["SecurityIncident.csv", "SecurityAlert"]
        # if not any([x in filename for x in ["SecurityIncident.csv", "SecurityAlert"]]):
        #     continue
        input_file_path = os.path.join(input_folder, filename)
        output_file_path = os.path.join(output_folder, filename)

        if not os.path.isdir(input_file_path):
            with open("queue.txt", "r") as f:
                queue = f.read().splitlines()
            if input_file_path in queue:
                print(f"File {filename} already in queue, skipping...", flush=True)
                continue
            with open("queue.txt", "a") as f:
                f.write(f"{input_file_path}\n")

        try: 
            # If the current path is a directory, recursively process its contents
            if os.path.isdir(input_file_path):
                replace_keys_in_file(input_file_path, output_file_path, replace_dict)
                continue

            # Process only CSV files
            if filename.endswith('.csv'):
                # Skip if file already exists in the output folder
                if os.path.exists(output_file_path):
                    print(f"File {filename} already exists in {output_folder}, skipping...", flush=True)
                    continue

                # Read the entire CSV file as text
                with open(input_file_path, 'r', encoding='utf-8') as file:
                    file_data = file.read()

                print(f"Processing {filename}...", flush=True)
                start_time = time.time()

                # Replace all occurrences of keys in the file content
                for key, value in replace_dict.items():
                    file_data = file_data.replace(key, value)

                # Write the modified content to the output file
                with open(output_file_path, 'w', encoding='utf-8') as file:
                    file.write(file_data)

                print(f"Processed and saved {filename} to {output_folder}", f"Time taken: {time.time() - start_time:.2f} seconds", flush=True)
        except Exception as e:
            print(f"Error processing {filename}: {e}", flush=True)

            # remove the file from the queue
            with open("queue.txt", "r") as f:
                queue = f.read().splitlines()
            queue.remove(input_file_path)
            with open("queue.txt", "w") as f:
                f.write("\n".join(queue))

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anonymize PII data in CSV files.")
    parser.add_argument("--folder", '-f', type=str, help="Folder containing CSV files to anonymize.")
    args = parser.parse_args()

    with open("./pii/filtered_hashlist.json", "r") as f:
        hashlist = json.load(f)

    # remove key and value same pairs from the hashlist

    for key, value in hashlist.copy().items():
        if key == value:
            del hashlist[key]

    if "./data" not in args.folder:
        raise ValueError("Folder must be inside the data directory.")

    new_folder = args.folder.replace("./data/", "./data_anonymized/")
    print("*" * 70, flush=True)
    print("*" * 70, flush=True)
    print(f"Processing {args.folder}...", flush=True)
    replace_keys_in_file(args.folder, new_folder, hashlist)
