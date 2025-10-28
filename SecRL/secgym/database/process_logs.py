# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import pandas as pd


def process_csv(csv_folder):
    """Create a single .sql file from a folder of CSV and .meta files.
    Not used.
    """
    for file_name in os.listdir(csv_folder):
        print(file_name)
        if file_name.endswith(".meta"):
            meta_file_path = os.path.join(csv_folder, file_name)
            table_name = file_name.replace(".meta", "")

            # Read the .meta file
            with open(meta_file_path, 'r') as meta_file:
                meta_data = json.load(meta_file)
                columns = meta_data['columns']
                dtypes = meta_data['dtypes']
            json_columns = [col for col, dtype in zip(columns, dtypes) if dtype == "dynamic"]

            # process with pandas
            # for column that is dynamic, check if empty and convert to {}, and save to the origninal file
            csv_file_path = os.path.join(csv_folder, f"{table_name}.csv")
            with open(csv_file_path, 'r') as csv_file:
                df = pd.read_csv(csv_file, sep=SEPARATOR, encoding='utf-8', on_bad_lines='skip', engine='python')
                for column in df.columns:

                    if column in json_columns:
                        print("Processing column:", column)
                        # convert empty string or None to {}
                        df[column] = df[column].apply(lambda x: "{}" if x == "" else x)
                        df[column] = df[column].apply(lambda x: "{}" if pd.isnull(x) else x)
                        print(df[column].head())
                # df.to_csv(csv_file_path, index=False)
                df.to_csv(csv_file_path, index=False, sep=SEPARATOR, quotechar='"', encoding='utf-8')

            break

def change_separator_in_csv_folder(
        csv_folder, 
        separator,
        quotechar,
        new_separator=None,
        new_quotechar=None,
    ):
    """Switch separator from ¤ to another
    """
    if new_separator is None and new_quotechar is None:
        raise ValueError("Either new_separator or new_quotechar must be provided")
    
    if new_separator is None:
        new_separator = separator
        print("new_separator is None, using separator as new_separator")
    if new_quotechar is None:
        new_quotechar = quotechar
        print("new_quotechar is None, using quotechar as new_quotechar")

    for file_name in os.listdir(csv_folder):
        table_name = file_name.split(".")[0]
        if file_name.endswith(".csv"):
            df = pd.read_csv(os.path.join(csv_folder, f"{table_name}.csv"), sep=separator, quotechar=quotechar, encoding='utf-8-sig', on_bad_lines='skip', engine='python')
            if len(df.columns) == 1:
                print("Error loading for file: ", table_name)
            else:
                df.to_csv(os.path.join(csv_folder, f"{table_name}.csv"), sep=new_separator, quotechar=new_quotechar, index=False, encoding='utf-8-sig')
                print(f"Processed {table_name}.csv")

        elif os.path.isdir(os.path.join(csv_folder, file_name)):
            table_name = file_name
            # get all the csv files
            csv_files = [f"{file_name}/{f}" for f in os.listdir(os.path.join(csv_folder, file_name)) if f.endswith(".csv")]

            for csv_file in csv_files:
                df = pd.read_csv(os.path.join(csv_folder, csv_file), sep=separator, quotechar=quotechar, encoding='utf-8-sig', on_bad_lines='skip', engine='python')
                if len(df.columns) == 1:
                    print("Error loadingfor file: ", csv_file)
                else:
                    df.to_csv(os.path.join(csv_folder, csv_file), sep=new_separator, quotechar=new_quotechar, index=False, encoding='utf-8-sig')
                    print(f"Processed {csv_file} ")

    

def convert_double_quotes(input_file, output_file):
    df = pd.read_csv(input_file, sep="❖", encoding='utf-8', on_bad_lines='skip', engine='python')
    df.replace('"', "'", regex=True, inplace=True)
    df.to_csv(output_file, sep="❖", encoding='utf-8', index=False)


def convert_double_quotes_for_one_folder(folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            # end in csv

            if not file.endswith(".csv"):
                continue
            print(file)
            input_file = os.path.join(root, file)
            convert_double_quotes(input_file, input_file)
    


SEPARATOR = "❖"
QUOTECHAR = '"'

if __name__ == "__main__":

    # convert double quotes for certain files
    need_conversion = [
        "./data/incidents/incident_34/DeviceFileEvents.csv",
        "./data/incidents/incident_55/DeviceRegistryEvents/DeviceRegistryEvents_0.csv",
        "./data/alphineskihouse/DeviceRegistryEvents/DeviceRegistryEvents_3.csv",
        "./data/alphineskihouse/DeviceFileEvents/DeviceFileEvents_2.csv"
    ]

    for file in need_conversion:
        convert_double_quotes(file, file)