# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import mysql.connector
from mysql.connector import Error
import time
import os
import json
import docker
from docker.errors import ContainerError, ImageNotFound, APIError, NotFound
import pandas as pd
# from secgym.utils import find_most_similar
import argparse

from secgym.database.process_logs import SEPARATOR, QUOTECHAR

def to_abs_path(path):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)

def create_container_per_attack(attack):
    # TODO
    attack_maps = {
        "attack 1" : "attack1container",
    }
    for i, key, value in enumerate(attack_maps.items()):
        create_container(
            csv_folder=to_abs_path(f"data/logs_jun3_30/"),
            sql_file_path=to_abs_path(f"data/logs_jun3_30/.sql"),
            database_name=f"env_monitor_db",
            container_name=value,
            port=str(3306 + i),
            respawn=True
        )

def create_container(
    csv_folder,
    sql_file_path,
    database_name,
    port="3306",
    container_name="mysql-container",
    respawn=False
):
    client = docker.from_env()
    if respawn:
        # delete the existing container if it exists
        try:
            container = client.containers.get(container_name)
            if container.status == "running":
                print(f"Stopping container {container_name}...")
                container.stop()
            container.remove()
            print(f"Stopped and removed existing container {container_name}")
            time.sleep(2)  # Wait for the container to be removed
        except NotFound:
            pass

    try:
        container = client.containers.get(container_name)
        if container.status == "running":
            print(f"Container {container_name} is already running.")
            return container, port
        else:
            container.start()
            print(f"Restarted stopped container with ID: {container.id}")
            return container, port
    except NotFound:
        print(f"Container {container_name} does not exist. Creating a new container...")
        try:
            container = client.containers.run(
                "mysql:9.0",
                name=container_name,
                environment={
                    "MYSQL_ROOT_PASSWORD": "admin",
                    "MYSQL_DATABASE": database_name,
                    "MYSQL_ROOT_HOST": "%"
                },
                ports={
                    "3306/tcp": port
                },
                volumes={
                    to_abs_path(sql_file_path): {
                        'bind': '/docker-entrypoint-initdb.d/init_data.sql',
                        'mode': 'ro'  # Mount the file as read-only
                    },
                    to_abs_path(csv_folder): {
                        'bind': '/var/lib/mysql-files',
                        'mode': 'rw'  # Mount the CSV folder as read-write
                    }
                },
                command=['--secure-file-priv=/var/lib/mysql-files'],
                detach=True
            )
            print(f"Started container {container_name} with ID: {container.id} on port {port}")
            container.reload()

            # check log to see "ready for connections"
            print("Waiting for the container to start...")

            time_limit = 600
            time_step = 2
            while time_limit > 0:
                container.reload()
                if container.status == "exited":
                    print(f"Container {container_name} has exited due to an error.")
                    break
                logs = container.logs().decode('utf-8')
                if "MySQL init process done. Ready for start up" in logs:
                    print(f"Container {container_name} is ready.")
                    break

                time.sleep(time_step)
                time_limit -= time_step
            
            time.sleep(4)  # Wait for the container to be ready
            # if time_limit == 0:
            #     raise Exception(f"Container {container_name} did not start within the time limit.")

            return container, port
        except (ContainerError, ImageNotFound) as e:
            print(f"Error: {e}")
            raise

def create_sql_file_from_csv_folder(
        csv_folder, 
        sql_file_path, 
        database_name,
        skip_tables=["SecurityAlert", "SecurityIncident"],
        verbose=False
        ):
    """Create a single .sql file from a folder of CSV and .meta files.

    Args:
        csv_folder (str): Path to the folder containing the CSV and .meta files.
        sql_file_path (str): Path to the output SQL file.
        database_name (str): Name of the database to create.
        skip_tables (list): List of table names to skip.
    """
    sql_statements = [
        "CREATE USER 'admin'@'%' IDENTIFIED BY 'admin';",
        "GRANT ALL PRIVILEGES ON *.* TO 'admin'@'%';",
        "FLUSH PRIVILEGES;",
        f"CREATE DATABASE IF NOT EXISTS {database_name};",
        f"USE {database_name};"
    ]
    for file_name in os.listdir(csv_folder):

        #skipping apple metadata stuff
        if file_name.startswith("._"):
            continue    
            
        if file_name.replace(".csv", "").strip() in skip_tables: 
            if verbose:
                print(f"Skipping table {file_name}")
            continue
        if file_name.endswith(".csv"):
            table_name = file_name.replace(".csv", "")
            
            # check meta file exists
            if os.path.exists(os.path.join(csv_folder, f"{table_name}.meta")):
                with open(os.path.join(csv_folder, f"{table_name}.meta"),  'r') as meta_file:
                    type_map = json.load(meta_file)
            else:
                print(f"Meta file not found for {table_name}. Inferring types from the CSV file...")
                df = pd.read_csv(os.path.join(csv_folder, f"{table_name}.csv"), sep=SEPARATOR, encoding='utf-8-sig', on_bad_lines='skip', engine='python')
                type_map = {str(col): "string" for col in df.columns}
                # print(type_map)

            json_columns = [col for col, dtype in type_map.items() if dtype == "dynamic"]

            # Generate CREATE TABLE statement
            create_table_sql = generate_create_table_sql(table_name, type_map)
            sql_statements.append(create_table_sql)

            # Generate LOAD DATA INFILE statement
            load_data_sql = generate_load_data_sql(file_name, table_name, type_map.keys(), json_columns)
            sql_statements.append(load_data_sql)
        elif os.path.isdir(os.path.join(csv_folder, file_name)):
            # a folder of CSV files. file_name is the table name: SecurityAlert
            # In the folder, there are SecurityAlert_i.csv starting from 1 and a SecurityAlert.meta file
            # add the corresponding SQL statements: creating one table and loading all the data

            table_name = file_name
            with open(os.path.join(csv_folder, f"{file_name}/{file_name}_0.meta"), 'r') as meta_file:
                type_map = json.load(meta_file)
            json_columns = [col for col, dtype in type_map.items() if dtype == "dynamic"]

            # get all the csv files
            csv_files = [f"{file_name}/{f}" for f in os.listdir(os.path.join(csv_folder, file_name)) if f.endswith(".csv")]

            # Generate CREATE TABLE statement
            create_table_sql = generate_create_table_sql(table_name, type_map)
            sql_statements.append(create_table_sql)

            # Generate LOAD DATA INFILE statement
            for i, csv_file in enumerate(csv_files):
                load_data_sql = generate_load_data_sql(csv_file, table_name, type_map.keys(), json_columns)
                sql_statements.append(load_data_sql)
           
    # Write all SQL statements to the output file
    with open(sql_file_path, 'w', encoding='utf-8') as sql_file:
        sql_file.write("\n\n".join(sql_statements))

def dtype_to_sql(dtype):
    """Convert a custom dtype to a SQL data type."""
    mapping = {
        "string": "TEXT",
        "long": "TEXT", # BIGINT
        "datetime": "TEXT",
        "bool": "TEXT", # BOOLEAN
        "dynamic": "TEXT" # JSON
    }
    return mapping.get(dtype, "TEXT")

def generate_create_table_sql(table_name, type_map):
    """Generate a CREATE TABLE SQL statement."""
    column_defs = [f"{col} {dtype_to_sql(dtype)}" for col, dtype in type_map.items()]
    columns_sql = ",\n    ".join(column_defs)
    sql = f"CREATE TABLE {table_name} (\n    {columns_sql}\n);"
    if table_name == "Usage":
        sql = sql.replace("Usage", "`Usage`")
    return sql

def generate_load_data_sql(file_name, table_name, columns, json_columns):
    """Generate LOAD DATA INFILE SQL statement."""
    separator = SEPARATOR
    quotechar = QUOTECHAR
    load_data_sql = f"""
LOAD DATA INFILE '/var/lib/mysql-files/{file_name}'
INTO TABLE {table_name}
FIELDS TERMINATED BY '{separator}'
ENCLOSED BY '{quotechar}'
LINES TERMINATED BY '\\n'
IGNORE 1 ROWS;
"""
    load_data_sql = load_data_sql.replace("INTO TABLE Usage", "INTO TABLE `Usage`")
    set_statements = []
    for column in columns:
        if column in json_columns:
            # convert to {} if the column is empty
            set_statements.append(f"CASE WHEN @{column} = '' THEN '{{}}' ELSE @{column} END,")
        else:
            set_statements.append(f"{column} = NULLIF(@{column}, '')")

    tmp_set = ','.join(set_statements)
    # load_data_sql += f"""SET {tmp_set};"""
    return load_data_sql

def debug_tables(args):
    # remove one table from the list, compile and see if it works
    log_list = [
        "AADNonInteractiveUserSignInLogs",
        "AADServicePrincipalSignInLogs",
        "AuditLogs",
        "AZFWDnsQuery",
        "AzureDiagnostics",
        "AzureMetrics",
        "CloudAppEvents",
        "DeviceEvents",
        "DeviceFileEvents",
        "DeviceImageLoadEvents",
        "DeviceNetworkEvents",
        "DeviceProcessEvents",
        "DeviceRegistryEvents",
        "IdentityLogonEvents",
        "LAQueryLogs",
        "MicrosoftGraphActivityLogs",
        "SecurityAlert",
        "SigninLogs",
        "ThreatIntelligenceIndicator"
    ]
    skip_tables=["SecurityAlert", "SecurityIncident", "AzureDiagnostics", "LAQueryLogs"]
    error_list = []

    for i in range(len(log_list)):
        print(len(log_list))
        # make copy of the log_list
        log_list_copy = log_list.copy()
        # remove the ith element a
        removed_table = log_list_copy.pop(i)
        if removed_table in skip_tables:
            print(f"Skipping table {removed_table}")
            continue

        create_sql_file_from_csv_folder(
            csv_folder=csv_folder,
            sql_file_path=sql_file_path,
            database_name=args.database_name,
            skip_tables=log_list_copy
        )
        
        print(f"Processing table {removed_table}", end="...")

        # 2. start a MySQL docker container
        # print("> 2 Starting a MySQL container...")
        container, port = create_container(
            csv_folder=csv_folder,
            sql_file_path=sql_file_path,
            database_name=args.database_name,
            container_name=args.container_name,
            port=args.port,
            respawn=args.respawn
        )

        if container.status == "exited":
            error_list.append(removed_table)
            print(f"Error with {removed_table}")
        else:
            print(f"Success with {removed_table}")
        
        os.system("docker volume prune -f")
    print("Errors:", error_list)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Setup a MySQL database from CSV files')
    parser.add_argument('--csv', type=str, help='Folder containing the CSV files')
    parser.add_argument('--port', type=str, help='Port number for the MySQL container')
    parser.add_argument('--sql_file', type=str, help='Output SQL file')
    parser.add_argument('--container_name', type=str, help='Name of the MySQL container')
    parser.add_argument('--database_name', type=str, default="env_monitor_db", help='Name of the database')
    parser.add_argument('--respawn', action='store_true', help='Delete and recreate the container')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--layer', type=str, default="alert", help='Layer to use for the agent')
    args = parser.parse_args()
    # make sure the data is downloaded and stored in the 'large_data' folder

    csv_folder = to_abs_path(args.csv)
    sql_file_path = to_abs_path(args.sql_file)
    if args.layer == "alert_only":
        sql_file_path = sql_file_path.replace(".sql", "_alert_only.sql")
        args.container_name = args.container_name+"_alert_only"
    os.makedirs(os.path.dirname(sql_file_path), exist_ok=True)
    print(os.path.dirname(sql_file_path))
    
    if args.debug:
        args.respawn = True
        debug_tables(args)
        exit(0)

    # - Log level: minimum info, everything should be excluded
    if args.layer == "log":
        skip_tables = ["AzureDiagnostics", "LAQueryLogs", "SecurityIncident", "SecurityAlert", "AlertEvidence", "AlertInfo"]
    # - Incident level: Have access to security incidents, but not the alerts
    elif args.layer == "incident":
        skip_tables = ["AzureDiagnostics", "LAQueryLogs", "SecurityAlert", "AlertEvidence", "AlertInfo"]
    elif args.layer == "alert":
        # - Alert level: Have access to all:
        skip_tables = ["AzureDiagnostics", "LAQueryLogs"]
    elif args.layer == "alert_only":
        skip_tables = []
        for fname in os.listdir(csv_folder):
            if fname.endswith(".meta") or fname.startswith("._") or fname.startswith(".DS_Store"):
                continue
            if fname.endswith(".csv"):
                skip_tables.append(fname.replace(".csv", ""))
            elif os.path.isdir(os.path.join(csv_folder, fname)):
                skip_tables.append(fname)
            else:
                raise ValueError(f"Invalid file type: {fname}")

            # remove SecurityIncident", "SecurityAlert", "AlertEvidence", "AlertInfo" from the list
        skip_tables.remove("SecurityIncident")
        skip_tables.remove("SecurityAlert")
        skip_tables.remove("AlertEvidence")
        skip_tables.remove("AlertInfo")
    else:
        raise ValueError(f"Invalid layer: {args.layer}")

    # 1. create a .sql file from the CSV  filesin the 'large_data' folder
    #skip_tables = ["AzureDiagnostics", "LAQueryLogs", "SecurityIncident"] #TODO: add "AlertEvidence", "AlertInfo","SecurityAlert"
    # skip_tables += ["DeviceFileEvents"]
    create_sql_file_from_csv_folder(
        csv_folder=csv_folder,
        sql_file_path=sql_file_path,
        database_name=args.database_name,
        skip_tables=skip_tables,
        verbose=True
    )
    print(f"> 1. SQL file created: {sql_file_path}")

    # 2. start a MySQL docker container
    print("> 2 Starting a MySQL container...")
    container, port = create_container(
        csv_folder=csv_folder,
        sql_file_path=sql_file_path,
        database_name=args.database_name,
        container_name=args.container_name,
        port=args.port,
        respawn=args.respawn
    )

    if container.status == "exited":
        print(f"Error: Container {args.container_name} has exited due to an error.")
        exit(1)
    # 3. test connection to the MySQL container
    connection = mysql.connector.connect(
        host='localhost',
        port=port,
        user='root',
        password='admin'
    )
    print("> 3. Successfully connected to the database")

    print("> 4. Testing connection to the MySQL container...")
    if connection.is_connected():
        cursor = connection.cursor()
        # select the database
        cursor.execute(f"USE {args.database_name};")
        # list all tables in the database
        cursor.execute("SHOW TABLES;")
        tables = cursor.fetchall()
        print("Tables in the database:", tables)
    
    print("> 5. Stopping the MySQL container...")
    container.stop()