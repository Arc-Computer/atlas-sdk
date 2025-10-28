# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import random
import uuid
import re
import os
import json
import pandas as pd
from process_logs import SEPARATOR
from secgym.utils.utils import LLM_call

new_pii_prompt = """Given an input string, you will help create a new string that anonymizes the PII data in the input string.
- For email addresses, don't change the domain name.
- For names, replace with a random name. Please do not use faker names like "Jane Doe", keep it as realistic as possible.
- If it is a random character string, such as j234 or mm46, please also replace it with a random string.
- If it is not a PII and have specific meaning, for example, "MailItemsAccessed", "Microsoft.Compute/virtualMachines", "OneDrive for Business", please keep it as is or skip it.
- If it is u + number, please replace it with u + another number.
Your output should be in json format:
{
    "<orginal_str>" : "<anonymized_str>",
    ...
}

Example
Input: 
joss@vnevado.alpineskihouse.co
Macy Smith
MSFT Kev
MailItemsAccessed
u343
jyura@vnevado.onmicrosoft.com
OneDrive for Business
{
    "joss@vnevado.alpineskihouse.co": "kalex@nevado.alpineskihouse.co"
    "Macy Smith": "Jayce Lee"
    "MSFT Kev": "MSFT Jay"
    "u343": "u42"
    "jyura@vnevado.onmicrosoft.com" : "peteks@vnevado.onmicrosoft.com"
}
"""

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

def anony_lat_lon(input:str):
    # random perturbation
    # convert to float
    ninput = float(input)
    if ninput == 0:
        return str(ninput)
    
    ninput += random.uniform(-4, 4)
    return str(ninput)

def extract_email_from_string(input) -> list:
    # if single 
    return re.findall(r'[\w\.-]+@[\w\.-]+', input)


class PIIAnonymizer:
    def __init__(self, 
                 pii_column_file:str,
                 hashlist_file:str,
                 config_list:list,
                 cache_seed=41
                 ):
        with open(pii_column_file, "r") as f:
            self.pii_columns_dict = json.load(f)
            self.pii_columns = list(self.pii_columns_dict.keys())

        self.hashlist = {}
        self.hashlist_file = hashlist_file
        self.hash_hit = 0
        if os.path.exists(hashlist_file):
            with open(hashlist_file, "r") as f:
                self.hashlist = json.load(f)
                self.revese_hash = {v: k for k, v in self.hashlist.items()}
        else:
            self.hashlist = {}
            self.revese_hash = {}
        
        self.need_manual = {}
            
        self.config_list = config_list
        self.cache_seed = cache_seed
        
        self.batch_in = []
        self.accum_cost = 0
        self.exempt_list = [".exe", ".ps1", ".lockbit", ".xlsx", '.dll', '.doc', '.sys', '.zip', "Microsoft.", "vm-", "Unauthorized", "Service Account", ".7z", "Microsoft ", "microsoftxdr", "LgAAAA", "VNEVADO-Win", "vnevado-win"]

    def match_field_dict(self, input:str, match_fields:list):
        all_matches = set()
        for field in match_fields:
            if not isinstance(input, str):
                print("Convert to string:", input)
                input = str(input)
            pattern = f'"{field}":"([^"]+)"'
            matches = re.findall(pattern, input)
            # print(f'{field}: {matches}')  

            # merge set
            all_matches.update(set(matches))
        return all_matches  


    def match_one_csv(self, csv_file:str):
        df = pd.read_csv(csv_file, sep=SEPARATOR, encoding='utf-8', on_bad_lines='skip', engine='python')

        original_length = 0
        all_matches = set()
        for col in df.columns:
            if col in self.pii_columns:
                uniques = df[col].unique()
                original_length += len(uniques)
                print(f"Column: {col} Unique values: {len(uniques)}")
                if self.pii_columns_dict[col]['is_dict']:
                    pii_fields = self.pii_columns_dict[col]['pii_fields']
                    for cvalue in uniques:
                        matched_strs = self.match_field_dict(cvalue, pii_fields)
                        # print(f"Getting {len(matched_strs)} matches.")
                        all_matches.update(matched_strs)
                else:
                    all_matches.update(set(uniques))

        # filter out hashlist
        filtered_matches = set()
        for match in all_matches:
            if match in self.hashlist:
                self.hash_hit += 1
            elif any([i in match for i in self.exempt_list]):
                self.need_manual[match] = "Exempted"
                continue

            filtered_matches.add(match)

        print(f"Num str for processing: {len(filtered_matches)}")
        print("-" * 20)
        for m in filtered_matches:
            # skip nan
            if pd.isna(m):
                continue
            k = self.convert_value(m)
            if len(k) > 0:
                # print(k)
                self.hashlist.update(k)
        
        # clean up batch_in
        if len(self.batch_in) > 0:
            print("Clean up batch_in")
        for i in range(20):
            if len(self.batch_in) > 0:
                new_gen = self._llm_gen()
                self.hashlist.update(new_gen)
            break   
        
        print("Total cost: ", self.accum_cost, "Hash hit: ", self.hash_hit)
        with open(self.hashlist_file, "w") as f:
            json.dump(self.hashlist, f)

    def _llm_gen(self):
        task = "\n".join(self.batch_in)
        print("Batch in: ", self.batch_in)
        self.batch_in = []
        response, cost = LLM_call(
                instruction=new_pii_prompt,
                task=task,
                config_list=self.config_list,
                response_format={"type": "json_object"},
                cache_seed=self.cache_seed,
                return_cost=True
            )
        self.accum_cost += cost
        
        response_dict = json.loads(response)
        print("Response: ", response_dict)
        for k, v in response_dict.copy().items():   
            if v in self.revese_hash:
                # add to manual
                self.need_manual[k] = task
            else:
                self.revese_hash[v] = k
        
        return response_dict
                
        
    def convert_value(self, input:str):
        # match uuid
        uuids = extract_uuid_from_string(input)
        if len(uuids) > 0:
            # print(f"uuids: {uuids}")
            return { i: generate_uuid() for i in uuids }  
        # match ip
        ips = extract_ip_from_string(input)
        if len(ips) > 0:
            # print(f"ips: {ips}")
            return { i: generate_ip() for i in ips }
        
        # match ipv6
        ipv6s = extract_ipv6_from_string(input)
        if len(ipv6s) > 0:
            # print(f"ipv6s: {ipv6s}")
            return { i: generate_ipv6(i) for i in ipv6s }
        
        # match mac address
        macs = extract_mac_address_from_string(input)
        if len(macs) > 0:
            if not isinstance(macs[0], str):
                print(input)
            # print(f"macs: {macs}")
            return { i: generate_mac() for i in macs }
        
        # match sid
        sids = extract_sid_from_string(input)
        if len(sids) > 0:
            # print(f"sids: {sids}")
            return { i: generate_sid(i) for i in sids }

        # match lat lon
        lat_lon = match_latitute_longitude(input)
        if len(lat_lon) > 0:
            return { i: anony_lat_lon(i) for i in lat_lon }
            
        # match sharepoint url
        sharepoint_user_name = extract_sharepoint_url_account_from_string(input)
        if len(sharepoint_user_name) > 0:
            # check if in hashlist
            for s in sharepoint_user_name:
                if s not in self.hashlist:
                    # add to batch
                    self.batch_in.append(s)
            return {}
        
        # match email
        email = extract_email_from_string(input)
        if len(email) > 0:
            # extract string before @ and check if that str is in hashlist
            for e in email:
                prestr = e.split("@")[0]
                if prestr not in self.hashlist:
                    # add to batch
                    self.batch_in.append(prestr)

        if not (len(sharepoint_user_name) > 0 or len(email) > 0):
            self.batch_in.append(input)

        if len(self.batch_in) < 10:
            return {}

        return self._llm_gen() # call LLM to generate


if __name__ == "__main__":
    from secgym.myconfig import config_list_4o

    anoynimizer = PIIAnonymizer(
        pii_column_file="pii/final_filter.json",
        hashlist_file="pii/hashlist.json",
        config_list=config_list_4o
    )

    folders = [
        # "./data/incidents/incident_5",
        # "./data/incidents/incident_38",
        # "./data/incidents/incident_34",
        # "./data/incidents/incident_39",
        # "./data/incidents/incident_55",
        # "./data/incidents/incident_122",
        # "./data/incidents/incident_134",
        # "./data/incidents/incident_166",
        # "./data/incidents/incident_322",
        "./data/alphineskihouse",
    ]
    # folder = "./data/incidents/incident_5"

    for folder in folders:
        for file_name in os.listdir(folder):
            print("Processing: ", file_name)
            if file_name.endswith(".csv"):
                csv_files = [os.path.join(folder, file_name)]
            elif os.path.isdir(os.path.join(folder, file_name)):
                for f in os.listdir(os.path.join(folder, file_name)):
                    if f.endswith(".csv"):
                        csv_files.append(os.path.join(folder, file_name, f))

            else:
                continue
            
            for c in csv_files:
                print("*" * 50)
                print("*" * 50)
                print("> Generate hashlist for ", c)

                anoynimizer.match_one_csv(c)
        
