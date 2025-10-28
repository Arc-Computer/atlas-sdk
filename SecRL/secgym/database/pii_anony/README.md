# PII Anonymization 


## 1. Indentify PII Columns

- Checkout out `pii_col_identify.py`.

First we take out each column name, and sample 5 examples to ask LLM to identify where a column is PII or not (`examined_columns.json`). On top of these identified columns, we ask LLM again to check the following:
1. Whether the column is PII the second time.
2. Whether the column is a dict field.
3. If it is a dict field, what fields contain PII. 

(`second_filter.json`)


After the two calls to LLM, we manually go through the tables and the identified columns to validate and refine the results. (`final_filter.json`)


## 2. PII Value Mapping

Next, we want to create a mapping of PII values to anonymized values. For example, with a PII name "John", we may want to map it to "Javier" or some other name.

**Method** We iterate over all the tables and columns that are identified as PII, take out the unique values: 
- For non-dict columns, we take out all unique values, 
- For dict columns, we take out the unique values that are in the identified PII fields.

We then use regex to map each value to see whether it is a known PII like IP, IPv6, email, uuid, mac address, and some observed PII values like latitude.
For these known PII values, we have a correponding random generator or modification function to generate a new value. If the value is not a known PII, we add it to a batch list to be processed by LLM, that the LLM will generate an anonymized value. We set the batch size to 10.

All these anonymized values are stored in a dict, and we will check whether a new value is in the dict before anonymizing it. The results are stored in `hashlist.json`.

After the mapping, we classify the PII to different categories like ip, email, others (processed by LLM). We again examine the mapping and get rid of unnecessary mappings such as "u234", a number, or a hash value, which is stored in `filtered_hashlist.json`. (`pii_hashlist_filter.ipynb`)

We note that a majority of the PII values are ip addresses (156838), with a total of 164888 PII values.


## 3. Anonymize PII
- File: `pii_replace.py`
We then go through all tables, read in the csv file and do a global replacement of all the mapped PII values. This is for consistency and to avoid any query errors, for example, when querying the database using ond anonymized ip address.


