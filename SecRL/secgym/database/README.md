
# Original Handling Process 

1. Register an account at Alphine Ski House.
2. Login in to the azure account.
    ```bash
    az login
    ```
3. Uncomment the code in `download_data.py` to download the Alphine Ski House data or for 8 different incidents.
The `alphineskihouse` folder contains logs from Jun 20, 2024 to Jul 23, 2024. While each incident folder contains logs for that specific incident from the start of incident to the end of the incident.
    ```bash
    python download_logs.py
    ```
4. Run process_logs.py to process the data. This will change the some file's entry from double quotes to single quotes.
    ```bash
    python process_logs.py
    ```
5. To anonymize the data, please refer to `secgym/database/pii_anony/README.md` for the anonymization process.

