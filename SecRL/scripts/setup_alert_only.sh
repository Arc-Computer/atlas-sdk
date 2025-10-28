# python secgym/database/setup_database.py \
#     --csv data_anonymized/alphineskihouse \
#     --port 3314 \
#     --sql_file sql_files/alpineskihouse.sql \
#     --container_name alpineskihouse \
#     --respawn \
#     --layer alert_only

python secgym/database/setup_database.py \
  --csv data_anonymized/incidents/incident_5 \
  --port 3306 \
  --sql_file sql_files/incident_5.sql \
  --container_name incident_5 \
  --respawn \
  --layer alert_only


python secgym/database/setup_database.py \
    --csv data_anonymized/incidents/incident_38 \
    --port 3307 \
    --sql_file sql_files/incident_38.sql \
    --container_name incident_38 \
    --respawn \
    --layer alert_only

python secgym/database/setup_database.py \
    --csv data_anonymized/incidents/incident_34 \
    --port 3308 \
    --sql_file sql_files/incident_34.sql \
    --container_name incident_34 \
    --respawn \
    --layer alert_only

python secgym/database/setup_database.py \
    --csv data_anonymized/incidents/incident_39 \
    --port 3309 \
    --sql_file sql_files/incident_39.sql \
    --container_name incident_39 \
    --respawn \
    --layer alert_only

python secgym/database/setup_database.py \
    --csv data_anonymized/incidents/incident_55 \
    --port 3310 \
    --sql_file sql_files/incident_55.sql \
    --container_name incident_55 \
    --respawn \
    --layer alert_only

python secgym/database/setup_database.py \
    --csv data_anonymized/incidents/incident_134 \
    --port 3311 \
    --sql_file sql_files/incident_134.sql \
    --container_name incident_134 \
    --respawn \
    --layer alert_only

python secgym/database/setup_database.py \
    --csv data_anonymized/incidents/incident_166 \
    --port 3312 \
    --sql_file sql_files/incident_166.sql \
    --container_name incident_166 \
    --respawn \
    --layer alert_only

python secgym/database/setup_database.py \
    --csv data_anonymized/incidents/incident_322 \
    --port 3313 \
    --sql_file sql_files/incident_322.sql \
    --container_name incident_322 \
    --respawn \
    --layer alert_only
