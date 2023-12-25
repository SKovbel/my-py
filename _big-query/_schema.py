# google-keys.json
# go to the Google Cloud Console.
# to "IAM & Admin" > "Service accounts."
# Create new JSON keys (not API Keys) and Service Account if need
import os
from google.cloud import bigquery

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'google-keys.json')

client = bigquery.Client()

# TABLES LIST
dataset_ref = client.dataset("stackoverflow", project="bigquery-public-data")
dataset = client.get_dataset(dataset_ref)
tables = list(client.list_tables(dataset))
for table in tables:  
    print(table.table_id)

# TABLE INFO
table_ref = dataset_ref.table("stackoverflow_posts")
table = client.get_table(table_ref)
print(table.schema)

# LISG ROWS
rows = client.list_rows(table, max_results=5).to_dataframe()
print(rows)