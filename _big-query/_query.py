# google-keys.json
# go to the Google Cloud Console.
# to "IAM & Admin" > "Service accounts."
# Create new JSON keys (not API Keys) and Service Account if need
import os
from google.cloud import bigquery

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'google-keys.json')

client = bigquery.Client()

# SQL
query = f"""
    SELECT *
    FROM `bigquery-public-data.stackoverflow.stackoverflow_posts`
    LIMIT @limit
    OFFSET 0
"""
# SELECT EXTRACT(YEAR FROM datetime_field) FROM
# SELECT EXTRACT(DAYOFWEEK FROM datetime_field) FROM

query_job = client.query(query, job_config=bigquery.QueryJobConfig(
    maximum_bytes_billed=10**10,
    query_parameters=[
        bigquery.ScalarQueryParameter("limit", "INT64", "10"),
        #bigquery.ScalarQueryParameter("gender", "STRING", "M"),
        #bigquery.ArrayQueryParameter("states", "STRING", ["WA", "WI", "WV", "WY"]),
        #bigquery.ScalarQueryParameter("ts_value", "TIMESTAMP", datetime.datetime(2016, 12, 7, 8, 0, tzinfo=pytz.UTC),)
    ]
))

result = query_job.result()
i = 0
rides_per_year_result = query_job.to_dataframe() # Your code goes here

for row in query_job:
    print(row)
    break
