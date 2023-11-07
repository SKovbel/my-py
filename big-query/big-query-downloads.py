# go to the Google Cloud Console.
# to "IAM & Admin" > "Service accounts."
# Create new for BigQuery
import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'google-keys.json')

from google.cloud import bigquery

# Initialize a BigQuery client
client = bigquery.Client()

limit = 100
offset = 0

script_dir = os.path.dirname(os.path.abspath(__file__))
local_file_path = os.path.join(script_dir, '../var/stackoverflow_posts.csv')

with open(local_file_path, 'w') as outfile:

    while True:
        query = f"""
        SELECT *
        FROM `bigquery-public-data.stackoverflow.stackoverflow_posts`
        LIMIT {limit}
        OFFSET {offset}
        """

        print(query)

        query_job = client.query(query)
        result = query_job.result()
        offset += limit

        for row in query_job:
            outfile.write(','.join(map(str, row)) + '\n')

        if result.total_rows != limit:
            break
        break

    outfile.close()