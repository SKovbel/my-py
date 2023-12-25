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

dir = os.path.dirname(os.path.abspath(__file__))
local_file_path = os.path.join(dir, 'stackoverflow_posts.csv')

with open(local_file_path, 'w') as outfile:

    while True:
        print(offset)
        query = """
        SELECT *
        FROM `bigquery-public-data.stackoverflow.stackoverflow_posts`
        LIMIT @limit
        OFFSET @offset
        """

        query_job = client.query(query, job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
                bigquery.ScalarQueryParameter("offset", "INT64", offset)
            ]
        ))

        result = query_job.result()
        offset += limit
        
        for row in query_job:
            outfile.write(','.join(map(str, row)) + '\n')

        if result.total_rows != limit:
            break
        break

    outfile.close()