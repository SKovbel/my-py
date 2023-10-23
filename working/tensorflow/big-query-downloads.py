from google.cloud import bigquery

# Initialize a BigQuery client
client = bigquery.Client()

# Define your SQL query
query = """
SELECT *
FROM `bigquery-public-data.stackoverflow.stackoverflow_posts`
"""

# Run the query and fetch the results into a Pandas DataFrame
df = client.query(query).to_dataframe()