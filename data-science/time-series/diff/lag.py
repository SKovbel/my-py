import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'google-keys.json')
from google.cloud import bigquery
import matplotlib.pyplot as plt
from statsmodels.tsa.deterministic import DeterministicProcess
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import seaborn as sns

client = bigquery.Client()

query = client.query(f"""
    SELECT DATE(date) AS agr_date,
           COUNT(*) AS items,
           SUM(sale_dollars) AS sales
      FROM bigquery-public-data.iowa_liquor_sales.sales
     WHERE date >= @from_date AND date <= @to_date
     GROUP BY agr_date
     ORDER BY agr_date
""", job_config=bigquery.QueryJobConfig(
    query_parameters=[
        bigquery.ScalarQueryParameter("from_date", "DATE", "2022-01-01"),
        bigquery.ScalarQueryParameter("to_date", "DATE", "2022-12-31")
    ]
))
df = query.to_dataframe()
df['date'] = pd.to_datetime(df['agr_date'])
df.set_index('date', inplace=True)

#2
df = df.dropna().to_period('D')
num = 10
fig, axes = plt.subplots(num//2, 2, figsize=(10, 6))  # Adjust figsize as needed
for i in range(1, 1+num):
    x = i // 2
    y = i % 2
    axes[x, y].scatter(df['sales'].shift(i), df['sales'], s=1)
plt.show()

#3
df['lag_sales1'] = df['sales'].shift(30)
df['lag_sales2'] = df['sales'].shift(2)

fig, ax = plt.subplots()
ax = sns.regplot(x='lag_sales1', y='sales', data=df, ci=None, scatter_kws=dict(color='0.25'))
ax.set_aspect('equal')
ax.set_title('Lag Plot of Hardcover Sales')
plt.show()

#4
plt.figure(figsize=(10, 6))
plt.plot(df['agr_date'], df['sales'], color='orange')
plt.plot(df['agr_date'], df['lag_sales1'], color="blue")
plt.plot(df['agr_date'], df['lag_sales2'], color="red")
plt.xlabel('Day')

plt.ylabel('Count')
plt.title('DataFrame 1')
plt.xticks(rotation=45)
plt.show()
