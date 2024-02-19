import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'google-keys.json')
from google.cloud import bigquery
import matplotlib.pyplot as plt
from statsmodels.tsa.deterministic import DeterministicProcess
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

client = bigquery.Client()

query = client.query(f"""
    SELECT DATE(date) AS day,
           COUNT(*) AS items,
           SUM(sale_dollars) AS sales
      FROM bigquery-public-data.iowa_liquor_sales.sales
     WHERE date >= @from_date AND date <= @to_date
     GROUP BY day
     ORDER BY day
""", job_config=bigquery.QueryJobConfig(
    query_parameters=[
        bigquery.ScalarQueryParameter("from_date", "DATE", "2022-01-01"),
        bigquery.ScalarQueryParameter("to_date", "DATE", "2022-12-31")
    ]
))

def trend(df, column='sales', order=1):
    # order=2 quadratic
    X = DeterministicProcess(index=df.index, constant=True, order=order, drop=True).in_sample()
    trend_model = LinearRegression(fit_intercept=False)
    trend_model.fit(X, df[column])

    X = DeterministicProcess(index=df.index, constant=True, order=order, drop=True).in_sample()
    return pd.Series(trend_model.predict(X), index=X.index)

part = -50
df = query.to_dataframe()
df['lag_items'] = df['items'].shift(1)
df['lag_sales'] = df['sales'].shift(1)
df_train, df_test = df[:part+1],  df[part:]

df['trend'] = trend(df)
df['trend2'] = trend(df, order=2)
df_train['trend'] = trend(df_train)
df_test['trend'] = trend(df_test)


plt.figure(figsize=(10, 6))
plt.plot(df_train['day'], df_train['sales'], color='orange')
plt.plot(df_test['day'], df_test['sales'], color='grey')

plt.plot(df['day'], df['trend'], color="blue")
plt.plot(df['day'], df['trend2'], color="red")
plt.plot(df_train['day'], df_train['trend'], color="blue")
plt.plot(df_test['day'], df_test['trend'], color="blue")

plt.xlabel('Day')
plt.ylabel('Count')
plt.title('DataFrame 1')
plt.xticks(rotation=45)
plt.show()
