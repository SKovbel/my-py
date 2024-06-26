import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'google-keys.json')
from google.cloud import bigquery
import matplotlib.pyplot as plt
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

client = bigquery.Client()

query = client.query(f"""
    SELECT DATE(date) AS agr_date,
           EXTRACT(WEEK FROM DATE(date)) AS week,
           EXTRACT(DAYOFWEEK FROM DATE(date)) AS day,
           COUNT(*) AS items,
           SUM(sale_dollars) AS sales
      FROM bigquery-public-data.iowa_liquor_sales.sales
     WHERE date >= @from_date AND date <= @to_date
     GROUP BY agr_date, week, day
     ORDER BY agr_date
""", job_config=bigquery.QueryJobConfig(
    query_parameters=[
        bigquery.ScalarQueryParameter("from_date", "DATE", "2022-01-01"),
        bigquery.ScalarQueryParameter("to_date", "DATE", "2022-12-31")
    ]
))

def trend(df, df_train, column='sales', order=1):
    # order=2 quadratic
    X = DeterministicProcess(index=df_train.index, constant=True, order=order, drop=True).in_sample()
    trend_model = LinearRegression(fit_intercept=False)
    trend_model.fit(X, df_train[column])

    X = DeterministicProcess(index=df.index, constant=True, order=order, drop=True).in_sample()
    return pd.Series(trend_model.predict(X), index=X.index)

def moving_average(df, column='sales', window=30):
    return df[column].rolling(window=window, center=True, min_periods=int(window/2)).mean()

def seasonly(df, df_train, column='sales'):
    fourier = CalendarFourier(freq='W', order=2) 
    X = DeterministicProcess(index=df_train.index, constant=True, order=1, seasonal=True, additional_terms=[fourier], drop=True).in_sample()
    trend_model = LinearRegression(fit_intercept=False)
    trend_model.fit(X, df_train[column])

    X = DeterministicProcess(index=df.index, constant=True, order=1, seasonal=True, additional_terms=[fourier], drop=True).in_sample()
    return pd.Series(trend_model.predict(X), index=X.index)

part = -50
df = query.to_dataframe()
df['date'] = pd.to_datetime(df['agr_date'])
df.set_index('date', inplace=True)
df.index = df.index.to_period('D')

df['lag_items'] = df['items'].shift(1)
df['lag_sales'] = df['sales'].shift(1)

df_train, df_test = df[:part+1],  df[part:]

df['trend'] = trend(df, df_train, order=1)
df['avg'] = moving_average(df, window=30)
df['seas'] = seasonly(df, df_train)

print(df_train)

plt.figure(figsize=(10, 6))
plt.plot(df_train['agr_date'], df_train['sales'], color='orange')
plt.plot(df_test['agr_date'], df_test['sales'], color='grey')
plt.plot(df['agr_date'], df['trend'], color="orange", linestyle='-.')
plt.plot(df['agr_date'], df['avg'], color="blue")
plt.plot(df['agr_date'], df['seas'], color="green", linewidth=1)
plt.xlabel('Date')
plt.ylabel('Count')
plt.title('DataFrame 1')
plt.xticks(rotation=45)
plt.show()
