import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'google-keys.json')
from google.cloud import bigquery
import matplotlib.pyplot as plt
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
import pandas as pd
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

def trend(df_train, df_test, column='sales', order=1):
    # order=2 quadratic
    X = DeterministicProcess(index=df_train.index, constant=True, order=order, drop=True).in_sample()
    trend_model = LinearRegression(fit_intercept=False)
    trend_model.fit(X, df_train[column])

    X = DeterministicProcess(index=df_test.index, constant=True, order=order, drop=True).in_sample()
    return pd.Series(trend_model.predict(X), index=X.index)

def moving_average(df, column='sales', window=30):
    return df[column].rolling(window=window, center=True, min_periods=int(window/2)).mean()

def seasonly(df_train, df_test, column='sales'):
    fourier = CalendarFourier(freq='W', order=2) 
    X = DeterministicProcess(index=df_train.index, constant=True, order=1, seasonal=True, additional_terms=[fourier], drop=True).in_sample()
    trend_model = LinearRegression(fit_intercept=False)
    trend_model.fit(X, df_train[column])

    X = DeterministicProcess(index=df_test.index, constant=True, order=1, seasonal=True, additional_terms=[fourier], drop=True).in_sample()
    return pd.Series(trend_model.predict(X), index=X.index)

def cycle(df_train, df_test, column='sales'):
    df_train['lag'] = df_train[column].shift(1)
    Xy = df_train.dropna()
    X = Xy.loc[:, ['lag']]
    y = Xy.loc[:, 'sales']
    trend_model = LinearRegression(fit_intercept=False)
    trend_model.fit(X, y)

    df_test['lag'] = df_test[column].shift(1)
    Xy = df_test.dropna()
    X = Xy.loc[:, ['lag']]
    return pd.Series(trend_model.predict(X), index=X.index)

part = -50
df = query.to_dataframe()
df['date'] = pd.to_datetime(df['agr_date'])
df.set_index('date', inplace=True)
df.index = df.index.to_period('D')

df_train, df_test = df[:part+1],  df[part:]

df['trdA'] = trend(df_train, df, order=1)
df['trd'] = trend(df_train, df_test, order=1)
df['avg'] = moving_average(df_train, window=30)
df['ssn'] = seasonly(df_train, df_test)
df['ccl'] = cycle(df_train, df_test)

fig = plt.figure(figsize=(10, 6))
plt.plot(df_train['agr_date'], df_train['sales'], color='orange', linewidth=1)
plt.plot(df_test['agr_date'], df_test['sales'], color='grey', linewidth=1)
plt.plot(df['agr_date'], df['trdA'], color="orange", linewidth=1, label="TradeA")
plt.plot(df['agr_date'], df['trd'], color="orange", linewidth=1, label="Trade")
plt.plot(df['agr_date'], df['avg'], color="blue", linewidth=1, label='AVG')
plt.plot(df['agr_date'], df['ssn'], color="green", linewidth=1, label='Season')
plt.plot(df['agr_date'], df['ccl'], color="red", linewidth=1, label='Cycle')
plt.legend()
plt.show()
