
import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'google-keys.json')
from google.cloud import bigquery
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from xgboost import XGBRegressor
from datetime import datetime, timedelta

client = bigquery.Client()
query = client.query(f"""
    SELECT DATE(date) AS date,
           COUNT(*) AS items,
           SUM(sale_dollars) AS sales
      FROM bigquery-public-data.iowa_liquor_sales.sales
     WHERE date >= @from_date AND date <= @to_date
     GROUP BY date
     ORDER BY date
""", job_config=bigquery.QueryJobConfig(
    query_parameters=[
        bigquery.ScalarQueryParameter("from_date", "DATE", "2021-01-01"),
        bigquery.ScalarQueryParameter("to_date", "DATE", "2022-12-31")
    ]
))

sep_date = "2022-07-01"
sep_date_2 = (datetime.strptime(sep_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

df = query.to_dataframe()
df['date'] = pd.to_datetime(df['date'])
df['lag'] = df['sales'].shift(1)
if 1==1:
    df = df.resample('W-Mon', on='date').mean()
else:
    df = df.resample('D', on='date').sum()
df = df.dropna()

df_train, df_valid = df[:sep_date], df[sep_date_2:]

class ModelHybrid:
    def __init__(self):
        self.model_trend = LinearRegression()
        self.model_season = LinearRegression()
        self.model_cycle = XGBRegressor()

        # dp from 2010
        pd_date_range = pd.date_range(start=datetime(2010, 1, 1), end=datetime.now(), freq='D')
        self.dp_trend = DeterministicProcess(index=pd_date_range, order=1).in_sample()
        fourier = CalendarFourier(freq='M', order=1)
        self.dp_season = DeterministicProcess(index=pd_date_range, constant=True, order=1, seasonal=True, additional_terms=[fourier], drop=True).in_sample()

    def __get_params(self, df):
        y = df.loc[:, 'sales']
        X_trend = self.dp_trend[self.dp_trend.index.isin(df.index)]

        #X_cycle = df.loc[:, 'sales']
        X_cycle = df.loc[:, 'lag']

        X_season = self.dp_season[self.dp_season.index.isin(df.index)]

        return X_trend, X_season, X_cycle, y

    def fit(self, df):
        X_trend, X_season, X_cycle, y = self.__get_params(df)

        # fit trend
        self.model_trend.fit(X_trend, y)
        y_trend = pd.DataFrame(self.model_trend.predict(X_trend), index=X_trend.index, columns=['sales'])

        # fit season
        y_resid = y - y_trend['sales']
        self.model_season.fit(X_season, y_resid)
        y_season = pd.DataFrame(self.model_season.predict(X_season), index=X_season.index, columns=['sales'])

        # fit cycle
        y_resid = y_resid - y_season['sales']
        self.model_cycle.fit(X_cycle, y_resid)
        y_cycle = pd.DataFrame(self.model_cycle.predict(X_cycle), index=X_cycle.index, columns=['sales'])
  
        # next one
        y_resid = y_resid - y_cycle['sales']

    def predict(self, df):
        X_trend, X_season, X_cycle, y = self.__get_params(df)

        y_trend = pd.DataFrame(
            self.model_trend.predict(X_trend), 
            index=X_trend.index, columns=['sales'],
        )

        y_season = pd.DataFrame(
            self.model_season.predict(X_season), 
            index=X_season.index, columns=['sales'],
        )
        y_season += y_trend

        y_cycle = pd.DataFrame(
            self.model_cycle.predict(X_cycle),
            index=X_cycle.index, columns=['sales'],
        )
        y_cycle += y_season

        return y_trend, y_season, y_cycle


# fit on full range for best test
model = ModelHybrid()
model.fit(df)
y_trend, y_season, y_cycle = model.predict(df)

# fit on train and valid range for prod mode
model2 = ModelHybrid()
model2.fit(df_train)
y_fit_trend, y_fit_season, y_fit_cycle = model2.predict(df_train)
y_pred_trend, y_pred_season, y_pred_cycle = model2.predict(df_valid)

# chart
fig, ax = plt.subplots(len(['sales']), 1, figsize=(10, len(['sales']) * 5), sharex=True)
plt.axvline(x=datetime.strptime(sep_date, "%Y-%m-%d"), color='red', linestyle='--')

y_fit_trend.plot(subplots=True, sharex=True, color='pink', ax=ax, linestyle='-')
y_fit_season.loc(axis=1)[['sales']].plot(subplots=True, sharex=True, color='pink', ax=ax, linestyle='dotted')
y_fit_cycle.loc(axis=1)[['sales']].plot(subplots=True, sharex=True, color='pink', ax=ax, linestyle='dashed')

y_pred_trend.plot(subplots=True, sharex=True, color='orange', ax=ax, linestyle='-')
y_pred_season.loc(axis=1)[['sales']].plot(subplots=True, sharex=True, color='orange', ax=ax, linestyle='dotted')
y_pred_cycle.loc(axis=1)[['sales']].plot(subplots=True, sharex=True, color='orange', ax=ax, linestyle='dashed')

df['sales'].plot(subplots=True, sharex=True, color='blue', ax=ax, linestyle='-')
y_trend.plot(subplots=True, sharex=True, color='blue', ax=ax, linestyle='-')
y_season.loc(axis=1)[['sales']].plot(subplots=True, sharex=True, color='blue', ax=ax, linestyle='dotted')
y_cycle.loc(axis=1)[['sales']].plot(subplots=True, sharex=True, color='blue', ax=ax, linestyle='dashed')

ax.legend([])
plt.show()