import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'google-keys.json')
from google.cloud import bigquery
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import DeterministicProcess
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
        bigquery.ScalarQueryParameter("from_date", "DATE", "2022-01-01"),
        bigquery.ScalarQueryParameter("to_date", "DATE", "2022-12-31")
    ]
))

sep_date = "2022-07-01"
sep_date_2 = (datetime.strptime(sep_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

df = query.to_dataframe()
df['date'] = pd.to_datetime(df['date'])
if 1==1:
    df = df.resample('W-Mon', on='date').mean()
else:
    df['date'] = df.date.dt.to_period('D')
    df = df.set_index(['date']).sort_index()
df_train, df_valid = df[:sep_date], df[sep_date_2:]


class ModelHybrid:
    def __init__(self):
        self.y_columns = ['sales']
        self.model_trend = LinearRegression()
        self.model_cycle = XGBRegressor()
        # dp from 2010
        pd_date_range = pd.date_range(start=datetime(2010, 1, 1), end=datetime.now(), freq='D')
        self.dp_trend = DeterministicProcess(index=pd_date_range, order=1).in_sample()

    def __get_params(self, df):
        y = df.loc[:, 'sales']
        X_trend = self.dp_trend[self.dp_trend.index.isin(df.index)]
        X_cycle = df.loc[:, 'sales']
        return X_trend, X_cycle, y

    def fit(self, df):
        X_trend, X_cycle, y = self.__get_params(df)

        # fit trend
        self.model_trend.fit(X_trend, y)
        y_trend = pd.DataFrame(self.model_trend.predict(X_trend), index=X_trend.index, columns=self.y_columns)
        y_resid_trend = y - y_trend['sales']

        # fit cycle
        self.model_cycle.fit(X_cycle, y_resid_trend)
        y_fit_cycle = pd.DataFrame(self.model_cycle.predict(X_cycle), index=X_cycle.index, columns=self.y_columns)

    def predict(self, df):
        X_trend, X_cycle, y = self.__get_params(df)
        print(X_trend)
        y_trend = pd.DataFrame(
            self.model_trend.predict(X_trend), 
            index=X_trend.index, columns=self.y_columns,
        )

        y_cycle = y_trend.stack().squeeze()  # wide to long
        y_cycle += self.model_cycle.predict(X_cycle)
        return y_trend, y_cycle.unstack()




# fit on full range for best test
model = ModelHybrid()
model.fit(df)
y_1, y_2 = model.predict(df)
y_2 = y_2.clip(0.0)

# fit on train and valid range for prod mode
model2 = ModelHybrid()
model2.fit(df_train)
y_fit_trend, y_fit_cycle = model2.predict(df_train)
y_fit_cycle = y_fit_cycle.clip(0.0)
y_pred_trend, y_pred_cycle = model2.predict(df_valid)
y_pred_cycle = y_pred_cycle.clip(0.0)

# chart
fig, ax = plt.subplots(len(model2.y_columns), 1, figsize=(10, len(model2.y_columns) * 5), sharex=True)
plt.axvline(x=datetime.strptime(sep_date, "%Y-%m-%d"), color='red', linestyle='--')
y_fit_trend.plot(subplots=True, sharex=True, color='orange', ax=ax)
y_fit_cycle.loc(axis=1)[model2.y_columns].plot(subplots=True, sharex=True, color='orange', ax=ax)
y_pred_trend.plot(subplots=True, sharex=True, color='orange', ax=ax)
y_pred_cycle.loc(axis=1)[model2.y_columns].plot(subplots=True, sharex=True, color='orange', ax=ax)
y_1.plot(subplots=True, sharex=True, color='blue', ax=ax, linestyle=':')
y_2.loc(axis=1)[model2.y_columns].plot(subplots=True, sharex=True, color='blue', ax=ax, linestyle=':')
ax.legend([])
plt.show()
