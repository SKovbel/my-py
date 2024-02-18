import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'google-keys.json')
from google.cloud import bigquery
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.deterministic import DeterministicProcess
from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from datetime import datetime, timedelta

sep_date = "2022-07-01"

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

class BoostedHybrid:
    def __init__(self):
        self.y_columns = ['sales']
        self.model_1 = LinearRegression()
        self.model_2 = XGBRegressor()

    def fit(self, X_1, X_2, y):
        self.model_1.fit(X_1, y)
        y_fit_1 = pd.DataFrame(self.model_1.predict(X_1), index=X_1.index, columns=self.y_columns)
        y_resid_1 = y - y_fit_1['sales']

        self.model_2.fit(X_2, y_resid_1)
        self.y_fit_1 = y_fit_1
        self.y_resid_1 = y_resid_1

    def predict(self, X_1, X_2):
        y_pred_1 = pd.DataFrame(
            self.model_1.predict(X_1), 
            index=X_1.index, columns=self.y_columns,
        )
        y_pred_2 = y_pred_1.stack().squeeze()  # wide to long
        y_pred_2 += self.model_2.predict(X_2)
        return y_pred_1, y_pred_2.unstack()


df = query.to_dataframe()
df['date'] = pd.to_datetime(df['date'])
if 1==1:
    df = df.resample('W-Mon', on='date').mean()
else:
    df['date'] = df.date.dt.to_period('D')
    df = df.set_index(['date']).sort_index()

y = df.loc[:, 'sales']

dp = DeterministicProcess(index=y.index, order=1)
X_1 = dp.in_sample()
X_2 = df.loc[:, 'sales']

sep_date_2 = (datetime.strptime(sep_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
y_train, y_valid = y[:sep_date], y[sep_date_2:]
X1_train, X1_valid = X_1[: sep_date], X_1[sep_date_2 :]
X2_train, X2_valid = X_2.loc[:sep_date], X_2.loc[sep_date_2:]

model = BoostedHybrid()
model.fit(X_1, X_2, y)
y_1, y_2 = model.predict(X_1, X_2)
y_2 = y_2.clip(0.0)

model2 = BoostedHybrid()
model2.fit(X1_train, X2_train, y_train)

y_fit_1, y_fit_2 = model2.predict(X1_train, X2_train)
y_fit_2 = y_fit_2.clip(0.0)

y_pred_1, y_pred_2 = model2.predict(X1_valid, X2_valid)
y_pred_2 = y_pred_2.clip(0.0)

fig, ax = plt.subplots(len(model2.y_columns), 1, figsize=(10, len(model2.y_columns) * 5), sharex=True)
plt.axvline(x=datetime.strptime(sep_date, "%Y-%m-%d"), color='red', linestyle='--')

y_fit_1.plot(subplots=True, sharex=True, color='orange', ax=ax)
y_fit_2.loc(axis=1)[model2.y_columns].plot(subplots=True, sharex=True, color='orange', ax=ax)

y_pred_1.plot(subplots=True, sharex=True, color='orange', ax=ax)
y_pred_2.loc(axis=1)[model2.y_columns].plot(subplots=True, sharex=True, color='orange', ax=ax)

y_1.plot(subplots=True, sharex=True, color='blue', ax=ax, linestyle=':')
y_2.loc(axis=1)[model2.y_columns].plot(subplots=True, sharex=True, color='blue', ax=ax, linestyle=':')

ax.legend([])

plt.show()
