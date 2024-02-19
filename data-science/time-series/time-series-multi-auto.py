
import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../big-query/google-keys.json')
import numpy as np
from google.cloud import bigquery
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from xgboost import XGBRegressor
from datetime import datetime, timedelta
from itertools import combinations, permutations

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
    def __init__(self, dirs):
        self.dirs = dirs

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
        X_cycle = df.loc[:, 'lag'] # lag, sales
        X_season = self.dp_season[self.dp_season.index.isin(df.index)]
        return X_trend, X_season, X_cycle, y

    def fit(self, df):
        X_trend, X_season, X_cycle, y = self.__get_params(df)

        y_resid = y
        for dir in self.dirs:
            if dir == 'T':
                self.model_trend.fit(X_trend, y_resid)
                y_trend = pd.DataFrame(self.model_trend.predict(X_trend), index=X_trend.index, columns=['sales'])
                y_resid = y_resid - y_trend['sales']
            if dir == 'S':
                self.model_season.fit(X_season, y_resid)
                y_season = pd.DataFrame(self.model_season.predict(X_season), index=X_season.index, columns=['sales'])
                y_resid = y_resid - y_season['sales']
            if dir == 'C':
                self.model_cycle.fit(X_cycle, y_resid)
                y_cycle = pd.DataFrame(self.model_cycle.predict(X_cycle), index=X_cycle.index, columns=['sales'])
                y_resid = y_resid - y_cycle['sales']

    def predict(self, df):
        X_trend, X_season, X_cycle, y = self.__get_params(df)

        result = {}
        y_resid = pd.Series(0, index=y.index)
        for dir in self.dirs:
            if dir == 'T':
                y_resid += pd.DataFrame(
                    self.model_trend.predict(X_trend), 
                    index=X_trend.index, columns=['sales'],
                )['sales']
            elif dir == 'S':
                y_resid += pd.DataFrame(
                    self.model_season.predict(X_season), 
                    index=X_season.index, columns=['sales'],
                )['sales']
            elif dir == 'C':
                y_resid += pd.DataFrame(
                    self.model_cycle.predict(X_cycle),
                    index=X_cycle.index, columns=['sales'],
                )['sales']

            result[dir] = y_resid.copy()
        return result


def get_direct_variants(arr):
    variants = []
    for r in range(1, len(arr) + 1):
        for subset in combinations(arr, r):
            variants.append(subset)
    return variants

def get_variants(arr):
    variants = []
    for r in range(1, len(arr) + 1):
        for permutation in permutations(arr, r):
            variants.append(permutation)
    return variants


variants = get_variants(['T', 'C', 'S'])
linestyle =  lambda t: {'T': 'dashdot', 'C':'dashed', 'S': 'dotted'}[t]

variants = [arr for arr in variants if len(arr) == 3]


fig, axs = plt.subplots(len(variants), figsize=(20, 40))
for i, dirs in enumerate(variants):
    print(i, dirs)
    # fit on full range for best test
    model = ModelHybrid(dirs)
    model.fit(df)

    model2 = ModelHybrid(dirs)
    model2.fit(df_train)

    y = model.predict(df)
    y_fit = model2.predict(df_train)
    y_pred = model2.predict(df_valid)

    # chart
    axs[i].set_title('-'.join(dirs))
    axs[i].axvline(x=datetime.strptime(sep_date, "%Y-%m-%d"), color='red', linestyle='--')
    df['sales'].plot(subplots=True, sharex=True, color='blue', ax=axs[i], linestyle='-')

    for code, data in y_fit.items():
        data.plot(subplots=True, sharex=True, color='pink', ax=axs[i], linestyle=linestyle(code))

    for code, data in y_pred.items():
        data.plot(subplots=True, sharex=True, color='orange', ax=axs[i], linestyle=linestyle(code))

    for code, data in y.items():
        data.plot(subplots=True, sharex=True, color='blue', ax=axs[i], linestyle=linestyle(code))

    axs[i].legend([])

plt.tight_layout()
plt.show()
