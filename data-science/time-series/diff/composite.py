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


client = bigquery.Client()
query = client.query(f"""
    SELECT DATE(date) AS date,
           city,
           COUNT(*) AS items,
           SUM(sale_dollars) AS sales
      FROM bigquery-public-data.iowa_liquor_sales.sales
     WHERE date >= @from_date AND date <= @to_date
     GROUP BY city, date
     ORDER BY city, date
""", job_config=bigquery.QueryJobConfig(
    query_parameters=[
        bigquery.ScalarQueryParameter("from_date", "DATE", "2022-01-01"),
        bigquery.ScalarQueryParameter("to_date", "DATE", "2022-12-31")
    ]
))

class BoostedHybrid:
    def __init__(self):
        self.y_columns = None
        self.model_1 = LinearRegression()
        self.model_2 = XGBRegressor()

    def fit(self, X_1, X_2, y):
        self.model_1.fit(X_1, y)
        y_fit = pd.DataFrame(self.model_1.predict(X_1), index=X_1.index, columns=y.columns)
        
        y_resid = y - y_fit
        y_resid = y_resid.stack().squeeze() # wide to long

        self.model_2.fit(X_2, y_resid)
        self.y_columns = y.columns
        self.y_fit = y_fit
        self.y_resid = y_resid


    def predict(self, X_1, X_2):
        y_pred = pd.DataFrame(
            self.model_1.predict(X_1), 
            index=X_1.index, columns=self.y_columns,
        )
        y_pred = y_pred.stack().squeeze()  # wide to long
        y_pred += self.model_2.predict(X_2)
        return y_pred.unstack()


df = query.to_dataframe()
df['date'] = pd.to_datetime(df['date'])
df['date'] = df.date.dt.to_period('D')
df = df.set_index(['date', 'city']).sort_index()

sales = (
    df.groupby(['date', 'city'])
    .mean()
    .unstack('city')
    .loc['2022']

)
sales.fillna(0, inplace=True)

y = sales.loc[:, 'sales']
print(type(sales))
exit(0)
dp = DeterministicProcess(index=y.index, order=1)
X_1 = dp.in_sample()

le = LabelEncoder()
X_2 = sales.drop('sales', axis=1).stack()
X_2 = X_2.reset_index('city')
X_2['city'] = le.fit_transform(X_2['city'])
X_2['day'] = X_2.index.day

# trend vs 
model = BoostedHybrid()
model.fit(X_1, X_2, y)
y_pred = model.predict(X_1, X_2)
y_pred = y_pred.clip(0.0)


y_train, y_valid = y[:"2022-07-01"], y["2022-07-02":]
X1_train, X1_valid = X_1[: "2022-07-01"], X_1["2022-07-02" :]
X2_train, X2_valid = X_2.loc[:"2022-07-01"], X_2.loc["2022-07-02":]

model.fit(X1_train, X2_train, y_train)
y_fit = model.predict(X1_train, X2_train).clip(0.0)
y_pred = model.predict(X1_valid, X2_valid).clip(0.0)

cities = y.columns[0:6]
axs = y.loc(axis=1)[cities].plot(
    subplots=True, sharex=True, figsize=(11, 9), alpha=0.5,
)
_ = y_fit.loc(axis=1)[cities].plot(subplots=True, sharex=True, color='C0', ax=axs)
_ = y_pred.loc(axis=1)[cities].plot(subplots=True, sharex=True, color='C3', ax=axs)
for ax, family in zip(axs, cities):
    ax.legend([])
    ax.set_ylabel(family)
plt.show()
