# https://towardsdev.com/five-killer-optimization-techniques-that-most-pandas-arent-aware-of-f1e31af2257a
import os
import time
import pandas as pd
import datatable as dt
import numpy as np
import random

DIR = os.path.dirname(os.path.abspath(__file__))
path = lambda name: os.path.join(DIR, '..', 'data', name)
tmp = lambda name: os.path.join(DIR, '..', 'tmp', name)

# CASE 1
df = pd.read_csv(path("iris.csv"))    

# 18s
df.to_csv(tmp("iris.csv")) 
csv_data = pd.read_csv(tmp("iris.csv"))

# 9s
df.to_pickle(tmp("iris.pickle"))
pickle_data = pd.read_pickle(tmp("iris.pickle"))

# 8s
df.to_parquet(tmp("iris.parquet"))
parquet_data = pd.read_parquet(tmp("iris.parquet"))

# 6s
df.to_feather(tmp("iris.parquet"))
feather_data = pd.read_feather(tmp("iris.parquet"))

# case 2
# open datatable and covert to panda,  1.8s vs 5.4 (read_csv directly)
dt_df = dt.fread(path("iris.csv"))
pd_df = dt_df.to_pandas()

# panda to datatable and to file, 19s vs 87 (to_csv)
dt_data = dt.Frame(df)
dt_data.to_csv(tmp("csv.csv"))


# CASE2 filtering
city_list = ["New York", "Manchester", "California", "Munich", "Bombay", 
             "Sydeny", "London", "Moscow", "Dubai", "Tokyo"]

job_list = ["Software Development Engineer", "Research Engineer", 
            "Test Engineer", "Software Development Engineer-II", 
            "Python Developer", "Back End Developer", 
            "Front End Developer", "Data Scientist", 
            "IOS Developer", "Android Developer"]

cmp_list = ["Amazon", "Google", "Infosys", "Mastercard", "Microsoft", 
            "Uber", "IBM", "Apple", "Wipro", "Cognizant"]

def init_data():
    data = []
    for i in range(4_096_000):
        company = random.choice(cmp_list)
        job = random.choice(job_list)
        city = random.choice(city_list)
        salary = int(round(np.random.rand(), 3)*10**6)
        employment = random.choices(["Full Time", "Intern"], weights=(80, 20))[0]
        rating = round((np.random.rand()*5), 1)
        
        data.append([company, job, city, salary, employment, rating])
        
    return pd.DataFrame(data, columns=["Company Name", "Employee Job Title",
                                    "Employee Work Location",  "Employee Salary", 
                                    "Employment Status", "Employee Rating"])
    
data = init_data()

t1 = time.time()  # Record the start time
df = data[data["Company Name"] == "Amazon"]
print('B1', time.time()-t1, 'seconds')

# or x2 times faster
t1 = time.time()  # Record the start time
data_grp = data.groupby("Company Name")
df = data_grp.get_group("Amazon")
print('B2', time.time()-t1, 'seconds')

# or x6 times faster
t1 = time.time()  # Record the start time
df2 = data_grp.get_group("Amazon")
print('B3', time.time()-t1, 'seconds')


# CASE 3 Merging DataFrames
t1 = time.time()  # Record the start time
df1 = pd.DataFrame([["A", 1], ["B", 2]], columns = ["col_a", "col_b"])
df2 = pd.DataFrame([["A", 3], ["B", 4]], columns = ["col_a", "col_c"])
pd.merge(df1, df2, on = "col_a", how = "inner")
print('C1', time.time()-t1, 'seconds')

t1 = time.time()  # Record the start time
df1.set_index("col_a", inplace=True)
df2.set_index("col_a", inplace=True)
df1.join(df2)
print('C2', time.time()-t1, 'seconds')


# CASE 4 Value_counts() vs GroupBy()
t1 = time.time()  # 
data["Company Name"].value_counts()
print('D1', time.time()-t1, 'seconds')

t1 = time.time()  # x2
data.groupby("Company Name").size()
print('D2', time.time()-t1, 'seconds')

t1 = time.time()  # x2
data["Company Name"].value_counts(normalize=True)
print('D3', time.time()-t1, 'seconds')

t1 = time.time()  # 
a = data.groupby("Company Name").size()
b = a/a.sum()
print('D4', time.time()-t1, 'seconds')


# CASE 5 Iterating over a DataFrame

def apply_loop(df):
    salary_sum = 0
    
    for i in range(len(df)):
        salary_sum += df.iloc[i]['Employee Salary']

    return salary_sum/df.shape[0]

def salary_iterrows(df):
    salary_sum = 0
    
    for index, row in df.iterrows():
        salary_sum += row['Employee Salary']
        
    return salary_sum/df.shape[0]

t1 = time.time()  # 
apply_loop(data)
print('E1', time.time()-t1, 'seconds')


t1 = time.time()  # 
salary_iterrows(data)
print('E2', time.time()-t1, 'seconds')
