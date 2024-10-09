import pandas as pd

df = pd.DataFrame({
    'Brand': ['Toyota', 'Honda', 'Ford'],
    'Color': ['red black', 'blue black', 'red'],
    'Price': [20000, 21000, 25000],
    'Year': [2020, 2021, 2022]
})
df = df.join(df['Color'].str.lower().str.replace('/', ' ', regex=False).str.split(expand=True).stack().str.get_dummies().groupby(level=0).sum().add_prefix('Color_'))
print(df)




id	price
0	188533	21730.648438
1	188534	60707.929688
2	188535	76600.210938
3	188536	14632.178711
4	188537	28508.964844

0	188533	25172.949219
1	188534	92907.882812
2	188535	51233.062500
3	188536	35289.191406
4	188537	30019.988281