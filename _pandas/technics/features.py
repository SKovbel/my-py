import pandas as pd

df = pd.DataFrame({
    'W': [1, 2, 3, 4, 5],
    'H': [5, 6, 7, 3, 2],
    'Code': ['CA', 'CB', 'AC', 'CC', 'CB'],
    'Status': ['proc-proc', 'proc-new', 'comp-comp', 'proc-proc', 'comp-done'],
    'Phone': ['(420)1234312', '(320)1234566', '(550)1234321', '(420)1234312', '(520)0234312'],
})

df["Ratio"] = df["W"] / df["H"]

# split column to several
df[["Status-A", "Status-B"]] = (df["Status"].str.split("-", expand=True))

# new column as sum another two
df["WH"] = df[["W", "H"]].sum(axis=1)

# union two columns
df["W-H"] = df["W"].astype(str) + "-" + df["H"].astype(str)

# group mean
df["WH_MEAN_CODE"] = df.groupby("Code")["WH"].transform("mean")
df["CODE_COUNT"] = df.groupby("Code")["Code"].transform("count")

# Merge the values into the validation set
df = df.merge(df[["Status-A", "Status-B"]].drop_duplicates(), on="Status-A", how="left")

# new df as dummies columns, multiply and then join to main df
df2 = pd.get_dummies(df["Code"], prefix="X")
print(df2)
df2 = df2.mul(df['WH'], axis=0)
df = df.join(df2)


df["GT_3"] = df[["W", "H"]].gt(0.0).sum(axis=1)
               

print(df)
