import pandas as pd

fruits = pd.DataFrame({
    'Apples':[30],
    'Bananas':[21]})

fruits = pd.DataFrame(
    [[30, 21]],
    columns=['Apples', 'Bananas'])

fruit_sales = pd.DataFrame(
    [[35, 21], [41,34]],
    columns=['Apples', 'Bananas'],
    index=['2017 Sales', '2018 Sales'])


df = pd.DataFrame({
    'A': [1,2,3],
    'B': [11,12,13]
})

def process1(v):
    print(v)

print('Case A:')
df.apply(process1)
print('Case A1:')
df.apply(process1, axis='rows')
print('Case B:')
df.apply(process1, axis=0)
print('Case C:')
df.apply(process1, axis='columns')
print('Case D:')
df.apply(process1, axis=1)
