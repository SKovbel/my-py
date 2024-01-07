import pandas as pd

people_dict = {
    'birthyear': pd.Series([1980, 1992, 2003], index=['alice', 'bob', 'charles']),
    'weight': pd.Series([68, 83, 112], index=['alice', 'bob', 'charles']),
    'children': pd.Series([0, 3], index=['charles', 'bob']),
    'hobby': pd.Series(['biking', 'dancing'], index=['alice', 'bob']),
}

people = pd.DataFrame(people_dict)
print('A:\n', people)
print('B:\n', people['birthyear'] > 1991)
print('C:\n', people[people['birthyear'] > 1991])
print('head:\n', people.head())
print('tail:\n', people.tail())
print('info:\n', people.info())
print('describe:\n', people.describe())