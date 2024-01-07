
d = {'a': 5, 'b': 12, 'c': 8, 'd': 5, 'e': 12}

unique_values_count = len(set(d.values()))
print(f'Count of unique values: {unique_values_count}')



high_cardinality_numcols = [k for k,v in d.items() if v > 10]
print(high_cardinality_numcols)