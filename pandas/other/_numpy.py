import pandas as pd

# Assuming 'df' is your DataFrame
df = pd.DataFrame({
    'Column1': [1, 2, 3],
    'Column2': [4, 5, 6],
    'Column3': [7, 8, 9]
})

# Convert the entire DataFrame to a NumPy array of float values
numpy_array = df.values.astype(float)

# Display the NumPy array
print(numpy_array)
