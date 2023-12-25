from keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np

# Example time series data
data = np.array([[i] for i in range(100)])
targets = np.array([[i] for i in range(1, 101)])

# Define the TimeseriesGenerator
sequence_length = 10
generator = TimeseriesGenerator(data, targets, length=sequence_length, batch_size=1)

# Access the generated batches
for i in range(len(generator)):
    x, y = generator[i]
    print(f"Batch {i + 1} - Input: {x}, Target: {y}")
