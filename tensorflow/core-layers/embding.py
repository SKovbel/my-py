import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
import tensorflow as tf

model = Sequential()
embedding = Embedding(6, 2, embeddings_initializer="uniform")
model.add(embedding)
input_array = np.expand_dims(np.array((0,1,2,3,4,5)), axis=0)
output_array = model.predict(input_array)

print(input_array)
print(output_array)