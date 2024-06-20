import tensorflow.keras as keras

# Define two input branches
input1 = keras.layers.Input(shape=(10,))
input2 = keras.layers.Input(shape=(5,))

# Some sample layers for each branch
dense1 = keras.layers.Dense(64, activation='relu')(input1)
dense2 = keras.layers.Dense(32, activation='relu')(input2)

# Concatenate the outputs from both branches
concatenated = keras.layers.Concatenate()([dense1, dense2])

print(dense1.shape)
print(dense2.shape)
print(concatenated.shape)