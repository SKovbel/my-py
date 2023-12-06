import tensorflow as tf
from tensorflow import keras

# What is this layer doing? Notice that we no longer have the `Flatten` layer that usually comes after
# the base to transform the 2D feature data to 1D data needed by the classifier.
# Now the `GlobalAvgPool2D` layer is serving this function.
# But, instead of "unstacking" the feature (like `Flatten`), it simply replaces the entire feature map with its average value.
# Though very destructive, it often works quite well and has the advantage of reducing the number of parameters in the model.

model = keras.Sequential([
    keras.layers.GlobalMaxPooling2D()
])

# remove timeseries dimension (as example)
x = tf.constant([
    [1, 4, 7],
    [12, 14, 16],
    [20, 30, 40]
])

print(x)
x = tf.reshape(x, [1, 1, 3, 3])
print(x)
y = model.predict(x)
print(x)
print(y)
