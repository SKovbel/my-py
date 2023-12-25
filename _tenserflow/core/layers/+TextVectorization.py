import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.TextVectorization(max_tokens=5000, output_mode='int', output_sequence_length=5)
])

# prepare data
x =  tf.constant([["Good morning"], ["How are you doing"]])
lower = tf.strings.lower(x)
split = tf.strings.split(lower)
flat = tf.reshape(split, [-1])
dict = tf.data.Dataset.from_tensor_slices(flat)

model.layers[0].adapt(dict.batch(64))
y = model.predict(x)

# [
#   ['Good morning'],
#   ['How are you doing']
# ]
print(x)

#['good', 'morning', 'how', 'are', 'you', 'doing']
print(flat)

# [
#   [5 3 0 0 0]
#   [4 7 2 6 0]
# ]
print(y)
