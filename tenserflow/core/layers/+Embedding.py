import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.TextVectorization(
        max_tokens=5000, 
        output_mode='int', 
        output_sequence_length=10
    ),
    keras.layers.Embedding(
        input_dim=5000,   # vocabulary size 
        output_dim=4,     #  Length of output sequences
        input_length=10   #  Length of input sequences,
    ),
    keras.layers.GlobalAveragePooling1D()
])

x = tf.constant([["Good morning"], ["Good morning 2"], ["Good morning 21"], 
                 ["morning Good morning Good"], ["How are you doing"], ["I am well, thank you"], ["NO"]])
x = tf.strings.lower(x)

#split = tf.strings.split(lower)
#flat = tf.reshape(split, [-1]) #['good', 'morning', 'how', 'are', 'you', 'doing' ...]
#dict = tf.data.Dataset.from_tensor_slices(flat)


model.layers[0].adapt(x)
y = model.predict(x)
print(y)


exit(0)
################## 2
vectorizer = keras.layers.TextVectorization(
        max_tokens=5000, 
        output_mode='int', 
        output_sequence_length=10)

embdding = keras.layers.Embedding(
        input_dim=5000, # vocabulary size 
        output_dim=3, #  Length of output
        input_length=10) #  Length of input sequences,

vectorizer.adapt(dict.batch(64))

V = vectorizer(x)
print('vectorizer', V)

E = embdding(V)
print('embdding', E)

# RMSLE: 0.024302245859257388
# RMSLE: 0.024352714684550258
# RMSLE: 0.024387786034942002
# RMSLE: 0.02440371380066847
