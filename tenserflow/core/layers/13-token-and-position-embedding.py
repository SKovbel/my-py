import tensorflow as tf
from tensorflow import keras
import keras_nlp

model = keras.Sequential([
    keras.layers.TextVectorization(
        max_tokens=5000, 
        output_mode='int', 
        output_sequence_length=5
    ),
    #keras.layers.Embedding(
    #    input_dim=5000, # vocabulary size 
    #    output_dim=3, #  Length of output
    #    input_length=10 #  Length of input sequences,
    #),
    keras_nlp.layers.TokenAndPositionEmbedding(
        mask_zero=True,
        vocabulary_size=5000, # vocabulary size 
        sequence_length=10, #  Length of output
        embedding_dim=3, #  Length of output
    )
])

# prepare data
x =  tf.constant([["Good morning"], ["How are you doing"], ["I am well, thank you"]])
lower = tf.strings.lower(x)
split = tf.strings.split(lower)
flat = tf.reshape(split, [-1]) #['good', 'morning', 'how', 'are', 'you', 'doing' ...]
dict = tf.data.Dataset.from_tensor_slices(flat)

# text vectoriazation layer adaptation
model.layers[0].adapt(dict.batch(64))

y = model.predict(x)

# [[[-0.01567411 -0.02615447 -0.04952345]
#   [-0.03134163  0.00893166  0.04197231]
#   [-0.04370204 -0.01348071  0.01738245]
#   [-0.04370204 -0.01348071  0.01738245]
#   [-0.04370204 -0.01348071  0.01738245]]
# 
#  [[ 0.01053116 -0.0307356  -0.02847686]
#   [-0.03112371  0.00320128 -0.00609349]
#   [-0.04619465  0.04933066 -0.04071984]
#   [-0.01638418 -0.00893801 -0.01988443]
#   [-0.04370204 -0.01348071  0.01738245]]
# 
# [[-0.00860243 -0.02533059 -0.00257198]
#   [ 0.04490963  0.04273863  0.00586807]
#   [ 0.00084133  0.04648209 -0.02876893]
#   [ 0.01868332 -0.04133495  0.01682564]
#   [-0.04619465  0.04933066 -0.04071984]]]

print(y)
