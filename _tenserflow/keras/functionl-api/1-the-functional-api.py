# https://keras.io/guides/functional_api/#extend-the-api-using-custom-layers

import os
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from PIL import Image

tmp_dir = os.path.join(os.path.dirname(__file__), '../../../tmp/keras-basic')
os.makedirs(tmp_dir, exist_ok=True)

# Directed Acyclic Graph (DAG) 

inputs = keras.Input(shape=(784,))
# img_inputs = keras.Input(shape=(32, 32, 3))

dense = layers.Dense(64, activation="relu")
x = dense(inputs)
x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(10)(x)

print(inputs.shape, inputs.dtype)

model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
model.summary()

image_path = os.path.join(tmp_dir, "graph-1.png")
keras.utils.plot_model(
    model,
    image_path,
    show_shapes=True,
    show_dtype=True,
    expand_nested=True,
    show_layer_activations=True,
    show_trainable=True
)

image = Image.open(image_path)
image.show()



(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.RMSprop(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
history = model.fit(x_train, y_train, batch_size=64, epochs=2, validation_split=0.2)
test_scores = model.evaluate(x_test, y_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])

model.save("path_to_my_model.keras")
del model
# Recreate the exact same model purely from the file:
model = keras.models.load_model("path_to_my_model.keras")
