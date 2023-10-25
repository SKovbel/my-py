import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow_hub as hub
import tensorflow_datasets as tfds

from dexa.chart import train

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE")

dirup=os.path.join(os.path.dirname(__file__),  "../var/")
embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
dataset_name = "imdb_reviews"

train_data, validation_data, test_data = tfds.load(
    name=dataset_name,
    data_dir=dirup,
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True
)
train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))

hub_layer = hub.KerasLayer(embedding, input_shape=[],  dtype=tf.string, trainable=True)
hub_layer(train_examples_batch[:3])

model = tf.keras.Sequential(layers=[
    hub_layer,
    layers.Dense(16, activation='relu'),
    layers.Dense(1)
])

model.summary()

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
)

fit_result = model.fit(
    train_data.shuffle(10000).batch(512),
    epochs=5,
    validation_data=validation_data.batch(512),
    verbose=1
)

loss, accuracy = model.evaluate(test_data.batch(512), verbose=2)

fit_chart = train.FitChart()
fit_chart.chart(model, fit_result)
fit_chart.print(model, fit_result)

examples = [
  "The movie was great!",
  "The movie was okay.",
  "The movie was terrible..."
]

result = model.predict(examples)
for i in range(len(result)):
    print(f"{i}# {examples[i]}", result[i])

