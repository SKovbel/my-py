import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import matplotlib.pyplot as plt
import re
import numpy as np
import shutil
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses

from dexa.chart import train

debug = True
updir=os.path.join(os.path.dirname(__file__),  "../../../tmp/stack_overflow_16k")

batch_size = 32
seed = 42
max_features = 10000
sequence_len = 1000
embedding_dim = 16
epochs = 100

def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')


def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label


def load_data(vectorize_layer):
    url = "https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz"
    dataset = tf.keras.utils.get_file(fname="stack_overflow_16k.tar.gz", origin=url, extract=True, cache_dir='.', cache_subdir=updir)
    dataset_dir = os.path.join(os.path.dirname(dataset))

    test_dir = os.path.join(dataset_dir, 'test')
    train_dir = os.path.join(dataset_dir, 'train')

    raw_val_ds = tf.keras.utils.text_dataset_from_directory(train_dir, batch_size=batch_size, validation_split=0.2, subset='validation', seed=seed)
    raw_test_ds = tf.keras.utils.text_dataset_from_directory(test_dir, batch_size=batch_size)
    raw_train_ds = tf.keras.utils.text_dataset_from_directory(train_dir, batch_size=batch_size, validation_split=0.2, subset='training', seed=seed)

    vectorize_layer.adapt(raw_train_ds.map(lambda x, y: x))

    val_ds = raw_val_ds.cache()
    test_ds = raw_test_ds.cache()
    train_ds = raw_train_ds.cache()

    val_ds = raw_val_ds.map(vectorize_text)
    test_ds = raw_test_ds.map(vectorize_text)
    train_ds = raw_train_ds.map(vectorize_text)

    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)


    if debug:
        for text_batch, label_batch in raw_train_ds.take(1):
            for i in range(3):
                #print("Review", text_batch.numpy()[i])
                print("Label", label_batch.numpy()[i])

        for i in range(4):
            print(f"Label {i} corresponds to {raw_train_ds.class_names[i]}")

    return raw_val_ds, raw_test_ds, raw_train_ds, val_ds, test_ds, train_ds, val_ds, test_ds, train_ds, val_ds, test_ds, train_ds


vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_len
)

raw_val_ds, raw_test_ds, raw_train_ds, val_ds, test_ds, train_ds, val_ds, test_ds, train_ds, val_ds, test_ds, train_ds = load_data(vectorize_layer)

model = tf.keras.Sequential([
    vectorize_layer,
    layers.Embedding(max_features + 1, embedding_dim),
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),
    layers.Dense(4)
])

if debug:
    model.summary()

model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics='accuracy'
)

fit_result = model.fit(
    raw_train_ds,
    validation_data=raw_val_ds,
    epochs=epochs,
    verbose=2 if debug else 1
)

loss, accuracy = model.evaluate(raw_test_ds)

if debug:
    print("Loss: ", loss)
    print("Accuracy: ", accuracy)

fit_chart = train.FitChart()
fit_chart.chart(model, fit_result)

examples = [
  "import Tenserflow as tf",
  "document.window",
  "System.out.println"
]

result = model.predict(examples)

labels = [raw_train_ds.class_names[i] for i in range(4)]
print(f"Labebls: ", labels)
for i in range(len(result)):
    max_index = np.argmax(result[i])
    print(result[i])
    print(f"{i}# {labels[max_index]} - {examples[i]}")
