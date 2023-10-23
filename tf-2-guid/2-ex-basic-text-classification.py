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

debug = True
updir = 'var/stack_overflow_16k'
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

url = "https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz"
dataset = tf.keras.utils.get_file(fname="stack_overflow_16k.tar.gz", origin=url, extract=True, cache_dir='.', cache_subdir=updir)
dataset_dir = os.path.join(os.path.dirname(dataset))

test_dir = os.path.join(dataset_dir, 'test')
train_dir = os.path.join(dataset_dir, 'train')

raw_val_ds = tf.keras.utils.text_dataset_from_directory(train_dir, batch_size=batch_size, validation_split=0.2, subset='validation', seed=seed)
raw_test_ds = tf.keras.utils.text_dataset_from_directory(test_dir, batch_size=batch_size)
raw_train_ds = tf.keras.utils.text_dataset_from_directory(train_dir, batch_size=batch_size, validation_split=0.2, subset='training', seed=seed)

if debug:
    for text_batch, label_batch in raw_train_ds.take(1):
        for i in range(3):
            #print("Review", text_batch.numpy()[i])
            print("Label", label_batch.numpy()[i])

vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_len
)
vectorize_layer.adapt(raw_train_ds.map(lambda x, y: x))

model = tf.keras.Sequential([
    vectorize_layer,
    layers.Embedding(max_features + 1, embedding_dim),
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),
    layers.Dense(4)
])

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
    for i in range(4):
        print(f"Label {i} corresponds to {raw_train_ds.class_names[i]}")

if debug:
    model.summary()

model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics='accuracy'
)

history = model.fit(
    raw_train_ds,
    validation_data=raw_val_ds,
    epochs=epochs,
    verbose=2 if debug else 0
)
loss, accuracy = model.evaluate(raw_test_ds)

print("Loss: ", loss)
print("Accuracy: ", accuracy)

history_dict = history.history

# test
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(acc) + 1)

plt.subplot(1, 2, 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()

# Test it with `raw_test_ds`, which yields raw stringsgit status
loss, accuracy = model.evaluate(
    raw_test_ds,
    verbose=2 if debug else 0
)
print(accuracy)

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
