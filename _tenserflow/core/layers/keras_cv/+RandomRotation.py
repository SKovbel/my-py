import keras_cv
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# https://pytorch.org/vision/main/generated/torchvision.transforms.RandomResizedCrop.html

BATCH_SIZE = 9
NUM_CLASSES = 101

train_ds, eval_ds = tfds.load("caltech101", split=["train", "test"], as_supervised="true")
train_ds = train_ds.map(lambda img, lbl: {"images": img, "labels": tf.one_hot(lbl, NUM_CLASSES)}, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.ragged_batch(BATCH_SIZE)

batch = next(iter(train_ds.take(1)))
original = batch["images"]

# Create a RandomCutout layer
model = keras_cv.layers.RandomRotation(factor=0.10)
modified = model(original)
plt.subplot(121);plt.imshow(original)
plt.subplot(122);plt.imshow(modified)
plt.show()
