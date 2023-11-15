import keras_cv
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# https://pytorch.org/vision/main/generated/torchvision.transforms.RandomResizedCrop.html

BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 101

def package_inputs(image, label):
    return {"images": image, "labels": tf.one_hot(label, NUM_CLASSES)}

train_ds, eval_ds = tfds.load("caltech101", split=["train", "test"], as_supervised="true")
train_ds = train_ds.map(package_inputs, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(BATCH_SIZE * 16)
train_ds = train_ds.ragged_batch(BATCH_SIZE)

batch = next(iter(train_ds.take(1)))
original = batch["images"][1:10]

# Create a RandomCutout layer
model = keras_cv.layers.RandomFlip()

modified = model(original)

keras_cv.visualization.plot_image_gallery(original, rows=3, cols=3, value_range=(0, 255))
keras_cv.visualization.plot_image_gallery(modified, rows=3, cols=3, value_range=(0, 255))
plt.show()

