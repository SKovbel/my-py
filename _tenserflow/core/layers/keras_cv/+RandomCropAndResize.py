import keras_cv
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# https://pytorch.org/vision/main/generated/torchvision.transforms.RandomResizedCrop.html

BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 101

train_ds, eval_ds = tfds.load("caltech101", split=["train", "test"], as_supervised="true")
train_ds = train_ds.map(lambda image, label: {"images": image}, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(BATCH_SIZE * 16)
train_ds = train_ds.ragged_batch(BATCH_SIZE)

batch = next(iter(train_ds.take(1)))
image_batch = batch["images"]

crop_and_resize = keras_cv.layers.RandomCropAndResize(
    target_size=IMAGE_SIZE,
    crop_area_factor=(0.8, 1.0),
    aspect_ratio_factor=(0.9, 1.1)
)

crop_resize_batch = crop_and_resize(image_batch)

keras_cv.visualization.plot_image_gallery(image_batch, rows=3, cols=3, value_range=(0, 255))
keras_cv.visualization.plot_image_gallery(crop_resize_batch, rows=3, cols=3, value_range=(0, 255))
plt.show()