import os
import pathlib

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_addons as tfa
import tensorflow_datasets as tfds

up_dir = os.path.join(os.path.dirname(__file__), '../../../tmp/cyclegan')

datasets = tfds.list_builders()
for dataset in datasets:
    if 'cycle' in dataset:
        print(dataset)

tfds.disable_progress_bar()

autotune = tf.data.AUTOTUNE
orig_img_size = (286, 286) # Define the standard image size.
input_img_size = (256, 256, 3) # Size of the random crops to be used during training.
kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02) # Weights initializer for the layers.
gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02) # Gamma initializer for instance normalization.
buffer_size = 256
batch_size = 1

def download_and_parsing():
    def do_filter(class_name_idx):
        def custom_filter(image, label):
            return label[0] == class_name_idx
        return custom_filter

    archive = keras.utils.get_file(
        origin='https://efrosgans.eecs.berkeley.edu/cyclegan/datasets/horse2zebra.zip',
        extract=True,
        cache_dir='.',
        cache_subdir=up_dir
    )
    data_dir = pathlib.Path(archive).with_suffix('')


    dataset = tf.keras.utils.image_dataset_from_directory(data_dir)
    class_names = dataset.class_names
    print(class_names)

    train_horses = dataset.filter(do_filter(class_names.index('trainA')))
    train_zebras = dataset.filter(do_filter(class_names.index('trainB')))
    test_horses = dataset.filter(do_filter(class_names.index('testA')))
    test_zebras = dataset.filter(do_filter(class_names.index('testB')))
    return train_horses, train_zebras, test_horses, test_zebras

def normalize_img(img):
    img = tf.cast(img, dtype=tf.float32)
    # Map values in the range [-1, 1]
    return (img / 127.5) - 1.0

def preprocess_train_image(img, label):
    img = tf.image.random_flip_left_right(img) # Random flip
    img = tf.image.resize(img, [*orig_img_size]) # Resize to the original size first
    img = tf.image.random_crop(img, size=[*input_img_size]) # Random crop to 256X256
    img = normalize_img(img) # Normalize the pixel values in the range [-1, 1]
    return img

def preprocess_test_image(img, label):
    img = tf.image.resize(img, [input_img_size[0], input_img_size[1]]) # Only resizing and normalization for the test images.
    img = normalize_img(img)
    return img

# Load the horse-zebra dataset using tensorflow-datasets.
# dataset, _ = tfds.load("cycle_gan/horse2zebra", with_info=True, as_supervised=True)
# train_horses, train_zebras = dataset["trainA"], dataset["trainB"]
# test_horses, test_zebras = dataset["testA"], dataset["testB"]
train_horses, train_zebras, test_horses, test_zebras = download_and_parsing()

# Create Dataset objects
train_horses = (
    train_horses.map(preprocess_train_image, num_parallel_calls=autotune)
    .cache()
    .shuffle(buffer_size)
    .batch(batch_size)
)
train_zebras = (
    train_zebras.map(preprocess_train_image, num_parallel_calls=autotune)
    .cache()
    .shuffle(buffer_size)
    .batch(batch_size)
)
test_horses = (
    test_horses.map(preprocess_test_image, num_parallel_calls=autotune)
    .cache()
    .shuffle(buffer_size)
    .batch(batch_size)
)
test_zebras = (
    test_zebras.map(preprocess_test_image, num_parallel_calls=autotune)
    .cache()
    .shuffle(buffer_size)
    .batch(batch_size)
)

_, ax = plt.subplots(4, 2, figsize=(10, 15))
for i, samples in enumerate(zip(train_horses.take(4), train_zebras.take(4))):
    horse = (((samples[0][0] * 127.5) + 127.5).numpy()).astype(np.uint8)
    zebra = (((samples[1][0] * 127.5) + 127.5).numpy()).astype(np.uint8)
    ax[i, 0].imshow(horse)
    ax[i, 1].imshow(zebra)
plt.show()
