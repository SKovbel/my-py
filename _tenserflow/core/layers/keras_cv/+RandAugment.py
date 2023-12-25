import keras_cv
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# https://pytorch.org/vision/main/generated/torchvision.transforms.RandomResizedCrop.html

BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
IMAGE_SIZE = (224, 224)

def to_dict(image, label):
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    label = tf.one_hot(label, num_classes)
    return {"images": image, "labels": label}

def visualize_dataset(dataset1, dataset2):
    fig, axs = plt.subplots(2, 7)
    for i, samples in enumerate(iter(dataset1.take(7))):
        images = samples["images"]
        axs[0, i].imshow(images[0].numpy().astype("uint8"))
    for i, samples in enumerate(iter(dataset2.take(7))):
        images = samples["images"]
        axs[1, i].imshow(images[0].numpy().astype("uint8"))
    plt.show()

def apply_rand_augment(inputs):
    inputs["images"] = model(inputs["images"])
    return inputs

model = keras_cv.layers.RandAugment(
    value_range=(0, 255),
    augmentations_per_image=3,
    magnitude=0.3,
    magnitude_stddev=0.2,
    rate=0.5,
)

data, dataset_info = tfds.load("caltech101", with_info=True, as_supervised=True)
num_classes = dataset_info.features["label"].num_classes
dataset = data["train"]
original = dataset.map(to_dict, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)
modified = original.map(apply_rand_augment, num_parallel_calls=AUTOTUNE)

visualize_dataset(original, modified)
