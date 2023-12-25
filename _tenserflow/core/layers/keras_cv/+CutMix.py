import keras_cv
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# https://pytorch.org/vision/main/generated/torchvision.transforms.RandomResizedCrop.html
# see UpMix

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
    dataset1 = dataset1["images"].numpy()[0:7]
    dataset2 = dataset2["images"].numpy()[0:7]
    for i in range(7):
        axs[0, i].imshow(dataset1[i].astype("uint8"))
    for i in range(7):
        axs[1, i].imshow(dataset2[i].astype("uint8"))
    plt.show()

def apply_rand_augment(inputs):
    return model(inputs, training=True)

model = keras_cv.layers.CutMix()

data, dataset_info = tfds.load("oxford_flowers102", with_info=True, as_supervised=True)
num_classes = dataset_info.features["label"].num_classes
dataset = data["train"]
original = dataset.map(to_dict, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)

original = next(iter(original.take(7)))
modified = model(original, training=True)

visualize_dataset(original, modified)
