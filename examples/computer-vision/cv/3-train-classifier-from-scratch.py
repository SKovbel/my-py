# https://keras.io/guides/keras_cv/classification_with_keras_cv/
import json
import math
import keras_cv
import tensorflow as tf
import tensorflow_datasets as tfds
import keras
from keras import losses
import numpy as np
from keras import optimizers
from tensorflow.keras.optimizers import schedules
from keras import metrics

BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
AUTOTUNE = tf.data.AUTOTUNE
NUM_CLASSES = 101
EPOCHS = 1


def package_inputs(image, label):
    return {"images": image, "labels": tf.one_hot(label, NUM_CLASSES)}

train_ds, eval_ds = tfds.load(
    "caltech101", split=["train", "test"], as_supervised="true"
)
train_ds = train_ds.map(package_inputs, num_parallel_calls=tf.data.AUTOTUNE)
eval_ds = eval_ds.map(package_inputs, num_parallel_calls=tf.data.AUTOTUNE)

train_ds = train_ds.shuffle(BATCH_SIZE * 16)
train_ds = train_ds.ragged_batch(BATCH_SIZE)
eval_ds = eval_ds.ragged_batch(BATCH_SIZE)

batch = next(iter(train_ds.take(1)))
image_batch = batch["images"]
label_batch = batch["labels"]

model = keras.Sequential([
    keras_cv.layers.RandomFlip(),
    keras_cv.layers.RandomCropAndResize(
        target_size=IMAGE_SIZE,
        crop_area_factor=(0.8, 1.0),
        aspect_ratio_factor=(0.9, 1.1),
    ),
    # AutoContrast, Equalize, Solarize, RandomColorJitter, RandomContrast, RandomBrightness, ShearX, ShearY, TranslateXта TranslateY
    keras_cv.layers.RandAugment(
        augmentations_per_image=3,
        magnitude=0.3,
        value_range=(0, 255),
    ),        
    keras_cv.layers.RandomChoice(
        [
            # keras_cv.layers.RandomCutout(width_factor=0.4, height_factor=0.4),
            keras_cv.layers.CutMix(),
            keras_cv.layers.MixUp(),
        ],
        batchwise=True
    )
])

inference_resizing = keras_cv.layers.Resizing(
    IMAGE_SIZE[0], IMAGE_SIZE[1], crop_to_aspect_ratio=True
)
eval_ds = eval_ds.map(inference_resizing, num_parallel_calls=tf.data.AUTOTUNE)
inference_resizing = keras_cv.layers.Resizing(
    IMAGE_SIZE[0], IMAGE_SIZE[1], crop_to_aspect_ratio=True
)
eval_ds = eval_ds.map(inference_resizing, num_parallel_calls=tf.data.AUTOTUNE)



# unpack dataset
def unpackage_dict(inputs):
    return inputs["images"], inputs["labels"]


train_ds = train_ds.map(unpackage_dict, num_parallel_calls=tf.data.AUTOTUNE)
eval_ds = eval_ds.map(unpackage_dict, num_parallel_calls=tf.data.AUTOTUNE)

#Optimizer Tuning
def lr_warmup_cosine_decay(
    global_step,
    warmup_steps,
    hold=0,
    total_steps=0,
    start_lr=0.0,
    target_lr=1e-2,
):
    # Cosine decay
    learning_rate = (
        0.5
        * target_lr
        * (
            1
            + tf.cos(
                tf.constant(math.pi)
                * tf.cast(global_step - warmup_steps - hold, tf.float32)
                / float(total_steps - warmup_steps - hold)
            )
        )
    )

    warmup_lr = tf.cast(target_lr * (global_step / warmup_steps), tf.float32)
    target_lr = tf.cast(target_lr, tf.float32)

    if hold > 0:
        learning_rate = tf.where(
            global_step > warmup_steps + hold, learning_rate, target_lr
        )

    learning_rate = tf.where(global_step < warmup_steps, warmup_lr, learning_rate)
    return learning_rate


class WarmUpCosineDecay(schedules.LearningRateSchedule):
    def __init__(self, warmup_steps, total_steps, hold, start_lr=0.0, target_lr=1e-2):
        super().__init__()
        self.start_lr = start_lr
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.hold = hold

    def __call__(self, step):
        lr = lr_warmup_cosine_decay(
            global_step=step,
            total_steps=self.total_steps,
            warmup_steps=self.warmup_steps,
            start_lr=self.start_lr,
            target_lr=self.target_lr,
            hold=self.hold,
        )

        return tf.where(step > self.total_steps, 0.0, lr, name="learning_rate")


total_images = 9000
total_steps = (total_images // BATCH_SIZE) * EPOCHS
warmup_steps = int(0.1 * total_steps)
hold_steps = int(0.45 * total_steps)
schedule = WarmUpCosineDecay(
    start_lr=0.05,
    target_lr=1e-2,
    warmup_steps=warmup_steps,
    total_steps=total_steps,
    hold=hold_steps,
)

initial_learning_rate = 0.01
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)

model = keras.Sequential(
    [
        keras_cv.models.EfficientNetV2B1Backbone(),
        keras.layers.GlobalMaxPooling2D(),
        keras.layers.Dropout(rate=0.5),
        keras.layers.Dense(101, activation="softmax"),
    ]
)

loss = losses.CategoricalCrossentropy(label_smoothing=0.1)
model.compile(
    loss=loss,
    optimizer=optimizer,
    metrics=[
        metrics.CategoricalAccuracy(),
        metrics.TopKCategoricalAccuracy(k=5),
    ],
)

model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=eval_ds,
)
