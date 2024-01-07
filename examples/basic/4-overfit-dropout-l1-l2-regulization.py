import os
import pathlib
import shutil
import tempfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots

import numpy as np

from IPython import display
from matplotlib import pyplot as plt

print(tf.__version__)

charts = True
debut = True

EPOCHS = 1000
FEATURES = 28
N_VALIDATION = int(1e3)
N_TRAIN = int(1e4)
BUFFER_SIZE = int(1e4)
BATCH_SIZE = 500
STEPS_PER_EPOCH = N_TRAIN//BATCH_SIZE

updir = os.path.join(os.path.dirname(__file__), '../../../tmp/higgs')
logdir = os.path.join(os.path.dirname(__file__), '../../../tmp/higgs/logdir')

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    0.001,
    decay_steps = 1000 * STEPS_PER_EPOCH,
    decay_rate = 1,
    staircase = False
)

def load_data():
    def pack_row(*row):
        label = row[0]
        features = tf.stack(row[1:],1)
        return features, label

    url = 'http://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz'
    gz = tf.keras.utils.get_file(fname="HIGGS.csv.gz", origin=url, extract=False, cache_dir='.', cache_subdir=updir)

    ds = tf.data.experimental.CsvDataset(gz, [float(),]*(FEATURES + 1), compression_type="GZIP")

    packed_ds = ds.batch(10000).map(pack_row).unbatch()

    for features,label in packed_ds.batch(1000).take(1):
        print(features[0])
        plt.hist(features.numpy().flatten(), bins = 101)
        plt.show()

    validate_ds = packed_ds.take(N_VALIDATION).cache()
    train_ds = packed_ds.skip(N_VALIDATION).take(N_TRAIN).cache()

    validate_ds = validate_ds.batch(BATCH_SIZE)
    train_ds = train_ds.shuffle(BUFFER_SIZE).repeat().batch(BATCH_SIZE)

    return train_ds, validate_ds


def get_callbacks(name):
    return [
        tfdocs.modeling.EpochDots(),
        tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=200),
        tf.keras.callbacks.TensorBoard(f"{logdir}/{name}"),
    ]

def compile_and_fit(name, model, trains, validates):
    model.compile(
        optimizer = tf.keras.optimizers.Adam(lr_schedule),
        loss = tf.keras.losses.BinaryCrossentropy(from_logits = True),
        metrics = [
            'accuracy',
            tf.keras.metrics.BinaryCrossentropy(
                from_logits = True,
                name = 'binary_crossentropy'
            )
        ]
    )

    model.summary()

    history = model.fit(
        trains,
        steps_per_epoch = STEPS_PER_EPOCH,
        epochs = EPOCHS,
        validation_data = validates,
        callbacks = get_callbacks(name),
        verbose=0
    )
    return history



train_ds, validate_ds = load_data()

if charts:
    step = np.linspace(0,100000)
    lr = lr_schedule(step)
    plt.figure(figsize = (8,6))
    plt.plot(step/STEPS_PER_EPOCH, lr)
    plt.ylim([0,max(plt.ylim())])
    plt.xlabel('Epoch')
    plt.title('Decay steps = ' + str(STEPS_PER_EPOCH*1000))
    _ = plt.ylabel('Learning Rate')
    plt.show()

tiny_model = tf.keras.Sequential([
    layers.Dense(16, activation='elu', input_shape=(FEATURES,)),
    layers.Dense(1)
])

small_model = tf.keras.Sequential([
    # `input_shape` is only required here so that `.summary` works.
    layers.Dense(16, activation='elu', input_shape=(FEATURES,)),
    layers.Dense(16, activation='elu'),
    layers.Dense(1)
])

medium_model = tf.keras.Sequential([
    layers.Dense(64, activation='elu', input_shape=(FEATURES,)),
    layers.Dense(64, activation='elu'),
    layers.Dense(64, activation='elu'),
    layers.Dense(1)
])

large_model = tf.keras.Sequential([
    layers.Dense(512, activation='elu', input_shape=(FEATURES,)),
    layers.Dense(512, activation='elu'),
    layers.Dense(512, activation='elu'),
    layers.Dense(512, activation='elu'),
    layers.Dense(1)
])

l1_model = tf.keras.Sequential([
    layers.Dense(512, activation='elu', kernel_regularizer=regularizers.l1(0.001), input_shape=(FEATURES,)),
    layers.Dense(512, activation='elu', kernel_regularizer=regularizers.l1(0.001)),
    layers.Dense(512, activation='elu', kernel_regularizer=regularizers.l1(0.001)),
    layers.Dense(512, activation='elu', kernel_regularizer=regularizers.l1(0.001)),
    layers.Dense(1)
])

l2_model = tf.keras.Sequential([
    layers.Dense(512, activation='elu', kernel_regularizer=regularizers.l2(0.001), input_shape=(FEATURES,)),
    layers.Dense(512, activation='elu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(512, activation='elu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(512, activation='elu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(1)
])

l1_l2_model = tf.keras.Sequential([
    layers.Dense(512, activation='elu', kernel_regularizer=regularizers.L1L2(0.001), input_shape=(FEATURES,)),
    layers.Dense(512, activation='elu', kernel_regularizer=regularizers.L1L2(0.001)),
    layers.Dense(512, activation='elu', kernel_regularizer=regularizers.L1L2(0.001)),
    layers.Dense(512, activation='elu', kernel_regularizer=regularizers.L1L2(0.001)),
    layers.Dense(1)
])

dropout_model = tf.keras.Sequential([
    layers.Dense(512, activation='elu', input_shape=(FEATURES,)),
    layers.Dropout(0.5),
    layers.Dense(512, activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(512, activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(512, activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(1)
])

l2_dropout_model = tf.keras.Sequential([
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001), activation='elu', input_shape=(FEATURES,)),
    layers.Dropout(0.5),
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001), activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001), activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001), activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(1)
])

l1_l2_dropout_model = tf.keras.Sequential([
    layers.Dense(512, kernel_regularizer=regularizers.L1L2(0.0001), activation='elu', input_shape=(FEATURES,)),
    layers.Dropout(0.5),
    layers.Dense(512, kernel_regularizer=regularizers.L1L2(0.0001), activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(512, kernel_regularizer=regularizers.L1L2(0.0001), activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(512, kernel_regularizer=regularizers.L1L2(0.0001), activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(1)
])

size_histories = {
    'Tiny': compile_and_fit("sizes/Tiny", tiny_model),
    'Small': compile_and_fit("sizes/Small", small_model),
    'Medium': compile_and_fit("sizes/Medium", medium_model),
    'large': compile_and_fit("sizes/large", large_model),
    'l1': compile_and_fit("regularizers/l1", l1_model),
    'l2': compile_and_fit("regularizers/l2", l2_model),
    'l1l2': compile_and_fit("regularizers/l1l2", l1_l2_model),
    'dropout': compile_and_fit("regularizers/dropout", dropout_model),
    'l2_dropout': compile_and_fit(l2_dropout_model, "regularizers/l2_dropout"),
    'l1l2_dropout': compile_and_fit(l1l2dropout_model, "regularizers/l1l2_dropout"),
}

plotter = tfdocs.plots.HistoryPlotter(metric = 'binary_crossentropy', smoothing_std=10)
plotter.plot(size_histories)
plt.ylim([0.5, 0.7])
plt.show()
