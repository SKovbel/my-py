import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pixelsnail.loaders as loaders
from pixelsnail.layers import pixelSNAIL
from pixelsnail.losses import discretized_mix_logistic_loss, sample_from_discretized_mix_logistic
from pixelsnail.sampling import SampleCallback

N_WORKERS = 8
DIR = f'{os.path.dirname(os.path.abspath(__file__))}/../../tmp/gems'
os.makedirs(DIR, exist_ok=True)

rng = np.random.RandomState(10)
tf.compat.v1.set_random_seed(10)
tf.random.set_seed(10)

def preprocess(x):
    return (x-127.5) / 127.5
    
datagen_train, datagen_test = loaders.load_gemstone_data(preprocess, data_dir=DIR, batch_size=32)

plt.imshow((datagen_train.__next__()[0][0]*127.5 + 127.5).astype(int))

model = pixelSNAIL(
    attention=True,
    num_grb_per_pixel_block=2, 
    dropout=0.2,
    num_pixel_blocks=2,
    nr_filters=128)
model.predict(datagen_train.__next__()[0])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1, clipvalue=1)

model.compile(
    optimizer=optimizer,
    loss=discretized_mix_logistic_loss,
    metrics=[])

sample_callback = SampleCallback(save_every=10)

save_callback = tf.keras.callbacks.ModelCheckpoint(
    sample_callback.save_dir + os.sep + "test_gems_1.h5",
    monitor='val_loss',
    verbose=0,
    save_best_only=True,
    save_weights_only=True,
    mode='auto',
    save_freq='epoch')

tb_callback = tf.keras.callbacks.TensorBoard(write_grads=True, histogram_freq=1, log_dir=sample_callback.save_dir)

history = model.fit(
    datagen_train,
    epochs=500,
    steps_per_epoch=len(datagen_train),
    validation_steps=len(datagen_test),
    validation_data=datagen_test,
    callbacks=[save_callback, sample_callback, tb_callback],
    workers=N_WORKERS)

