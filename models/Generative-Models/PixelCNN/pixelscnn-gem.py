import os
import time
import random as rn

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

# Defining random seeds
random_seed = 42
tf.random.set_seed(random_seed)
np.random.seed(random_seed)
rn.seed(random_seed)

train_dir = 'tmp/gems/train/'
test_dir = 'tmp/gems/test/'
save_dir = 'tmp/checkpoints/'
os.makedirs(save_dir, exist_ok=True)

height = 32
width = 32
n_channel = 3
batch_size = 80
q_levels = 128
train_buf = 500


def to_numpy(dataset):
    images = []
    labels = []
    for image, label in dataset:
        images.append(image.numpy())
        labels.append(label.numpy())
    x = np.concatenate(images, axis=0)
    y = np.concatenate(labels, axis=0)
    return x, y

def quantisize(images, q_levels):
    """Digitize image into q levels"""
    return (np.digitize(images, np.arange(q_levels) / q_levels) - 1).astype('float32')

train_dataset = image_dataset_from_directory(
    train_dir,
    image_size=(height, width),  # resize
    batch_size=batch_size,
    shuffle=True
)

test_dataset = image_dataset_from_directory(
    test_dir,
    image_size=(height, width),  # resize
    batch_size=batch_size,
    shuffle=True
)

x_train, y_train = to_numpy(train_dataset)
x_test, y_test = to_numpy(test_dataset)

# normilize to 0..1
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Quantisize the input data in q levels, reduce noice and precision of pixel
x_train_quantised_of = quantisize(x_train, q_levels)
x_test_quantised_of = quantisize(x_test, q_levels)

print(x_train.shape)

# Creating input stream
train_dataset = tf.data.Dataset.from_tensor_slices((x_train_quantised_of / (q_levels - 1), x_train_quantised_of.astype('int32')))
train_dataset = train_dataset.shuffle(buffer_size=train_buf)
train_dataset = train_dataset.batch(batch_size)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test_quantised_of / (q_levels - 1), x_test_quantised_of.astype('int32')))
test_dataset = test_dataset.batch(batch_size)


class MaskedConv2D(tf.keras.layers.Layer):
    def __init__(
        self,
        mask_type,
        name,
        filters,
        kernel_size,
        strides=1,
        n_channels=3,
        padding='same',
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        **kwargs
    ):
        super(MaskedConv2D, self).__init__(name=name, **kwargs)

        self.mask_type = mask_type
        self.filters = filters
        self.strides = strides
        self.padding = padding.upper()
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)

    def __build_mask(self):
        center = self.kernel_size // 2
        self.mask = np.ones(self.kernel.shape, dtype=np.float32)
        self.mask[center, center + 1:, :, :] = 0
        self.mask[center + 1:, :, :, :] = 0

        for i in range(self.n_channels):
            for j in range(self.n_channels):
                if (self.mask_type == 'A' and i >= j) or (self.mask_type == 'B' and i > j):
                    self.mask[center, center, i::self.n_channels, j::self.n_channels] = 0
        np.set_printoptions(threshold=np.inf)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(self.kernel_size, self.kernel_size, int(input_shape[-1]), self.filters),
            initializer=self.kernel_initializer,
            trainable=True)

        self.bias = self.add_weight(
            shape=(self.filters,),
            initializer=self.bias_initializer,
            trainable=True)

        self.__build_mask()

    def call(self, input):
        masked_kernel = tf.math.multiply(self.mask, self.kernel)
        x = tf.nn.conv2d(input, masked_kernel, strides=[1, self.strides, self.strides, 1], padding=self.padding)
        x = tf.nn.bias_add(x, self.bias)
        return x

class ResidualBlock(tf.keras.Model):
    def __init__(self, n_filters, name):
        super(ResidualBlock, self).__init__()
        self.conv2a = MaskedConv2D('B', name=f"{name}-A", filters=n_filters//2, kernel_size=1, strides=1)
        self.conv2b = MaskedConv2D('B', name=f"{name}-B", filters=n_filters//2, kernel_size=7, strides=1)
        self.conv2c = MaskedConv2D('B', name=f"{name}-C", filters=n_filters, kernel_size=1, strides=1)

    def call(self, input_tensor):
        x = tf.nn.relu(input_tensor)
        x = self.conv2a(x)
        x = tf.nn.relu(x)
        x = self.conv2b(x)
        x = tf.nn.relu(x)
        x = self.conv2c(x)
        x += input_tensor
        return x
    


class PixelCNNLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_rate):
        super(PixelCNNLearningRateSchedule, self).__init__()
        self.initial_learning_rate = tf.convert_to_tensor(initial_learning_rate, dtype=tf.float32)
        self.decay_rate = tf.convert_to_tensor(decay_rate, dtype=tf.float32)
    
    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        return self.initial_learning_rate * tf.math.pow(self.decay_rate, step)
    
class Model:
    def __init__(self):
        self.epoch = 0
        self.n_filters = 120
        self.learning_rate_schedule = PixelCNNLearningRateSchedule(initial_learning_rate=5e-3, decay_rate=0.9999)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_schedule)
        self.compute_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.__build_model()

    def __build_model(self):
        inputs = keras.layers.Input(shape=(height, width, n_channel))
        x = MaskedConv2D("A", name="A", filters=self.n_filters, kernel_size=7)(inputs)
        for i in range(15):
            x = keras.layers.Activation(activation='relu', name=f"relu-{i}")(x)
            x = ResidualBlock(n_filters=self.n_filters, name=f"rb-{i}")(x)
        x = keras.layers.Activation(activation='relu')(x)
        x = MaskedConv2D("B", name="B", filters=self.n_filters, kernel_size=1)(x) 
        x = keras.layers.Activation(activation='relu')(x)
        x = MaskedConv2D("B", name="C", filters=n_channel * q_levels, kernel_size=1)(x)
        self.pixelcnn = tf.keras.Model(inputs=inputs, outputs=x)

    @tf.function
    def __train_step(self, batch_x, batch_y):
        with tf.GradientTape() as ae_tape:
            logits = self.pixelcnn(batch_x, training=True)
            logits = tf.reshape(logits, [-1, height, width, q_levels, n_channel])
            logits = tf.transpose(logits, perm=[0, 1, 2, 4, 3])
            loss = self.compute_loss(tf.one_hot(batch_y, q_levels), logits)
        gradients = ae_tape.gradient(loss, self.pixelcnn.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        self.optimizer.apply_gradients(zip(gradients, self.pixelcnn.trainable_variables))

    def train(self, n_epochs=1):
        for epoch in range(self.epoch, n_epochs):
            start = time.time()
            for i_batch, (batch_x, batch_y) in enumerate(train_dataset):
                self.__train_step(batch_x, batch_y)
                tf.print(f'Epoch {epoch}, batch {i_batch}, shape: {batch_x.shape}, time: {time.time() - start:.2f}')
            self.pixelcnn.save_weights(os.path.join(save_dir, f'{epoch:2d}.weights.h5'))
        print('Training complete.')

    def generate(self, n_samples=9):
        samples = np.zeros((n_samples, height, width, n_channel), dtype='float32')
        for i in range(height):
            print(f'Height position {i}')
            for j in range(width):
                for k in range(n_channel):
                    logits = self.pixelcnn(samples)
                    logits = tf.reshape(logits, [-1, height, width, q_levels, n_channel])
                    logits = tf.transpose(logits, perm=[0, 1, 2, 4, 3])
                    next_sample = tf.random.categorical(logits[:, i, j, k, :], 1)
                    samples[:, i, j, k] = (next_sample.numpy() / (q_levels - 1))[:, 0]
                print(f'{j}', end=',')
        return samples

    def load(self):
        files = [os.path.join(save_dir, f) for f in os.listdir(save_dir) if os.path.isfile(os.path.join(save_dir, f))]
        recent_file = max(files, key=os.path.getmtime)
        if recent_file:
            self.pixelcnn.load_weights(recent_file)
            self.epoch = int(os.path.basename(recent_file).split('.')[0]) + 1

model = Model()
model.load()
model.train(n_epochs=20)


# Check the first samples
print('Training examples')
fig,ax = plt.subplots(1, 10, figsize=(15, 10), dpi=80)
for i in range(10):
    ax[i].imshow(x_train_quantised_of[i]/(q_levels-1))
    #ax[i].imshow(x_train[i])
plt.show()

print('Testing examples')
fig,ax = plt.subplots(1, 10, figsize=(15, 10), dpi=80)
for i in range(10):
    ax[i].imshow(x_test_quantised_of[i]/(q_levels-1))
    #ax[i].imshow(x_test[i])
plt.show()


n_samples = 10
samples = model.generate(n_samples)
fig,ax = plt.subplots(1, 10, figsize=(15, 10), dpi=80)
for i in range(n_samples):
    ax[i].imshow(samples[i, :, :, :])
plt.show()