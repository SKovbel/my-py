import os
import time
import json
import numpy as np
import random as rn
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from sklearn.preprocessing import StandardScaler

# Defining random seeds
random_seed = 42
tf.random.set_seed(random_seed)
np.random.seed(random_seed)
rn.seed(random_seed)

DATA_DIR = 'tmp/arc/'
SAVE_DIR = 'tmp/arc-checkpoints/'
os.makedirs(SAVE_DIR, exist_ok=True)

COLORS = [
    (0, 0, 0), (0, 0, 255), (255, 0, 0), # red, black, blue
    (0, 128, 0), (255, 255, 0), (211, 211, 211), # green, yellow, lightgray
    (255, 0, 255), (255, 165, 0), (173, 216, 230), (128, 0, 0)] # magenta, orange, lightblue, maroon}
HEIGHT = 32
WIDTH = 32
CHANNEL = len(COLORS)

BATCH = 80
EPOCHS = 1
BUFFER = 500

class Preparation:
    def __init__(self):
        self.scaler = StandardScaler()
        self.keys = []
        data = []

        with open(f'{DATA_DIR}/arc-agi_training_challenges.json', 'r') as file:
            train = json.load(file)
            for key in train:
                self.keys.append(key)

                key_id = self.keys.index(key)
                data.append([[],[],[],[]])
                for index in range(0, len(train[key]['train'])):
                    data[key_id][0].append(self.__expand(train[key]['train'][index]['input']))
                    data[key_id][1].append(self.__expand(train[key]['train'][index]['output']))
                data[key_id][2].append(self.__expand(train[key]['test'][0]['input']))

        with open(f'{DATA_DIR}/arc-agi_training_solutions.json', 'r') as file:
            test = json.load(file)
            key_id = self.keys.index(key)
            for key in train:
                data[key_id][3].append(self.__expand(test[key][0]))

        self.data = np.array(data)

    def __expand(self, input):
        # scale
        output = np.zeros((WIDTH, HEIGHT), dtype=np.int64)
        output[:len(input), :len(input[0])] = input
        # expand
        return (output[None, :, :] == np.arange(1, CHANNEL + 1)[:, None, None]).astype(int)

    def debug(self):
        with np.printoptions(threshold=np.inf):
            print(self.X_train[0])
            print(self.y_train[0])

    def sample_to_image(self, sample, color=None):
        sample = np.expand_dims(sample, axis=0) if sample.ndim == 2 else sample
        image = Image.new('RGB', (sample.shape[2], sample.shape[1]))
        for c in range(0, sample.shape[0]):
            for y in range(0, sample.shape[1]):
                for x in range(0, sample.shape[2]):
                    if sample[c][y][x] != 0:
                        image.putpixel((x, y), COLORS[color] if color else COLORS[c])
        return image

    def size(self, key_id=0):
        return len(self.data[self.keys[key_id]])

    def samples(self, key_id=0):
        return self.data[self.keys[key_id]]

    def plot_layers(self, key_id=0, img_id=0):
        data = self.samples(key_id)
        X = data[0][img_id]
        Y = data[1][img_id]
        _, axes = plt.subplots(len(X)//2, 4, figsize=(20, 20))
        axes = np.atleast_2d(axes)
        for c in range(0, len(X)):
            d = 0 if c % 2 == 0 else 2
            axes[c // 2, d+0].imshow(self.sample_to_image(X[c], c))
            axes[c // 2, d+1].imshow(self.sample_to_image(Y[c], c))
        plt.show()

    def plot_sample(self, key_id=0, img_id=0):
        data = self.samples(key_id)
        _, axes = plt.subplots(1, 2, figsize=(20, 20))
        axes[0].imshow(self.sample_to_image(data[0][img_id]))
        axes[1].imshow(self.sample_to_image(data[1][img_id]))
        plt.show()

    def plot_test(self, key_id=0):
        data = self.samples(key_id)
        _, axes = plt.subplots(1, 2, figsize=(20, 20))
        axes[0].imshow(self.sample_to_image(data[2]))
        axes[1].imshow(self.sample_to_image(data[3]))
        plt.show()

prep = Preparation()

idx = 22
#prep.plot_layers(key_id=0, img_id=0)
#prep.plot_sample(key_id=0, img_id=0)
#prep.plot_test(key_id=0)

samples = prep.samples(0)
X_train, y_train = samples['X_train'], samples['y_train']
X_test, y_test = samples['X_test'], samples['y_test']

X_train[X_train == 0] = -1
y_train[y_train == 0] = -1
X_test[X_test == 0] = -1
y_test[y_test == 0] = -1

# normilize to 0..1
X_train = X_train / 255.
X_test = X_train / 255.

# transpose to whc
X_train = X_train.transpose(0, 3, 2, 1)
X_test = X_train.transpose(0, 3, 2, 1)

# Creating input stream
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, X_train))
train_dataset = train_dataset.shuffle(buffer_size=BUFFER)
train_dataset = train_dataset.batch(BATCH)

#test_dataset = tf.data.Dataset.from_tensor_slices((X_test_q / (q_levels - 1), X_test_q.astype('int32')))
#test_dataset = test_dataset.batch(batch_size)


class MaskedConv2D(tf.keras.layers.Layer):
    def __init__(
        self,
        mask_type,
        name,
        filters,
        kernel_size,
        strides=1,
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
        self.kernel_size = kernel_size
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)

    def __build_mask(self):
        center = self.kernel_size // 2
        self.mask = np.ones(self.kernel.shape, dtype=np.float32)
        self.mask[center, center + 1:, :, :] = 0
        self.mask[center + 1:, :, :, :] = 0

        for i in range(CHANNEL):
            for j in range(CHANNEL):
                if (self.mask_type == 'A' and i >= j) or (self.mask_type == 'B' and i > j):
                    self.mask[center, center, i::CHANNEL, j::CHANNEL] = 0
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
    
class ScheduleLearningRate(LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_rate):
        super(ScheduleLearningRate, self).__init__()
        self.initial_learning_rate = tf.convert_to_tensor(initial_learning_rate, dtype=tf.float32)
        self.decay_rate = tf.convert_to_tensor(decay_rate, dtype=tf.float32)
    
    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        return self.initial_learning_rate * tf.math.pow(self.decay_rate, step)
    
class Model:
    def __init__(self):
        self.epoch = 0
        self.n_filters = 120
        self.learning_rate_schedule = ScheduleLearningRate(initial_learning_rate=5e-3, decay_rate=0.9999)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_schedule)
        self.compute_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.__build_model()

    def __build_model(self):
        inputs = tf.keras.layers.Input(shape=(HEIGHT, WIDTH, CHANNEL))
        x = MaskedConv2D("A", name="A", filters=self.n_filters, kernel_size=7)(inputs)
        for i in range(15):
            x = tf.keras.layers.Activation(activation='relu', name=f"relu-{i}")(x)
            x = ResidualBlock(n_filters=self.n_filters, name=f"rb-{i}")(x)
        x = tf.keras.layers.Activation(activation='relu')(x)
        x = MaskedConv2D("B", name="B", filters=self.n_filters, kernel_size=1)(x) 
        x = tf.keras.layers.Activation(activation='relu')(x)
        x = MaskedConv2D("B", name="C", filters=CHANNEL, kernel_size=1)(x)
        self.pixelcnn = tf.keras.Model(inputs=inputs, outputs=x)

    @tf.function
    def __train_step(self, batch_x, batch_y):
        with tf.GradientTape() as ae_tape:
            logits = self.pixelcnn(batch_x, training=True)
            logits = tf.reshape(logits, [-1, HEIGHT, WIDTH, CHANNEL])
            logits = tf.transpose(logits, perm=[0, 1, 2, 3])
            loss = self.compute_loss(batch_y, logits)
        gradients = ae_tape.gradient(loss, self.pixelcnn.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        self.optimizer.apply_gradients(zip(gradients, self.pixelcnn.trainable_variables))

    def __print_weights(self):
        for layer in self.pixelcnn.layers:
            print(f"Layer: {layer.name}")
            weights = layer.get_weights()
            for i, weight in enumerate(weights):
                print(f"  Weight {i}: shape={weight.shape}")
                print(weight)

    def train(self, n_epochs=1):
        for epoch in range(self.epoch, n_epochs):
            start = time.time()
            for i_batch, (batch_x, batch_y) in enumerate(train_dataset):
                self.__train_step(batch_x, batch_y)
                self.__print_weights()
                print(f'Epoch {epoch}, batch {i_batch}, shape: {batch_x.shape}, time: {time.time() - start:.2f}')
            self.pixelcnn.save_weights(os.path.join(SAVE_DIR, f'{epoch:2d}.weights.h5'))
        print('Training complete.')
        exit(0)

    def generate(self, n_samples=9):
        samples = np.zeros((n_samples, HEIGHT, WIDTH, CHANNEL), dtype='float32')
        for i in range(HEIGHT):
            print(f'Height position {i}')
            for j in range(WIDTH):
                for k in range(CHANNEL):
                    logits = self.pixelcnn(samples)
                    logits = tf.reshape(logits, [-1, HEIGHT, WIDTH, CHANNEL])
                    logits = tf.transpose(logits, perm=[0, 1, 2, 3])
                    next_sample = tf.random.categorical(logits[:, i, j, k], 1)
                    samples[:, i, j, k] = (next_sample.numpy())[:, 0]
                print(f'{j}', end=',')
        return samples

    def load(self):
        file_list = os.listdir(SAVE_DIR)
        if file_list and False:
            files = [os.path.join(SAVE_DIR, f) for f in file_list if os.path.isfile(os.path.join(SAVE_DIR, f))]
            recent_file = max(files, key=os.path.getmtime)
            if recent_file:
                self.pixelcnn.load_weights(recent_file)
                self.epoch = int(os.path.basename(recent_file).split('.')[0]) + 1
                self.__print_weights()

model = Model()
model.load()
model.train(n_epochs=EPOCHS)


n_samples = 10
samples = model.generate(n_samples)
fig,ax = plt.subplots(1, 10, figsize=(15, 10), dpi=80)
for i in range(n_samples):
    ax[i].imshow(samples[i, :, :, :])
plt.show()