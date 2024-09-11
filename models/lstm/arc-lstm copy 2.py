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
    (128, 0, 0), (0, 0, 255), (255, 0, 0), # red, black, blue
    (0, 128, 0), (255, 255, 0), (211, 211, 211), # green, yellow, lightgray
    (255, 0, 255), (255, 165, 0), (173, 216, 230), (128, 0, 0)] # magenta, orange, lightblue, maroon}
HEIGHT = 9
WIDTH = 9 #32
CHANNEL = len(COLORS)

BATCH = 100
EPOCHS = 10
BUFFER = 500

class Dataset:
    def __init__(self):
        self.scaler = StandardScaler()
        self.keys = []
        self.data = {}
        self.__load()

    def __load(self):
        train_path = f'{DATA_DIR}/arc-agi_training_challenges.json'
        test_path = f'{DATA_DIR}/arc-agi_training_solutions.json'
        with open(train_path, 'r') as train_file, open(test_path, 'r') as test_file:
            train = json.load(train_file)
            test = json.load(test_file)

            for key in train:
                self.keys.append(key)
                self.data[key] = {'X_train': [], 'y_train': [], 'X_test': None, 'y_test': None}
                self.data[key]['X_test'] = self.__expand(train[key]['test'][0]['input'])
                self.data[key]['y_test'] = self.__expand(test[key][0])
                for index in range(0, len(train[key]['train'])):
                    self.data[key]['X_train'].append(self.__expand(train[key]['train'][index]['input']))
                    self.data[key]['y_train'].append(self.__expand(train[key]['train'][index]['output']))


    def __expand(self, input):
        height, width = min(len(input), HEIGHT), min(len(input[0]), WIDTH)
        output = np.zeros((HEIGHT, WIDTH), dtype=np.int64)
        output[:height, :width] = [row[0:width] for row in input[0:height]]
        return output

    def samples(self, key_id=0):
        return self.data[self.keys[key_id]]

    def train_dataset_1d(self):
        x, y = [], []
        for key in self.data:
            for sample in self.data[key]['X_train']:
                x.append([element for row in sample for element in row])
            for sample in self.data[key]['y_train']:
                y.append([element for row in sample for element in row])
        return tf.data.Dataset.from_tensor_slices((x, y))

    def debug(self, key_id=0, img_id=0):
        print(self.samples(key_id)['X_train'][img_id])

    def plot_image(self, sample):
        sample = sample.reshape(WIDTH, HEIGHT)
        image = Image.new('RGB', (WIDTH, HEIGHT))
        for y in range(0, HEIGHT):
            for x in range(0, WIDTH):
                color = sample[y][x]
                if color != 0:
                    image.putpixel((x, y), COLORS[color] if color else COLORS[color])
        return image

    def plot_samples(self, key_id=0,):
        data = self.samples(key_id)
        count = len(data)
        _, axes = plt.subplots(count+1, 2, figsize=(20, 20))
        axes = np.atleast_2d(axes)
        for i in range(0, count):
            axes[i, 0].imshow(self.plot_image(data['X_train'][i]))
            axes[i, 1].imshow(self.plot_image(data['y_train'][i]))
        axes[count, 0].imshow(self.plot_image(data['X_test']))
        axes[count, 1].imshow(self.plot_image(data['y_test']))
        plt.show()

    def plot_sample(self, key_id=0, img_id=0):
        data = self.samples(key_id)
        _, axes = plt.subplots(1, 2, figsize=(20, 20))
        axes[0].imshow(self.plot_image(data['X_train'][img_id]))
        axes[1].imshow(self.plot_image(data['y_train'][img_id]))
        plt.show()

    def plot_test(self, key_id=0):
        data = self.samples(key_id)
        _, axes = plt.subplots(1, 2, figsize=(20, 20))
        axes[0].imshow(self.plot_image(data['X_test']))
        axes[1].imshow(self.plot_image(data['y_test']))
        plt.show()

class ARCModel(tf.keras.Model):
    def __init__(self, input_size, output_size, hidden_size):
        super(ARCModel, self).__init__()
        self.lstm = tf.keras.layers.LSTM(hidden_size, return_sequences=True, input_shape=(None, input_size))
        self.fc = tf.keras.Sequential([
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(output_size, activation='relu')
        ])

    def call(self, inputs):
        lstm_out = self.lstm(inputs)
        predictions = self.fc(lstm_out)
        return predictions

dataset = Dataset()
#model = ARCModel(input_size=, output_size=WIDTH*HEIGHT, hidden_size=2*WIDTH*HEIGHT)

#prep.debug()
train_dataset_1d = dataset.train_dataset_1d()
print(len(train_dataset_1d))
dataset.plot_samples(key_id=0)
#prep.plot_sample(key_id=0, img_id=0)
#prep.plot_test(key_id=0)
