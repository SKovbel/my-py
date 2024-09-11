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

class Preparation:
    def __init__(self):
        self.scaler = StandardScaler()
        self.keys = []
        self.data = {}

        with open(f'{DATA_DIR}/arc-agi_training_challenges.json', 'r') as file:
            train = json.load(file)
            for key in train:
                self.keys.append(key)
                self.data[key]={'X_train': [], 'y_train': [], 'X_test': None, 'y_test': None}
                for index in range(0, len(train[key]['train'])):
                    self.data[key]['X_train'].append(self.__expand(train[key]['train'][index]['input']))
                    self.data[key]['y_train'].append(self.__expand(train[key]['train'][index]['output']))
                self.data[key]['X_test'] = self.__expand(train[key]['test'][0]['input'])

        with open(f'{DATA_DIR}/arc-agi_training_solutions.json', 'r') as file:
            test = json.load(file)
            for key in train:
                self.data[key]['y_test'] = self.__expand(test[key][0])

        start_time = time.time()
        for key in self.data:
            self.data[key]['X_train'] = np.array(self.data[key]['X_train'])
            self.data[key]['y_train'] = np.array(self.data[key]['y_train'])
            self.data[key]['X_test'] = np.array(self.data[key]['X_test'])
            self.data[key]['y_test'] = np.array(self.data[key]['y_test'])
        end_time = time.time()
        print(end_time - start_time)

    def __expand(self, input):
        height, width = min(len(input), HEIGHT), min(len(input[0]), WIDTH)
        output = np.zeros((HEIGHT, WIDTH), dtype=np.int64)
        output[:height, :width] = [row[0:width] for row in input[0:height]]
        return output.reshape(-1)

    def train_dataset(self):
        pass

    def size(self, key_id=0):
        return len(self.data[self.keys[key_id]])

    def samples(self, key_id=0):
        return self.data[self.keys[key_id]]

    def debug(self, key_id=0, img_id=0):
        with np.printoptions(threshold=np.inf):
            print(self.X_train[0])
            print(self.y_train[0])

    def plot_image(self, sample):
        sample = sample.reshape(WIDTH, HEIGHT)
        image = Image.new('RGB', (WIDTH, HEIGHT))
        for y in range(0, HEIGHT):
            for x in range(0, WIDTH):
                color = sample[y][x]
                if color != 0:
                    image.putpixel((x, y), COLORS[color] if color else COLORS[color])
        return image

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

prep = Preparation()
prep.debug()
prep.plot_sample(key_id=0, img_id=0)
prep.plot_test(key_id=0)
