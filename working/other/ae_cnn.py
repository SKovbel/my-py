import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load(dir='./images'):
    data = []
    for filename in os.listdir(dir):
        image = cv2.imread(os.path.join(dir, filename), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (512, 512))
        image = image.astype('float32') / 255.0  # Scale pixel values to [0, 1]
        data.append(image)
    return np.array(data)


def show(data, encoded):
    plt.figure(figsize=(10, 4))
    print(len(data))
    n=len(data)
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(data[i].reshape(512, 512))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(encoded[i].reshape(512, 512))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
