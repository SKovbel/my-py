import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from config import Config

class Plot:
    def plot_image(self, sample):
        sample = sample.reshape(Config.WIDTH, Config.HEIGHT)
        image = Image.new('RGB', (Config.WIDTH, Config.HEIGHT))
        for y in range(0, Config.HEIGHT):
            for x in range(0, Config.WIDTH):
                color = sample[y][x]
                if color != 0:
                    image.putpixel((x, y), Config.COLORS[color] if color else Config.COLORS[color])
        return image

    def plot_samples(self, samples):
        count = len(samples)
        _, axes = plt.subplots(count+1, 2, figsize=(20, 20))
        axes = np.atleast_2d(axes)
        for i in range(0, count):
            axes[i, 0].imshow(self.plot_image(samples['X_train'][i]))
            axes[i, 1].imshow(self.plot_image(samples['y_train'][i]))
        axes[count, 0].imshow(self.plot_image(samples['X_test']))
        axes[count, 1].imshow(self.plot_image(samples['y_test']))
        plt.show()

    def plot_sample(self, samples, img_id=0):
        _, axes = plt.subplots(1, 2, figsize=(20, 20))
        axes[0].imshow(self.plot_image(samples['X_train'][img_id]))
        axes[1].imshow(self.plot_image(samples['y_train'][img_id]))
        plt.show()

    def plot_test(self, samples):
        _, axes = plt.subplots(1, 2, figsize=(20, 20))
        axes[0].imshow(self.plot_image(samples['X_test']))
        axes[1].imshow(self.plot_image(samples['y_test']))
        plt.show()
