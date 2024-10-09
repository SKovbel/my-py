import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from config import Config

class Plot:
    def __plot_image(self, sample):
        sample = sample.reshape(Config.WIDTH, Config.HEIGHT)
        image = Image.new('RGB', (Config.WIDTH, Config.HEIGHT))
        for y in range(0, Config.HEIGHT):
            for x in range(0, Config.WIDTH):
                color = sample[y][x]
                if color != 0:
                    image.putpixel((x, y), Config.COLORS[color] if color else Config.COLORS[color])
        return image

    def plot_image(self, image):
        plt.imshow(self.__plot_image(image))
        plt.show()  

    def plot_images(self, images, ncols=1):
        nrows = len(images) // ncols
        _, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
        axes = axes.flatten()
        for ax, image in zip(axes, images):
            ax.imshow(self.__plot_image(image))
        plt.show()

    def plot_samples(self, samples):
        count = len(samples)
        _, axes = plt.subplots(count+1, 2, figsize=(20, 20))
        axes = np.atleast_2d(axes)
        for i in range(0, count):
            axes[i, 0].imshow(self.__plot_image(samples['X_train'][i]))
            axes[i, 1].imshow(self.__plot_image(samples['y_train'][i]))
        axes[count, 0].imshow(self.__plot_image(samples['X_test']))
        axes[count, 1].imshow(self.__plot_image(samples['y_test']))
        plt.show()

    def plot_sample(self, samples, img_id=0):
        _, axes = plt.subplots(1, 2, figsize=(20, 20))
        axes[0].imshow(self.__plot_image(samples['X_train'][img_id]))
        axes[1].imshow(self.__plot_image(samples['y_train'][img_id]))
        plt.show()

    def plot_test(self, samples):
        _, axes = plt.subplots(1, 2, figsize=(20, 20))
        axes[0].imshow(self.__plot_image(samples['X_test']))
        axes[1].imshow(self.__plot_image(samples['y_test']))
        plt.show()


if __name__ == '__main__':
    from dataset import Dataset
    ds = Dataset()
    train_x, train_y, test_x, test_y = ds.channel(type='wh')
    imgs = [img for i in range(10) for img in (test_x[i], test_y[i], test_y[i])]
    Plot().plot_images(imgs)