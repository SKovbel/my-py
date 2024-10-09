<<<<<<< HEAD
import numpy as np
import random as rn
import tensorflow as tf
from dataset import Dataset

# Defining random seeds
random_seed = 13
tf.random.set_seed(random_seed)
np.random.seed(random_seed)
rn.seed(random_seed)

ds = Dataset()

#ds_batch = ds.batch()
ds_series = ds.series()
ds_channel = ds.channels()
=======
import numpy as np
from config import Config
from dataset import Dataset
from plot import Plot

CASE = 1

# Defining random seeds

ds = Dataset()
dst = Dataset(what='test')
Config.init()

if CASE == 1:
    from models.cnn import Model as CNNModel
    model = CNNModel()
    plot = Plot()

    train_x, train_y, test_x, test_y = ds.channel(type='wh')
    test_x, test_y, prob_x, prob_y = dst.channel(type='wh_series')

    #imgs = [img for i in range(10) for img in (test_x[i], test_y[i], test_y[i])]
    #plot.plot_images(imgs, ncols=3)

    model.fit(train_x, train_y)

    a=0
    b=5

    pred_y = model.predict(prob_x[a:b])
    imgs = [img for i in range(b-a) for img in (test_x[i][0], test_y[i][0], prob_x[i], pred_y[i])]
    plot.plot_images(imgs, ncols=4)

    pred_y2 = model.predict_tail(test_x[a:b], test_y[a:b], prob_x[a:b])
    imgs2 = [img for i in range(b-a) for img in (test_x[i][0], test_y[i][0], prob_x[i], pred_y2[i])]
    plot.plot_images(imgs2, ncols=4)
>>>>>>> 13d3c5d91edb12c44a012de06137dcdf5651d9ab
