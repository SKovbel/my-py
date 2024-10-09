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
