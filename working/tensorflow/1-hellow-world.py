import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf

x = tf.constant('Hello world')
print(x.numpy())