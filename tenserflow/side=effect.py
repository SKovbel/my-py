import tensorflow as tf

class Model(tf.Module):
  def __init__(self):
    self.v = tf.Variable(0)
    self.counter = 0

  @tf.function
  def __call__(self):
    if self.counter == 0:
      # A python side-effect
      self.counter += 1
      self.v.assign_add(1)
    return self.v

m = Model()
for n in range(3):
  print(m().numpy()) # prints 1, 2, 3



class Model2(tf.Module):
  def __init__(self):
    self.v = tf.Variable(0)
    self.counter = 0

  @tf.function
  def __call__(self):
    if self.counter == 0:
      with tf.init_scope():
        self.counter += 1
        self.v.assign_add(1)
    return self.v

m = Model2()
for n in range(3):
  print(m().numpy()) # prints 1, 1, 1