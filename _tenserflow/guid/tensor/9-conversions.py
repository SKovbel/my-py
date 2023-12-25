import tensorflow as tf
import numpy as np

class CustomObject:
    def __init__(self, value):
        self.value = value

def custom_object_to_tensor(custom_obj, dtype=None, name=None, as_ref=None):
    if dtype is not None and dtype != tf.float32:
        raise ValueError("Unsupported dtype")
    return tf.constant(custom_obj.value, dtype=tf.float32, name=name)

tf.register_tensor_conversion_function(CustomObject, custom_object_to_tensor)
custom_instance = CustomObject(42.1)
custom_tensor = tf.convert_to_tensor(custom_instance)
print(custom_tensor)



numpy_array = np.array([1, 2, 3, 4, 5])
tensor = tf.convert_to_tensor(numpy_array, dtype=tf.float32)
result = tf.square(tensor)
print(result)
