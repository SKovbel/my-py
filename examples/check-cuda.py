# https://www.tensorflow.org/install/source#gpu
import torch

# Check if GPU is available
print("GPU Available: ", torch.cuda.is_available())

# Check PyTorch is using GPU
print("PyTorch GPU Support: ", torch.backends.cudnn.enabled)

import tensorflow as tf
physical_devices = tf.config.list_physical_devices(device_type=None) 
print(physical_devices)