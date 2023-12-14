# ImageNet is a large-scale visual database designed for use in visual object recognition software research.
# It contains millions of labeled images in thousands of categories.
# The ImageNet Large Scale Visual Recognition Challenge (ILSVRC) is a well-known competition that has played a significant role in advancing the field of computer vision.
# Many deep learning models for image classification, including convolutional neural networks (CNNs), have been trained and benchmarked on the ImageNet dataset.

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet')

# Load and preprocess the image
img_path = 'path_to_your_image.jpg'  # Replace with the path to your image file
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Make predictions
predictions = model.predict(img_array)

# Decode and print the top-3 predicted classes
decoded_predictions = decode_predictions(predictions, top=3)[0]
print("Predictions:")
for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
    print(f"{i + 1}: {label} ({score:.2f})")
