# Designed for efficient mobile and edge devices with low computational resources.7
# it's common to find pre-trained models on platforms like TensorFlow Hub or model repositories like Hugging Face Transformers.
# Example using TensorFlow and Keras to load a pre-trained CIFAR-10 model from TensorFlow 
# The base model is MobileNetV2, which is pre-trained on ImageNet. We load it with weights='imagenet'.
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Load CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# One-hot encode the labels
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

# Define the base model using MobileNetV2 from TensorFlow Hub
base_model = MobileNetV2(input_shape=(32, 32, 3), include_top=False, weights='imagenet')

# Freeze the base model
base_model.trainable = False

# Create a new model on top
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Load pre-trained weights on CIFAR-10
model.load_weights('path/to/your/cifar10_pretrained_weights.h5')

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")
