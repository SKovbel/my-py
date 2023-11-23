import os
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from PIL import Image
tmp_dir = os.path.join(os.path.dirname(__file__), '../../../tmp/keras-basic')

def get_classes():
    import requests

    imagenet_classes_url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
    response = requests.get(imagenet_classes_url)

    imagenet_classes = response.json() if response.status_code == 200 else Exception("Failed to download ImageNet class labels")

    # Extract class labels
    class_labels = [imagenet_classes[str(i)][1] for i in range(1000)]
    return class_labels

# vgg19
vgg19 = keras.applications.VGG19()
features_list = [layer.output for layer in vgg19.layers]

feat_extraction_model = keras.Model(inputs=vgg19.input, outputs=features_list)

img = np.random.random((1, 224, 224, 3)).astype("float32")
extracted_features = feat_extraction_model(img)
print(extracted_features)

extracted_classes = get_classes()
print(extracted_classes)

exit(0)
together = zip(extracted_classes, extracted_features)
# Print the zipped elements
for item in together:
    print(item)
