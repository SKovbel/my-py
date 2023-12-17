# The Inception architecture uses multiple filters at each layer, allowing the model to capture information at different scales.
# InceptionV3 is a popular variant.
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

model_inception = InceptionV3(weights='imagenet', include_top=True)

img_path = '_models\Computer-Vision-Models\cran.png'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Make predictions
predictions_inception = model_inception.predict(img_array)
print(predictions_inception)