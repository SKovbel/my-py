# https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/

from tensorflow.keras.applications import EfficientNetB0
model = EfficientNetB0(weights='imagenet')
model = EfficientNetB0(include_top=False, weights='imagenet')
model = EfficientNetB0(weights='imagenet', drop_connect_rate=0.4)
