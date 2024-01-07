import os

CLASSES = {0: 'OBJECT'}
CLASSES = ['object']

DIR = os.path.dirname(os.path.realpath(__file__))
path = lambda name: os.path.join(DIR, name)


def load_model(model_path):
    return YOLO(model_path)  # load a custom trained model