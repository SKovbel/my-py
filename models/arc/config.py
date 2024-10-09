import os
import numpy as np
import random as rn
import tensorflow as tf

class Config:
    RANDOM_SEED = 13

    CURR_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = 'data'
    SAVE_DIR = 'tmp/arc-checkpoints/'

    VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    COLORS = [
        (128, 0, 0), (0, 0, 255), (255, 0, 0), # red, black, blue
        (0, 128, 0), (255, 255, 0), (211, 211, 211), # green, yellow, lightgray
        (255, 0, 255), (255, 165, 0), (173, 216, 230), (128, 0, 0)] # magenta, orange, lightblue, maroon}

    HEIGHT = 12
    WIDTH = 12
    CHANNEL = len(COLORS)

    def init():
        os.makedirs(Config.SAVE_DIR, exist_ok=True)

        tf.random.set_seed(Config.RANDOM_SEED)
        np.random.seed(Config.RANDOM_SEED)
        rn.seed(Config.RANDOM_SEED)

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    def data_path(path):
        return Config.join_path(Config.join_path(Config.CURR_DIR, Config.DATA_DIR), path)

    def join_path(dir, filename):
        return os.path.join(dir, filename)

    def get_recent_file(DIR):
        os.makedirs(DIR, exist_ok=True)

        recent_file = None
        file_list = os.listdir(DIR)
        if file_list:
            files = [os.path.join(DIR, f) for f in file_list if os.path.isfile(os.path.join(DIR, f))]
            recent_file = max(files, key=os.path.getmtime)
        if recent_file:
            return recent_file, os.path.basename(recent_file)
        return None, None
                