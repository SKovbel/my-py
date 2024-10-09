import os

class Config:
    DATA_DIR = 'tmp/arc/'
    SAVE_DIR = 'tmp/arc-checkpoints/'

    VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    COLORS = [
        (128, 0, 0), (0, 0, 255), (255, 0, 0), # red, black, blue
        (0, 128, 0), (255, 255, 0), (211, 211, 211), # green, yellow, lightgray
        (255, 0, 255), (255, 165, 0), (173, 216, 230), (128, 0, 0)] # magenta, orange, lightblue, maroon}

    HEIGHT = 9
    WIDTH = 9
    CHANNEL = len(COLORS)

    BATCH = 100
    EPOCHS = 10
    BUFFER = 500

os.makedirs(Config.SAVE_DIR, exist_ok=True)
