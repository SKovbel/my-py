import numpy as np
import tensorflow as tf
from config import Config

class Model():
    DIR = Config.join_path(Config.SAVE_DIR, 'cnn0')
    BATCH = 32
    EPOCHs = 0
    RUN_EPOCHS = 10
    MAX_EPOCHS = 220
    INOUT_SHAPE = (Config.WIDTH, Config.HEIGHT, 1)  # height, width, channels number
    THRESHOLD = 0.5 # every prediction with val > THRESHOLD is 1 else 0

    def __init__(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=self.INOUT_SHAPE),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),

            tf.keras.layers.Conv2DTranspose(128, (3, 3), activation='relu'),
            tf.keras.layers.Conv2DTranspose(64, (3, 3), activation='relu'),
            tf.keras.layers.Conv2DTranspose(32, (3, 3), activation='relu'),
            tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid'),
        ])

        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.__load()

    def fit(self, x, y):
        history = None
        print(x[0])

        if self.MAX_EPOCHS > self.EPOCHS:
            history = self.model.fit(x, y, epochs=self.RUN_EPOCHS, batch_size=self.BATCH)
            self.EPOCHS += len(history.history['loss'])
            self.__save()
        return history

    def predict(self, x):
        predictions = self.model(x).numpy()
        predictions = np.where(predictions >= self.THRESHOLD, 1, 0)
        return predictions

    def __load(self):
        recent_file, recent_filename = Config.get_recent_file(self.DIR)
        if recent_file:
            self.model.load_weights(recent_file)
            self.EPOCHS = int(recent_filename.split('.')[0]) + 1
            print(f'Loaded, last epoch {self.EPOCHS}')

    def __save(self):
        filename = Config.join_path(self.DIR, f'{self.EPOCHS:2d}.weights.h5')
        self.model.save_weights(filename)
        print(f'Saved, last epoch {self.EPOCHS}')
