<<<<<<< HEAD
import tensorflow as tf


class CNNModel(tf.keras.Model):
    def __init__(self, input_size, output_size, hidden_size):
        super(CNNModel, self).__init__()
        self.lstm = tf.keras.layers.LSTM(hidden_size, return_sequences=True, input_shape=(None, input_size))
        self.fc = tf.keras.Sequential([
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(output_size, activation='relu')
        ])

    def create(self):
        pass

    def call(self, inputs):
        lstm_out = self.lstm(inputs)
        predictions = self.fc(lstm_out)
        return predictions
=======
import numpy as np
import tensorflow as tf
from config import Config

class Model():
    DIR = Config.join_path(Config.SAVE_DIR, 'cnn1')
    BATCH = 32
    EPOCHS = 0
    RUN_EPOCHS = 100
    MAX_EPOCHS = 200
    INOUT_SHAPE = (Config.WIDTH, Config.HEIGHT, 1)  # height, width, channels number
    THRESHOLD = 0.5 # every prediction with val > THRESHOLD is 1 else 0

    def __init__(self):
        self.main_model()
        self.__load()

    def fit(self, x, y):
        history = None

        if self.MAX_EPOCHS > self.EPOCHS:
            history = self.model.fit(x, y, epochs=self.RUN_EPOCHS, batch_size=self.BATCH)
            self.EPOCHS += len(history.history['loss'])
            self.__save()
        return history

    def main_model(self):
        input = tf.keras.layers.Input(self.INOUT_SHAPE)
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
        x = tf.keras.layers.Conv2DTranspose(128, (3, 3), activation='relu')(x)
        x = tf.keras.layers.Conv2DTranspose(64, (3, 3), activation='relu')(x)
        x = tf.keras.layers.Conv2DTranspose(32, (3, 3), activation='relu')(x)
        output = tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid')(x)
        self.model = tf.keras.Model(input, output, name='main')
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def tail_model2(self, x, y, prob_x):
        for layer in self.model.layers:
            layer.trainable = False

        input = tf.keras.layers.Input(self.INOUT_SHAPE)
        f = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input)
        f = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(f)
        f = tf.keras.layers.Conv2DTranspose(64, (3, 3), activation='relu')(f)
        f = tf.keras.layers.Conv2DTranspose(32, (3, 3), activation='relu')(f)
        output = tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid')(f)
        tail = tf.keras.Model(input, output, name='tail')

        combined_output = tail(self.model.output)
        combined = tf.keras.Model(inputs=self.model.input, outputs=combined_output)
        combined.compile(optimizer='adam', loss='mean_squared_error')
        combined.fit(x, y, epochs=50, batch_size=10)
        return combined(np.array(prob_x)).numpy()

    def tail_model(self, x, y, prob_x):
        for layer in self.model.layers:
            layer.trainable = False

        input = tf.keras.layers.Input(self.INOUT_SHAPE)
        f = tf.keras.layers.Flatten()(input)
        f = tf.keras.layers.Dense(64, activation='relu')(f)
        f = tf.keras.layers.Dense(12 * 12 * 1, activation='relu')(f)
        output = tf.keras.layers.Reshape((12, 12, 1))(f)
        tail = tf.keras.Model(input, output, name='tail')

        combined_output = tail(self.model.output)
        combined = tf.keras.Model(inputs=self.model.input, outputs=combined_output)
        combined.compile(optimizer='adam', loss='mean_squared_error')
        combined.fit(x, y, epochs=50, batch_size=10)
        return combined(np.array(prob_x)).numpy()

    def predict_tail(self, x, y, prob_x):
        predictions = []
        for i in range(len(x)):
            yy = y[i]
            xx = self.model(x[i])
            a = np.expand_dims(np.array(prob_x[i]), axis=-1)
            prob_y = self.tail_model2(xx, yy, [a])
            predictions.append(np.where(prob_y >= self.THRESHOLD, 1, 0))
        predictions = np.squeeze(predictions, axis=-1)
        return predictions

    def predict(self, x):
        x = np.expand_dims(x, axis=-1)
        predictions = self.model(x).numpy()
        predictions = np.where(predictions >= self.THRESHOLD, 1, 0)
        predictions = np.squeeze(predictions, axis=-1)
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
>>>>>>> 13d3c5d91edb12c44a012de06137dcdf5651d9ab
