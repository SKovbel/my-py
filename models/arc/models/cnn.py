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
