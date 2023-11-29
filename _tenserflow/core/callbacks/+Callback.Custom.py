import tensorflow as tf
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_classes=2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


class CustomLog(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 50 == 0:
            print(f"Epoch {epoch}/{self.params['epochs']}, loss: {logs['loss']:.4f}, accuracy: {logs['accuracy']:.4f}")
    def on_train_end(self, logs=None):
        print(f"End {self.params['epochs']}/{self.params['epochs']}, loss: {logs['loss']:.4f}, accuracy: {logs['accuracy']:.4f}")


models = Sequential(layers=[
    Dense(32, input_dim=20),
    Dense(1, activation='sigmoid')
])

models.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
result = models.fit(X_train, y_train, epochs=200, batch_size=32, verbose=0, callbacks=[CustomLog()])