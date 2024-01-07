import tensorflow as tf
from tensorflow.keras.metrics import Metric
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

class CustomAccuracy(Metric):
    def __init__(self, name='custom_accuracy', **kwargs):
        super(CustomAccuracy, self).__init__(name=name, **kwargs)
        self.total_samples = self.add_weight(name='total_samples', initializer='zeros')
        self.correct_predictions = self.add_weight(name='correct_predictions', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, dtype=tf.keras.backend.floatx())  # Cast y_true to the same dtype as y_pred
        y_pred_classes = tf.argmax(y_pred, axis=-1)
        correct = tf.cast(tf.equal(y_true, tf.cast(y_pred_classes, dtype=y_true.dtype)), dtype=tf.float32)
        self.total_samples.assign_add(tf.cast(tf.size(y_true), dtype=tf.float32))
        self.correct_predictions.assign_add(tf.reduce_sum(correct))

    def result(self):
        return self.correct_predictions / self.total_samples

# Load the Iris dataset for demonstration
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a simple neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Compile the model with custom accuracy metric
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=[CustomAccuracy()])

# Train the model
model.fit(X_train, y_train, epochs=50, verbose=1)

# Evaluate the model
loss, custom_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss}")
print(f"Test Custom Accuracy: {custom_accuracy}")
