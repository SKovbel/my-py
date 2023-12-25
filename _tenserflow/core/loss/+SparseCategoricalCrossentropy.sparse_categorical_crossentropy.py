#  The SparseCategoricalCrossentropy loss is commonly used in multi-class classification problems where 
#   the target labels are integers representing the class indices
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Generate some random data for demonstration
num_classes = 3
num_samples = 100
input_dim = 10

# Dummy data
x_train = tf.random.normal((num_samples, input_dim))
y_train = tf.random.uniform((num_samples,), minval=0, maxval=num_classes, dtype=tf.int32)

# Create a simple model
model = Sequential([
    Dense(64, activation='relu', input_shape=(input_dim,)),
    Dense(num_classes, activation='softmax')
])

# Compile the model with sparse categorical crossentropy loss
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)

