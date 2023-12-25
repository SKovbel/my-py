import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Generate some toy data (binary visible units)
visible_units = np.random.randint(2, size=(100, 10))

# Convert data to float32 (required by TensorFlow)
visible_units = visible_units.astype(np.float32)

# Define the RBM model
class RBM(tf.Module):
    def __init__(self, num_visible, num_hidden):
        self.W = tf.Variable(tf.random.normal([num_visible, num_hidden], stddev=0.01))
        self.bias_visible = tf.Variable(tf.zeros([num_visible]))
        self.bias_hidden = tf.Variable(tf.zeros([num_hidden]))

    def energy(self, v, h):
        return -tf.reduce_sum(tf.matmul(v, self.W) * h) - tf.reduce_sum(self.bias_visible * v) - tf.reduce_sum(self.bias_hidden * h)

    def probabilities_hidden(self, v):
        return tf.nn.sigmoid(tf.matmul(v, self.W) + self.bias_hidden)

    def probabilities_visible(self, h):
        return tf.nn.sigmoid(tf.matmul(h, tf.transpose(self.W)) + self.bias_visible)

    def sample_hidden(self, v):
        return tf.cast(tf.random.uniform(tf.shape(self.probabilities_hidden(v))) < self.probabilities_hidden(v), dtype=tf.float32)

    def sample_visible(self, h):
        return tf.cast(tf.random.uniform(tf.shape(self.probabilities_visible(h))) < self.probabilities_visible(h), dtype=tf.float32)

    def contrastive_divergence(self, v, k=1, learning_rate=0.1):
        for _ in range(k):
            h = self.sample_hidden(v)
            v_reconstructed = self.sample_visible(h)

        positive_phase = tf.matmul(tf.transpose(v), h)
        negative_phase = tf.matmul(tf.transpose(v_reconstructed), self.sample_hidden(v_reconstructed))

        gradient_W = (positive_phase - negative_phase) / tf.cast(tf.shape(v)[0], dtype=tf.float32)

        # Update weights and biases
        self.W.assign_add(learning_rate * gradient_W)
        self.bias_visible.assign_add(learning_rate * tf.reduce_mean(v - v_reconstructed, axis=0))
        self.bias_hidden.assign_add(learning_rate * tf.reduce_mean(h - self.sample_hidden(v_reconstructed), axis=0))

# Instantiate the RBM
num_visible_units = visible_units.shape[1]
num_hidden_units = 5
rbm = RBM(num_visible_units, num_hidden_units)

# Train the RBM
num_epochs = 100
for epoch in range(num_epochs):
    for batch in visible_units:
        rbm.contrastive_divergence(tf.expand_dims(batch, axis=0))

# Generate samples using Gibbs sampling
visible_units_sampled = np.random.randint(2, size=(10, num_visible_units)).astype(np.float32)
print(visible_units_sampled)

for _ in range(1000):
    hidden_units_sampled = rbm.sample_hidden(visible_units_sampled)
    visible_units_sampled = rbm.sample_visible(hidden_units_sampled)

# Print the sampled visible units
print("Generated Samples:")
print(visible_units_sampled.numpy())
