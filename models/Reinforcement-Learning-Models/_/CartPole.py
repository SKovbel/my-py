import numpy as np
import tensorflow as tf
import gym

# Create the CartPole environment
env = gym.make('CartPole-v1')

# Define the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(24, activation='relu', input_shape=(env.observation_space.shape[0],)),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(env.action_space.n, activation='linear')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='mse')

# Function to choose an action using epsilon-greedy policy
def choose_action(state, epsilon):
    if np.random.rand() < epsilon:
        return env.action_space.sample()  # Explore
    else:
        q_values = model.predict(state.reshape(1, -1))
        return np.argmax(q_values[0])  # Exploit

# Training parameters
num_episodes = 500
gamma = 0.99  # Discount factor
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay_steps = 200

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    state = np.reshape(state, [1, env.observation_space.shape[0]])

    epsilon = max(epsilon_end, epsilon_start - episode / epsilon_decay_steps)

    total_reward = 0
    done = False
    while not done:
        # Choose an action
        action = choose_action(state, epsilon)

        # Take the chosen action and observe the next state and reward
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])

        # Calculate the target Q-value
        target = reward + gamma * np.max(model.predict(next_state))

        # Get the current Q-values
        q_values = model.predict(state)

        # Update the Q-value for the chosen action
        q_values[0, action] = target

        # Train the model with the updated Q-values
        model.fit(state, q_values, verbose=0)

        total_reward += reward
        state = next_state

    print(f"Episode {episode + 1}, Total Reward: {total_reward}")

# Test the trained model
total_reward = 0
state = env.reset()
state = np.reshape(state, [1, env.observation_space.shape[0]])
while True:
    env.render()
    action = np.argmax(model.predict(state))
    next_state, reward, done, _ = env.step(action)
    state = np.reshape(next_state, [1, env.observation_space.shape[0]])
    total_reward += reward
    if done:
        break

print(f"Test Total Reward: {total_reward}")

# Close the environment
env.close()
