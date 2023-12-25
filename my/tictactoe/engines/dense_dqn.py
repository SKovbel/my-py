import numpy as np  # linear algebra
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping
import random
import gym
from collections import deque
from keras.models import Sequential
from keras.optimizers import RMSprop
from tf_agents.environments import py_environment, tf_py_environment, utils, wrappers
from tf_agents.specs import array_spec
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.utils import common
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.trajectories import time_step, policy_step, trajectory

class EngineDenseDQNEnv(py_environment.PyEnvironment):
    def __init__(self, game):
        self.game = game
        self._episode_ended = False
        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=8, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(shape=(9, 1), dtype=np.int32, minimum=0, maximum=1, name='observation')

    def _reset(self):
        #print('_reset')
        self.game.start()
        self._episode_ended = False
        return time_step.restart(np.expand_dims(np.copy(self.game.board()), axis=-1))

    def _step(self, action):
        if self._episode_ended:
            print('_episode_ended')
            self.game.print()
            return self.reset()

        if self.game.is_legal_move(action):
            self.game.move(action)
            print('+'+str(action), end='')
        else:
            print('-'+str(action), end='')
            # Invalid move, penalize the agent
            return time_step.termination(np.expand_dims(np.copy(self.game.board()), axis=-1), -1)

        board = self.game.board()
        if self.game.is_win():
            self._episode_ended = True
            return time_step.termination(np.expand_dims(np.copy(board), axis=-1), 1)
        elif self.game.is_draw():
            self._episode_ended = True
            return time_step.termination(np.expand_dims(np.copy(board), axis=-1), 0)
        else:
            self.game.auto_move()
            return time_step.transition(np.expand_dims(np.copy(board), axis=-1), reward=0.0, discount=1.0)

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec


class EngineDenseDQNQNetwork(q_network.QNetwork):
    def call(self, observation, step_type=None, network_state=(), training=False):
        q_values, network_state = super(EngineDenseDQNQNetwork, self).call(
            observation, step_type=step_type, network_state=network_state
        )
        action_probs = tf.nn.softmax(q_values)
        return action_probs, network_state

class EngineDenseDQN:
    def __init__(self, code='EngineDenseDQN', unique=False):
        # params
        self.env = None
        self.qnet = None
        self.code = code
        self.game = None
        self.agent = None
        self.unique = unique
        self.target_qnet = None
        self.counter = tf.Variable(0)
        self.replay_buffer = None

        self.model = Sequential([
            layers.ReLU(32, input_shape=(9,)),  # timeseries2x
            layers.ReLU(32),
            layers.Normalization(),
            layers.Dense(9, activation='sigmoid')
        ])

        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
        self.model.compile(optimizer=self.optimizer, loss='mse')  # binary_crossentropy
        self.model.summary()

    def init(self, game):
        # env
        if self.game:
            return
        self.game = game
        env = EngineDenseDQNEnv(self.game)
        self.env = tf_py_environment.TFPyEnvironment(env)

        self.qnet = EngineDenseDQNQNetwork(
            self.env.observation_spec(),
            self.env.action_spec(),
            fc_layer_params=(9, 32, 32, 9)
        )

        self.agent = dqn_agent.DqnAgent(
            self.env.time_step_spec(),
            self.env.action_spec(),
            q_network=self.qnet,
            optimizer=self.optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=self.counter
        )
        self.agent.initialize()

        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=self.env.batch_size,
            max_length=100000
        )

        replay_buffer_observer = self.replay_buffer.add_batch

        collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
            self.env,
            self.agent.collect_policy,
            observers=[replay_buffer_observer],
            num_episodes=100000
        )

        #initial_collect_episodes = 5
        #for _ in range(initial_collect_episodes):
        #    collect_driver.run()
        #collect_driver.run()

    def train(self, game):
        final_state = game.board()
        reshaped_observation = np.reshape(final_state, (9, 1))
        final_action = self.game.moves[-1]
        final_reward = 1
        step_type = 1

        time_step_obj = time_step.TimeStep(
            step_type=step_type,
            discount=1,
            reward=final_reward,
            observation=reshaped_observation
        )

        policy_step_obj = policy_step.PolicyStep(action=final_action)

        # Verify the creation of trajectory_obj
        trajectory_obj = trajectory.Trajectory(
            step_type=time_step_obj.step_type,
            observation=time_step_obj.observation,
            action=policy_step_obj.action,
            policy_info=(),
            next_step_type=None,
            reward=time_step_obj.reward,
            discount=time_step_obj.discount
        )

        #batch = tf.nest.map_structure(lambda t: tf.expand_dims(t, 0), trajectory_obj)
        print(trajectory_obj)
        # Attempt to add the batch to the replay buffer
        self.replay_buffer.add_batch(trajectory_obj)

    def move(self, game):
        state = game.board()
        observation_tensor = tf.constant(np.reshape(state, (1, 9, 1)), dtype=tf.int32)
        q_values = self.qnet(observation_tensor)
        q_values = game.remove_illegal_moves(q_values[0].numpy())
        #print(q_values)
        best_move = np.argmax(q_values)
        return best_move

    def end_move(self):
        pass

    def end_game(self):
        pass
