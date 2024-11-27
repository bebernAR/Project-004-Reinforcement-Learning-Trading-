import tensorflow as tf
import numpy as np
from collections import deque

class FinanceAgent:
    def __init__(self, state_dim, action_space, max_epochs=10, max_steps=500,
                 gamma=0.95, epsilon=0.5, epsilon_min=0.05, epsilon_decay=0.99,
                 batch_size=64, learning_rate=0.0001, replay_buffer_size=32000):
        self.state_dim = state_dim
        self.action_space = action_space
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Replay buffer
        self.replay_buffer = deque(maxlen=replay_buffer_size)

        # Initialize networks
        self.q_network = self._init_q_network()
        self.target_q_network = self._init_q_network()
        self.target_q_network.set_weights(self.q_network.get_weights())

        # Optimizer and loss function
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.loss_fn = tf.keras.losses.Huber()

    def _init_q_network(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.state_dim,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_space, activation='linear')
        ])
        return model

    def add_to_replay_buffer(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def sample_from_replay_buffer(self):
        batch = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.replay_buffer[i] for i in batch])
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample a batch from the replay buffer
        states, actions, rewards, next_states, dones = self.sample_from_replay_buffer()

        # Predict Q-values for next states
        next_q_values = self.target_q_network.predict(next_states, verbose=0)
        max_next_q_values = np.max(next_q_values, axis=1)
        target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        with tf.GradientTape() as tape:
            # Predict Q-values for current states
            q_values = self.q_network(states, training=True)
            actions_one_hot = tf.one_hot(actions, self.action_space)
            q_values_for_actions = tf.reduce_sum(q_values * actions_one_hot, axis=1)

            # Compute loss
            loss = self.loss_fn(target_q_values, q_values_for_actions)

        # Backpropagation
        grads = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))

    def train(self, train_data):
        total_rewards = []
        for epoch in range(self.max_epochs):
            total_reward = 0
            state = self.get_initial_state(train_data)
            for step in range(len(train_data) - 1):
                if np.random.rand() < self.epsilon:
                    action = np.random.choice(self.action_space)
                else:
                    state_tensor = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
                    q_values = self.q_network(state_tensor, training=False)
                    action = np.argmax(q_values.numpy())

                next_state = self.get_next_state(train_data.iloc[step + 1])
                reward = self.calculate_reward(action, train_data.iloc[step + 1]['Profit'])
                done = step == (len(train_data) - 2)
                self.add_to_replay_buffer(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                self.train_step()

                if done:
                    break

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            total_rewards.append(total_reward)
            print(f"Epoch {epoch + 1}/{self.max_epochs}, Total Reward: {total_reward}")

        return total_rewards

    def get_initial_state(self, row):
        # Convert the row to the initial state format
        return np.array([
            row['Returns_binned'], row['SMA_Ratio_binned'], row['MACD_hist_binned'],
            row['RSI_binned'], row['SO_diff_binned'], row['position']
        ], dtype=np.float32)

    def get_next_state(self, row):
        # Convert the row to the next state format
        return self.get_initial_state(row)

    def calculate_reward(self, position, profit):
        if position == 2:
            return profit
        elif position == 0:
            return -profit
        else:
            return 0

    def update_target_network(self):
        self.target_q_network.set_weights(self.q_network.get_weights())

    def save_model(self, path):
        self.q_network.save(path)

    def load_model(self, path):
        self.q_network = tf.keras.models.load_model(path)
        self.target_q_network.set_weights(self.q_network.get_weights())
