import numpy as np
import tensorflow as tf
import pickle

class DQN:
    def __init__(self, input_dim, action_space, learning_rate=0.001):
        self.input_dim = input_dim
        self.action_space = action_space
        self.memory = []
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.model = self.create_model(learning_rate)

    def create_model(self, learning_rate):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(64, input_dim=self.input_dim, activation='relu'))
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate))
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_space)
        return np.argmax(self.model.predict(state)[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        minibatch_loss = []
        if len(self.memory) < batch_size:
            return
        indices = np.random.randint(0, len(self.memory), size=batch_size)  # Sample indices
        minibatch = [self.memory[i] for i in indices]  # Fetch experiences using the indices

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.train_on_batch(state, target_f)
            loss = self.model.train_on_batch(state, target_f)
            minibatch_loss.append(loss)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        avg_loss = np.mean(minibatch_loss) if minibatch_loss else 0
        return avg_loss

    def save_model(self, filename):
        self.model.save(filename)
        # Save epsilon value
        with open(filename + '_epsilon.pkl', 'wb') as f:
            pickle.dump(self.epsilon, f)

    def load_model(self, filename):
        self.model = tf.keras.models.load_model(filename)
        # Load epsilon value
        try:
            with open(filename + '_epsilon.pkl', 'rb') as f:
                self.epsilon = pickle.load(f)
        except FileNotFoundError:
            # If the file doesn't exist, keep the default epsilon
            pass
