import os
import sys
import datetime
import math
import random
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber

def NeuralNet(state_size, action_size):
    # CNN model
    # input layer
    input_layer = Input(shape = state_size)
    
    # hidden layers
    x = Conv2D(filters = 64, kernel_size = 2, activation = 'relu')(input_layer)
    x = BatchNormalization()(x)
    x = Conv2D(filters = 128, kernel_size = 2, activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters = 256, kernel_size = 2, activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters = 512, kernel_size = 2, activation = 'relu')(x)
    x = BatchNormalization()(x)

    # output layer
    x = Flatten()(x)
    output_layer = Dense(math.prod(action_size), activation = 'softmax')(x)
    
    # compile model
    model = Model(inputs = input_layer, outputs = output_layer)
    optimizer = Adam(learning_rate = 0.001) 
        # optimization algorithm for gradient descent
        # https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam
    loss = Huber(delta = 1.0)
        # huber loss function in robust regression
        # https://en.wikipedia.org/wiki/Huber_loss
    model.compile(optimizer = optimizer, loss = loss)
    
    return model 


class DQN_Agent:
    def __init__(self):
        # For state_size and action_size, refer to AlphaZero paper
        self._state_size = (8, 8, 14 * 8 + 7)
            # state size: 8 * 8 * 119 = 7,616
        self._action_size = (8, 8, 8 * 7 + 8 + 9)
            # action size: 8 * 8 * 73 = 4,672

        self.experience_replay = deque(maxlen = 2000)

        self.gamma = 0.95
        self.epsilon = 0.90
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99

        self.q_network = NeuralNet(self._state_size, self._action_size)
        self.target_network = NeuralNet(self._state_size, self._action_size)

        self.align_target_model()

    def align_target_model(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.experience_replay.append((state, action, reward, next_state, done))

    def act(self, state, env):
        legal_actions = env._legal_actions()

        if np.random.rand() <= self.epsilon:
            # exploration (pick random action)
            return legal_actions[random.randrange(0, len(legal_actions))]

        # exploitation (invoke Q-network to make prediction)
        state = tf.expand_dims(state, axis = 0)
        act_values = self.q_network.predict(state)[0]

        legal_moves_score = act_values[legal_actions]
        idx = np.argmax(legal_moves_score)

        return legal_actions[idx]

    def predict(self, state, env):
        legal_actions = env._legal_actions()

        # exploitation (invoke target network to make prediction)
        state = tf.expand_dims(state, axis = 0)
        act_values = self.target_network.predict(state)[0]

        legal_moves_score = act_values[legal_actions]
        idx = np.argmax(legal_moves_score)

        return legal_actions[idx]

    def replay(self, batch_size):
        minibatch = random.sample(self.experience_replay, batch_size)

        for state, action, reward, next_state, done in minibatch:
            state = tf.expand_dims(state, axis = 0)
            target = self.q_network.predict(state)

            if done:
                target[0][action] = reward
            else:
                next_state = tf.expand_dims(next_state, axis = 0)
                t = self.target_network.predict(next_state)[0][0]
                target[0][action] = reward + self.gamma * np.amax(t)

            self.q_network.fit(state, target, epochs = 1, verbose = 0)

        if self.epsilon_min < self.epsilon:
            self.epsilon *= self.epsilon_decay

    def save(self, path_dir = './_saved_models'):
        print('Saving models...')

        if not os.path.exists(path_dir):
            os.makedirs(path_dir)

        self.q_network.save(os.path.join(path_dir, 'q_network.h5'))
        self.target_network.save(os.path.join(path_dir, 'target_network.h5'))

    def load(self, path_dir = './_saved_models'):
        print('Loading models...')

        q_network_path = os.path.join(path_dir, 'q_network.h5')
        if not os.path.exists(q_network_path):
            raise ValueError('Error loading model: Q-network model cannot be found')

        target_network_path = os.path.join(path_dir, 'target_network.h5')
        if not os.path.exists(target_network_path):
            raise ValueError('Error loading model: Target network model cannot be found')

        self.q_network = tf.keras.models.load_model(q_network_path)
        self.target_network = tf.keras.models.load_model(target_network_path)

        self.epsilon = 0.00997


