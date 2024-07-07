import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.regularizers import l2


class CriticNetwork(keras.Model):
    def __init__(self, input_size, action_size, fc1_dims=400, fc2_dims=300, l2_value=0.01, name='critic', chkpt_dir='tmp/ddpg'):
        super(CriticNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, name+'.weights.h5')

        # create weight initializers networks as described in the paper (Lillicrap, 2016, p.11)
        fc1_init = RandomUniform(minval=-1/np.sqrt(input_size), maxval=1/np.sqrt(input_size))
        fc2_init = RandomUniform(minval=-1/np.sqrt(fc1_dims+action_size), maxval=1/np.sqrt(fc1_dims+action_size))
        q_init = RandomUniform(minval=-0.003, maxval=0.003)

        # kernel_regularizer for L2 weight decay
        self.fc1 = Dense(fc1_dims, activation='relu', kernel_regularizer=l2(l2_value), kernel_initializer=fc1_init)
        self.fc2 = Dense(fc2_dims, activation='relu', kernel_regularizer=l2(l2_value), kernel_initializer=fc2_init)
        self.q = Dense(1, activation=None, kernel_regularizer=l2(l2_value), kernel_initializer=q_init)

    def call(self, state, action):
        state_value = self.fc1(state)
        action_value = self.fc2(tf.concat([state_value, action], axis=1)) # actions are not added until the second hidden layer
        q = self.q(action_value)
        return q


class ActorNetwork(keras.Model):
    def __init__(self, input_size, fc1_dims=400, fc2_dims=300, action_size=1, name='actor', chkpt_dir='tmp/ddpg'):
        super(ActorNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, name+'.weights.h5')

        # create weight initializers networks as described in the paper (Lillicrap, 2016, p.11)
        fc1_init = RandomUniform(minval=-1/np.sqrt(input_size), maxval=1/np.sqrt(input_size))
        fc2_init = RandomUniform(minval=-1/np.sqrt(fc1_dims), maxval=1/np.sqrt(fc1_dims))
        q_init = RandomUniform(minval=-0.003, maxval=0.003)

        # kernel_regularizer for L2 weight decay
        self.fc1 = Dense(fc1_dims, activation='relu', kernel_initializer=fc1_init)
        self.fc2 = Dense(fc2_dims, activation='relu', kernel_initializer=fc2_init)
        self.mu = Dense(action_size, activation='tanh', kernel_initializer=q_init)

    def call(self, state):
        prob = self.fc1(state)
        prob = self.fc2(prob)
        mu = self.mu(prob)
        return mu