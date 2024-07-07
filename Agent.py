import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from ReplayBuffer import ReplayBuffer
from Networks import ActorNetwork, CriticNetwork


class Agent:
    def __init__(self, input_size, alpha_a=0.0001, alpha_c=0.001, l2_value=0.01,
                 gamma=0.99, action_size=1, buffer_size=1000000, tau=0.001,
                 fc1=400, fc2=300, batch_size=64, noise=0.1): # parameters according to the paper (Lillicrap, 2016, p.11)
        
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.action_size = action_size
        self.noise = noise

        # create Replay Buffer
        self.replay_buffer = ReplayBuffer(buffer_size, input_size, action_size)

        # create networks
        self.actor = ActorNetwork(fc1_dims=fc1, fc2_dims=fc2, name='actor', action_size=action_size, input_size=input_size)
        self.critic = CriticNetwork(fc1_dims=fc1, fc2_dims=fc2, l2_value=l2_value, name='critic', action_size=action_size, input_size=input_size)
        self.target_actor = ActorNetwork(fc1_dims=fc1, fc2_dims=fc2, name='target_actor', action_size=action_size, input_size=input_size)
        self.target_critic = CriticNetwork(fc1_dims=fc1, fc2_dims=fc2, l2_value=l2_value, name='target_critic', action_size=action_size, input_size=input_size)

        # compile networks
        self.actor.compile(optimizer=Adam(learning_rate=alpha_a))
        self.critic.compile(optimizer=Adam(learning_rate=alpha_c))
        self.target_actor.compile(optimizer=Adam(learning_rate=alpha_a)) # learning rate has to be provided for compilation even if the target networks don't use optimizers
        self.target_critic.compile(optimizer=Adam(learning_rate=alpha_c)) # learning rate has to be provided for compilation even if the target networks don't use optimizers
        
        # initializing network weights by sampling random values
        tmp_action = self.actor(tf.random.normal((1, input_size)))
        self.target_actor(tf.random.normal((1, input_size)))
        self.critic(tf.random.normal((1, input_size)), tmp_action)
        self.target_critic(tf.random.normal((1, input_size)), tmp_action)
        # clone weights to target networks
        self.target_actor.set_weights(self.actor.weights)
        self.target_critic.set_weights(self.critic.weights)

    def choose_action(self, observation, evaluate=False):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        actions = self.actor(state)
        action_without_noise = actions[0]
        if not evaluate:
            actions += tf.random.normal(shape=[self.action_size], mean=0.0, stddev=self.noise).numpy().item()
            actions = tf.clip_by_value(actions, -1, 1)
        return actions[0], action_without_noise
    
    def remember(self, state, action, reward, new_state):
        self.replay_buffer.store_transisition(state, action, reward, new_state)

    def learn(self):
        if self.replay_buffer.counter < self.batch_size:
            return
        
        # sample mini batch
        state, action, reward, new_state = self.replay_buffer.sample_batch(self.batch_size)
        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        action = tf.convert_to_tensor(action, dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        
        # Gradient Decent for Critic Network
        with tf.GradientTape() as tape:
            critic_value = tf.squeeze(self.critic(states, action), 1) # = Q(s, a)
            target_actions_ = self.target_actor(states_) # = a' = µ'(s')
            critic_value_ = tf.squeeze(self.target_critic(states_, target_actions_), 1) # = Q'(s', a')
            target = reward + self.gamma*critic_value_ # = yi
            critic_loss = keras.losses.MSE(target, critic_value) # MSE(yi, Q(s, a))
        
        critic_network_gradient = tape.gradient(critic_loss, self.critic.trainable_variables) # Calculate Gradient
        self.critic.optimizer.apply_gradients(zip(critic_network_gradient, self.critic.trainable_variables)) # Update weights

        # Gradient Accent for Actor Network
        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states) # = µ(s)
            actor_loss = -self.critic(states, new_policy_actions) # = Q(s, µ(s))
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss, self.actor.trainable_variables) # chain rule is being applied internaly. loss based on critic, who gets the actions from the actor
        self.actor.optimizer.apply_gradients(zip(actor_network_gradient, self.actor.trainable_variables)) # therefore the loss can be derived by the weights of the actor

        self.soft_target_updates()

    def soft_target_updates(self):        
        # update actor target
        weights = []
        for i, actor_weight in enumerate(self.actor.weights):
            weights.append(actor_weight*self.tau + self.target_actor.weights[i]*(1-self.tau))
        self.target_actor.set_weights(weights)

        # update critic target
        weights = []
        for i, critic_weight in enumerate(self.critic.weights):
            weights.append(critic_weight*self.tau + self.target_critic.weights[i]*(1-self.tau))
        self.target_critic.set_weights(weights)
