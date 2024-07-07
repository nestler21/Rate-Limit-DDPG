import numpy as np

class ReplayBuffer:
    def __init__(self, max_size, input_size, action_size):
        self.size = max_size
        self.counter = 0
        self.state_memory = np.zeros((self.size, input_size))
        self.new_state_memory = np.zeros((self.size, input_size))
        self.action_memory = np.zeros((self.size, action_size))
        self.reward_memory = np.zeros(self.size)
    
    def store_transisition(self, state, action, reward, new_state):
        index = self.counter % self.size # index is reset if buffer size is reached

        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        
        self.counter += 1

    def sample_batch(self, batch_size):
        max_mem = min(self.counter, self.size)

        batch_indices = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch_indices]
        states_ = self.new_state_memory[batch_indices]
        actions = self.action_memory[batch_indices]
        rewards = self.reward_memory[batch_indices]

        return states, actions, rewards, states_
