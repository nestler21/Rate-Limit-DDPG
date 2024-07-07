import requests
from datetime import datetime, timedelta
import numpy as np


# Class for the Environment Wrapper

class Environment_Wrapper:

    def __init__(self) -> None:
        pass

    def request(self, delay):
        pass


# Class for the Environment Wrapper for the Test-Server

class Server(Environment_Wrapper):

    def __init__(self, target) -> None:
        super().__init__()
        self.target = target
        self.last_request = None

    def request(self, delay):
        delay = timedelta(seconds=delay)
        if self.last_request:
            while datetime.now() - self.last_request < delay:
                pass # wait until the delay is over, this is more accurate than time.sleep()
        
        self.last_request = datetime.now()
        response = requests.get(self.target)
        if response.status_code == 200:
            return True
        else:
            return False


# Class for the Environment Wrapper for the Simulation

class Simulation_Wrapper(Environment_Wrapper):

    def __init__(self, window, allowed_requests) -> None:
        super().__init__()
        self.window = window
        self.allowed_requests = allowed_requests
        self.requests = []

    def request(self, delay):
        self.requests = [age + delay for age in self.requests] # update age of the requests
        self.requests = [age for age in self.requests if age <= self.window] # remove requests outside of the window
        self.requests.append(0.0) # add recent request
        if len(self.requests) <= self.allowed_requests:
            return True
        else:
            return False
        

# Class for the Environment Manager

class Environment_Manager:
    
    def __init__(self, env_wrapper, max_delay, state_scope) -> None:
        self.env_wrapper = env_wrapper
        self.max_delay = max_delay
        self.state_scope = state_scope
        self.requests = []
    
    def act(self, delay):
        self.requests.append(datetime.now())
        success = self.env_wrapper.request(delay)           # send request to environment
        if len(self.requests) > self.state_scope:
            self.requests = self.requests[-self.state_scope:]   # truncate requests to state scope
        return self.get_state(), self.calculate_reward(success, delay)

    def calculate_reward(self, success, delay):
        if not success:
            return -1                                       # Reward function
        else:
            if delay == 0: delay = 0.000000001              # prevents log(0)
            return np.log(self.max_delay) - np.log(delay)   # Reward function

    def get_state(self):
        state = [(datetime.now() - timestamp).total_seconds() for timestamp in self.requests] # calculate age of requests
        return state[::-1] + [0] * (self.state_scope - len(state)) # sort and fill missing values with 0
    
    def get_initial_state(self):
        return np.zeros(self.state_scope)


# Class for the Environment Manager for the Simulation

class Simulation_Manager(Environment_Manager):
    def __init__(self, env_wrapper, max_delay, state_scope) -> None:
        super().__init__(env_wrapper, max_delay, state_scope)

    def act(self, delay):
        success = self.env_wrapper.request(delay)
        self.requests = [age + delay for age in self.requests]  # update age of the requests
        self.requests.append(0) # append recent request
        if len(self.requests) > self.state_scope:
            self.requests = self.requests[-self.state_scope:]   # truncate requests to state scope
        return self.get_state(), self.calculate_reward(success, delay)
    
    def get_state(self):
        return self.requests[::-1] + [0] * (self.state_scope - len(self.requests))