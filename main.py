import numpy as np
import json
import matplotlib.pyplot as plt
from Environment import Simulation_Manager, Simulation_Wrapper, Environment_Manager, Server
from Agent import Agent


def plot_reward(rewards, filename, episodes, steps, chunk_size, axhline):
    figure_file = f'plots/Runde 13/{filename}.png'
    
    rewards = np.array(rewards)
    num_full_chunks = len(rewards) // chunk_size
    chunks = np.array_split(rewards[:num_full_chunks * chunk_size], num_full_chunks)
    rewards = [np.mean(chunk) for chunk in chunks]

    x = [(i+1)*chunk_size for i in range(len(rewards))]

    plt.plot(x, rewards, label='Reward')
    plt.axhline(y=axhline, color='r', linestyle='--', label='Optimum Reward')
    for i in range(episodes):
        plt.axvline(x=(i+1)*steps, color='g', linestyle='--')
    plt.xlabel('Time Step')
    plt.ylabel('Reward')
    plt.title('Reward over Time')
    plt.legend()

    # plt.show()
    plt.savefig(figure_file)
    plt.clf()

def plot_actions(actions, filename, episodes, steps, axhline):
    figure_file = f'plots/Runde 13/{filename}.png'
    
    x = [(i+1) for i in range(len(actions))]

    plt.plot(x, actions, label='Delay')
    plt.axhline(y=axhline, color='r', linestyle='--', label='Optimum Delay')
    for i in range(episodes):
        plt.axvline(x=(i+1)*steps, color='g', linestyle='--')
    plt.xlabel('Time Step')
    plt.ylabel('Delay')
    plt.title('Delay over Time')
    plt.legend()
    
    # plt.show()
    plt.savefig(figure_file)
    plt.clf()

def save_data(rewards, actions, actions_wn, filename, episodes, steps, reward_axhline, action_axhline):
    file_path = f'data/Runde 13/{filename}.json'
    data = {
        "rewards": rewards,
        "actions": actions,
        "actions_wn": actions_wn,
        "episodes": episodes,
        "steps": steps,
        "reward_axhline": reward_axhline,
        "action_axhline": action_axhline
    }
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file)


# main method for executing the agent
def run_agent(args, filename):
    # environment config

    max_delay = 1
    state_scope = 10
    window = 2
    allowed_requests = 10

    # environment initialization

    # env = Environment_Manager(Server("http://127.0.0.1:5000/rate-limit"), max_delay, state_scope)
    env = Simulation_Manager(Simulation_Wrapper(window, allowed_requests), max_delay, state_scope)

    # agent initialization

    agent = Agent(input_size=state_scope,
                  alpha_a=args["alpha_a"],
                  alpha_c=args["alpha_c"],
                  l2_value=args["l2_value"],
                  gamma=args["gamma"],
                  buffer_size=args["buffer_size"],
                  tau=args["tau"],
                  fc1=args["fc1"],
                  fc2=args["fc2"],
                  batch_size=args["batch_size"],
                  noise=args["noise"])
    
    # metrics lists
    rewards = []
    actions = []
    actions_wn = []

    # run time config
    episodes = 5
    steps = 200
    evaluate = False
    
    for i in range(episodes):
        observation = env.get_initial_state()
        for j in range(steps):
            # select action
            action, action_without_noise = agent.choose_action(observation, evaluate)
            action = (action.numpy().item() + 1)*max_delay/2
            action_without_noise = (action_without_noise.numpy().item() + 1)*max_delay/2
            # execute action
            new_observation, reward = env.act(action)
            if not evaluate:
                # store transition
                agent.remember(observation, action, reward, new_observation)
                # sample minibatch and learn
                agent.learn()
            
            # print step infor
            print(str(i+1).zfill(len(str(episodes))) + "-" + str(j+1).zfill(len(str(steps))) + f" ({filename})")

            observation = new_observation

            # add current run to the metrics            
            rewards.append(reward)
            actions.append(action)
            actions_wn.append(action_without_noise)
    
    # generate the plots and save the data
    chunk_size = 10 # rewards are aggregated (average) into chunks
    filename_mod = filename + "_rewards"
    plot_reward(rewards, filename_mod, episodes, steps, chunk_size, axhline=env.calculate_reward(True, window/allowed_requests))
    filename_mod = filename + "_actions"
    plot_actions(actions, filename_mod, episodes, steps, axhline=window/allowed_requests)
    filename_mod = filename + "_actions_no_noise"
    plot_actions(actions_wn, filename_mod, episodes, steps, axhline=window/allowed_requests)
    save_data(rewards, actions, actions_wn, filename, episodes, steps, env.calculate_reward(True, window/allowed_requests), window/allowed_requests)


# method for hyperparameter tuning
def tune_hyperparameters(standard_args, prefix):
    
    arg_list={
        "alpha_a": [0.2, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001],
        "alpha_c":[0.2, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001],
        "l2_value":[0.2, 0.1, 0.01, 0.001, 0.0001],
        "gamma":[0.8, 0.9, 0.99, 0.999, 0.9999],
        "tau":[0.2, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001],
        "fc1":[100, 200, 300, 400, 500],
        "fc2":[100, 200, 300, 400, 500],
        "batch_size":[8, 16, 32, 64, 128],
        "noise":[0.3, 0.2, 0.1, 0.01, 0.001]
    }

    for arg in arg_list.keys():
        for value in arg_list[arg]:
            curr_args = standard_args.copy()
            curr_args[arg] = value
            filename = f"{prefix}_{arg}_{value}"
            run_agent(args=curr_args, filename=filename)


if __name__ == '__main__':
    
    # standard values for the Hyperparamters
    standard_args={
        "alpha_a":0.0001,
        "alpha_c":0.2,
        "l2_value": 0.2,
        "gamma":0.99,
        "buffer_size":1000000,
        "tau":0.001,
        "fc1":400,
        "fc2":300,
        "batch_size":64,
        "noise":0.1
    }

    # solo run with standard values
    
    run_agent(args=standard_args, filename="1_simulation_tuned")
    
    # tuning the hyperparamters

    # tune_hyperparameters(standard_args, prefix="1")
    # tune_hyperparameters(standard_args, prefix="2")
    # tune_hyperparameters(standard_args, prefix="3")
    # tune_hyperparameters(standard_args, prefix="4")
    # tune_hyperparameters(standard_args, prefix="5")
    # tune_hyperparameters(standard_args, prefix="6")



# Hyperparameters according to the paper (Lillicrap, 2016, p.11)

# standard_args={
#     "alpha_a":0.0001,
#     "alpha_c":0.001,
#     "l2_value": 0.01,
#     "gamma":0.99,
#     "buffer_size":1000000,
#     "tau":0.001,
#     "fc1":400,
#     "fc2":300,
#     "batch_size":64,
#     "noise":0.1
# }