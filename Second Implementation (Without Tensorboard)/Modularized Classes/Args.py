# Libraries Used:
import os
import json
import gymnasium as gym

# Function:
def make_env(env_id, idx):
    # Helper function to create the Gym environment.
    # We use RecordEpisodeStatistics to obtain rewards and episode lengths.
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk

# Class:
class Args:
    """
    Class to load and store configuration parameters from a JSON file.
    """
    def __init__(self, json_path):
        # Reads the JSON and stores the values in the dictionary.
        with open(json_path, "r") as f:
            config = json.load(f)
        
        # Sets the attributes from the JSON configurations with default values.
        self.gym_task = config.get("gym_task","CartPole-v1")
        self.num_steps = config.get("num_steps", 5)
        self.num_envs = config.get("num_envs", 1)
        self.learning_rate = config.get("learning_rate", 3e-4)
        self.gamma = config.get("gamma", 0.99)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.update_epochs = config.get("update_epochs", 4)
        self.batch_size = config.get("batch_size", 64)
        self.minibatch_size = config.get("minibatch_size", 32)
        self.clip_coef = config.get("clip_coef", 0.2)
        self.ent_coef = config.get("ent_coef", 0.01)
        self.vf_coef = config.get("vf_coef", 0.5)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        self.anneal_lr = config.get("anneal_lr", True)
        self.cuda = config.get("cuda", False)
        self.exp_name = config.get("exp_name", "ppo_cartpole")
        self.num_iterations = config.get("num_iterations", 1000)
        self.printOn = config.get("printOn", True)
        self.lineInput = config.get("lineInput", 64)
        self.columnInput = config.get("columnInput", 64)
        
        # Mapping of strings to activation functions.
        activation_map = {
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(),
            # Add more activations as needed.
        }
        
        activation_str = config.get("activation", "tanh").lower()
        self.activation = activation_map.get(activation_str, nn.Tanh())
