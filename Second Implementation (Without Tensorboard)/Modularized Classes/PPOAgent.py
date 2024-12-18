# Libraries Used:
import torch
import random
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.backends.cudnn
from gymnasium.spaces import Discrete, Box
from torch.distributions import Categorical, Normal

# Functions:
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    # Orthogonal weight initialization, following best practices suggested in PPO and other works.
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def get_action_dim(action_space):
    '''
    Discrete and continuous spaces in Gymnasium have
    different methods, so to generalize
    the implementation, it's necessary to check the space and assign the appropriate method
    to each class in the Gymnasium API.
    '''
    if isinstance(action_space, Discrete):
        return action_space.n
    elif isinstance(action_space, Box):
        return action_space.shape[0]
    else:
        raise NotImplementedError(f"Action space of type {type(action_space)} not supported.")

  # Class:
class PPOAgent(nn.Module):
    # PPO Agent: Actor and Critic networks
    # As described in the paper, we use two networks sharing initial layers or separate:
    # Here we have an actor (for action logits) and a critic (for value), both MLPs.
    def __init__(self, observation_space, action_space, args):
        super().__init__()
        self.args = args
        obs_shape = observation_space.shape        
        n_actions = get_action_dim(action_space)
        self.action_space_type = type(action_space)

        # Critic network
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.prod(obs_shape), self.args.lineInput)),
            self.args.activation,
            layer_init(nn.Linear(self.args.lineInput, self.args.columnInput)),
            self.args.activation,
            layer_init(nn.Linear(self.args.columnInput, 1), std=1.0),
        )

        # Actor network
        if isinstance(action_space, Discrete):
            self.actor = nn.Sequential(
                layer_init(nn.Linear(np.prod(obs_shape), self.args.lineInput)),
                nn.Tanh(),
                layer_init(nn.Linear(self.args.lineInput, self.args.columnInput)),
                nn.Tanh(),
                layer_init(nn.Linear(self.args.columnInput, n_actions), std=0.01),
            )
        elif isinstance(action_space, Box):
            # For continuous spaces, the policy outputs the mean of actions. The variance can be a separate parameter.
            self.actor = nn.Sequential(
                layer_init(nn.Linear(np.prod(obs_shape), 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, n_actions), std=0.01),
            )
            # Log standard deviation as a trainable parameter.
            self.log_std = nn.Parameter(torch.zeros(n_actions))
        else:
            raise NotImplementedError(f"Action space of type {type(action_space)} not supported.")

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        # PPO actors typically produce a probability distribution over actions.
        logits_or_mean = self.actor(x)
        if self.action_space_type == Discrete:
            probs = Categorical(logits=logits_or_mean)  # Distribution for discrete actions.
            if action is None:
                action = probs.sample()  # Randomly sample an action from the current policy.
            return action, probs.log_prob(action), probs.entropy(), self.critic(x)
        elif self.action_space_type == Box:
            std = self.log_std.exp().expand_as(logits_or_mean)
            dist = Normal(logits_or_mean, std)  # Distribution for continuous actions.
            if action is None:
                action = dist.sample()  # Randomly sample an action from the current policy.
            # For multi-dimensional continuous actions, we sum the log-probabilities and entropies.
            log_prob = dist.log_prob(action).sum(axis=-1)
            entropy = dist.entropy().sum(axis=-1)
            return action, log_prob, entropy, self.critic(x)
        else:
            raise NotImplementedError(f"Action space of type {type(self.action_space_type)} not supported.")
