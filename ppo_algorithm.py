import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PPO(nn.Module):
    def __init__(self, envs, hidden_size=64, gamma=0.99, lam=0.95, clip_epsilon=0.2, lr=3e-4, entropy_coef=0.01):
        super(PPO, self).__init__()

        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef

        obs_dim = np.array(envs.single_observation_space.shape).prod()
        action_dim = envs.single_action_space.n

        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, action_dim), std=0.01)
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0)
        )

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

    def forward(self, obs):
        logits = self.actor(obs)
        value = self.critic(obs)
        return logits, value

    def act(self, obs):
        logits, _ = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def compute_returns(self, rewards, values, next_value, dones):
        returns = torch.zeros_like(rewards)
        last_gae_lam = 0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0
            delta = rewards[t] + self.gamma * next_value - values[t]
            last_gae_lam = delta + self.gamma * self.lam * last_gae_lam
            returns[t] = last_gae_lam + values[t]
            next_value = values[t]
        return returns

    def update(self, observations, actions, log_probs_old, returns, advantages):
        logits, values = self.forward(observations)
        dist = Categorical(logits=logits)
        log_probs_new = dist.log_prob(actions)

        ratio = torch.exp(log_probs_new - log_probs_old)

        surrogate_loss = torch.min(
            ratio * advantages,
            torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        )
        actor_loss = -torch.mean(surrogate_loss)

        critic_loss = F.mse_loss(values.squeeze(), returns)

        entropy_loss = torch.mean(dist.entropy()) * self.entropy_coef

        loss = actor_loss + critic_loss - entropy_loss

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

    def train(self, envs, total_timesteps=100000, num_steps=128, batch_size=64, epochs=4):
        obs = torch.tensor(envs.reset(), dtype=torch.float32)

        for step in range(total_timesteps // num_steps):
            observations, actions, rewards, log_probs, dones, values = [], [], [], [], [], []

            for _ in range(num_steps):
                logits, value = self.forward(obs)
                dist = Categorical(logits=logits)
                action = dist.sample()

                observations.append(obs)
                actions.append(action)
                log_probs.append(dist.log_prob(action))
                values.append(value)

                next_obs, reward, done, _ = envs.step(action.cpu().numpy())
                obs = torch.tensor(next_obs, dtype=torch.float32)

                rewards.append(torch.tensor(reward, dtype=torch.float32))
                dones.append(torch.tensor(done, dtype=torch.float32))

            observations = torch.stack(observations)
            actions = torch.stack(actions)
            log_probs = torch.stack(log_probs)
            values = torch.stack(values).squeeze()
            rewards = torch.stack(rewards).squeeze()
            dones = torch.stack(dones).squeeze()

            next_value = self.critic(obs).detach()
            returns = self.compute_returns(rewards, values, next_value, dones)
            advantages = returns - values

            for _ in range(epochs):
                for i in range(0, num_steps, batch_size):
                    batch_obs = observations[i:i + batch_size]
                    batch_actions = actions[i:i + batch_size]
                    batch_log_probs = log_probs[i:i + batch_size]
                    batch_returns = returns[i:i + batch_size]
                    batch_advantages = advantages[i:i + batch_size]

                    self.update(batch_obs, batch_actions, batch_log_probs, batch_returns, batch_advantages)
