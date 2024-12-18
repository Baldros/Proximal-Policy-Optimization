# Libraries Used:
import os
import json
import time
import torch
import random
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.optim as optim
import torch.backends.cudnn
from gymnasium.spaces import Discrete, Box
from torch.distributions import Categorical, Normal

# Functions ##################################################################################################

def make_env(env_id, idx):
    # Helper function to create the Gym environment.
    # We use RecordEpisodeStatistics to obtain rewards and episode lengths.
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk

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

# Classes ########################################################################################################

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

class PPOTrainer:
    def __init__(self, agent, envs, args):
        self.agent = agent
        self.envs = envs
        self.args = args
        # Define the device (GPU or CPU) based on availability and preference.
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.args.cuda else "cpu")
        self.agent.to(self.device)

        # Adam optimizer for updating the agent's parameters.
        # PPO uses an optimizer like Adam to update the actor-critic network.
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.args.learning_rate, eps=1e-5)

        # Determines whether the action space is continuous or discrete.
        self.is_discrete = isinstance(self.envs.single_action_space, Discrete)
        self.is_continuous = isinstance(self.envs.single_action_space, Box)

        # Creation of tensors to store rollout data.
        # Here we store observations, actions, log-probs, rewards, and values
        # in buffers to compute advantages (GAE) and returns at the end.
        self.obs = torch.zeros((self.args.num_steps, self.args.num_envs) + envs.single_observation_space.shape).to(self.device)
        if self.is_discrete:
            self.actions = torch.zeros((self.args.num_steps, self.args.num_envs), dtype=torch.long).to(self.device)
        else:
            self.actions = torch.zeros((self.args.num_steps, self.args.num_envs) + envs.single_action_space.shape).to(self.device)
        self.logprobs = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        self.rewards = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        self.dones = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        self.values = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)

        # Dictionary to store performance and diagnostic metrics.
        self.metrics = {
            "episodic_return": [],
            "episodic_length": [],
            "episodic_time": [],
            "value_loss": [],
            "policy_loss": [],
            "entropy_loss": [],
            "old_approx_kl": [],
            "approx_kl": [],
            "clipfrac": [],
            "explained_variance": [],
            "learning_rate": [],
            "SPS": []
        }

        self.global_step = 0
        self.start_time = time.time()

        # Initial environment reset.
        # Gets the first observation from the environment.
        next_obs, _ = envs.reset(seed=None)
        self.next_obs = torch.Tensor(next_obs).to(self.device)
        self.next_done = torch.zeros(self.args.num_envs).to(self.device)

    def anneal_lr(self, iteration):
        # Adjustment of the learning rate (annealing).
        # PPO suggests gradually reducing the learning rate over training.
        # Here, lrnow is the current learning rate, calculated as linearly decreasing
        # from the initial learning_rate. This helps in training stability.
        if self.args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / self.args.num_iterations
            lrnow = frac * self.args.learning_rate
            self.optimizer.param_groups[0]["lr"] = lrnow

    def compute_advantages(self, next_value):
        # Initializes a tensor to store advantages (GAE).
        # Advantages measure how good an action was compared to the expected value.
        advantages = torch.zeros_like(self.rewards).to(self.device)
        
        # Initializes the variable lastgaelam, which stores the GAE for the next step.
        lastgaelam = 0
    
        # Iterates over steps in reverse order (from end to start of rollout).
        # This is necessary because the advantage calculation depends on the next value.
        for t in reversed(range(self.args.num_steps)):

            # If we are at the last step of the trajectory (rollout).
            if t == self.args.num_steps - 1:
                # If the episode ended in the next state, nextnonterminal will be 0.
                nextnonterminal = 1.0 - self.next_done  # 1 if not ended, 0 if ended.
                # Estimated value of the next state.
                nextvalues = next_value
            else:
                # Otherwise, we use the values from the next step in the trajectory.
                nextnonterminal = 1.0 - self.dones[t + 1]  # 1 if not ended, 0 if ended.
                nextvalues = self.values[t + 1]  # Value of the next state already stored in the rollout.

            # 1. Calculation of **delta**, which measures the immediate advantage.
            '''
            delta captures the **temporal difference** (TD Error), that is:
            - The difference between the received reward + discounted future value and the current value.
            - If delta > 0: the action was better than the estimated value.
            - If delta < 0: the action was worse than the estimated value.
            '''
            delta = self.rewards[t] + self.args.gamma * nextvalues * nextnonterminal - self.values[t]
            

            # 2. Calculation of the accumulated advantage (GAE).
            # GAE combines the current delta with the accumulated future advantage, amortized by the lambda factor.
            advantages[t] = lastgaelam = delta + self.args.gamma * self.args.gae_lambda * nextnonterminal * lastgaelam
            '''
            Interpretation:
            - delta contributes to the current advantage.
            - lastgaelam accumulates the future advantages, discounted by gamma and amortized by lambda.
            - This creates a "smoothed" estimate of the advantages, reducing variance.
            '''
        
        # 3. Calculation of returns.
        # The return is simply the sum of the advantage (GAE) and the state value.
        # Return = Advantage (GAE) + Estimated value.
        returns = advantages + self.values
        
        return advantages, returns


    def train_loop(self):
        # Main training loop of PPO.
        # Each iteration corresponds to collecting 'num_steps' of experiences in each of the 'num_envs' environments
        # and then performing parameter updates.
        for iteration in range(1, self.args.num_iterations + 1):
            # Adjusts the learning rate according to the iteration (annealing).
            self.anneal_lr(iteration)

            # Data collection (rollout).
            for step in range(self.args.num_steps):
                self.global_step += self.args.num_envs
                self.obs[step] = self.next_obs
                self.dones[step] = self.next_done

                # Obtains action and value from the current policy (actor-critic network).
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(self.next_obs)
                    self.values[step] = value.flatten()

                # Stores the action and log-probability of the current action.
                if self.is_discrete:
                    self.actions[step] = action
                else:
                    self.actions[step] = action
                self.logprobs[step] = logprob

                # Advances the environment with the chosen action.
                # Receives new observation, reward, and termination flags (terminations, truncations).
                next_obs, reward, terminations, truncations, infos = self.envs.step(action.cpu().numpy())
                self.next_done = np.logical_or(terminations, truncations)
                self.rewards[step] = torch.tensor(reward).to(self.device).view(-1)
                self.next_obs = torch.Tensor(next_obs).to(self.device)
                self.next_done = torch.Tensor(self.next_done).to(self.device)

                # Checking for episode end and logging metrics.
                if "episode" in infos:
                    for idx, episode_finished in enumerate(infos["_episode"]):
                        if episode_finished:
                            reward_sum = infos["episode"]["r"][idx]
                            episode_length = infos["episode"]["l"][idx]
                            episode_time = infos["episode"].get("t", [None])[idx]
                            
                            if self.args.printOn:
                                print(f"global_step={self.global_step}, episodic_return={reward_sum}")

                            # Stores episode metrics.
                            self.metrics["episodic_return"].append((self.global_step, reward_sum))
                            self.metrics["episodic_length"].append((self.global_step, episode_length))
                            if episode_time is not None:
                                self.metrics["episodic_time"].append((self.global_step, episode_time))

            # Calculation of bootstrap value at the next state (for GAE).
            with torch.no_grad():
                next_value = self.agent.get_value(self.next_obs).reshape(1, -1)
            advantages, returns = self.compute_advantages(next_value)

            # "Flattening" the batches to prepare for minibatch updates.
            b_obs = self.obs.reshape((-1,) + self.envs.single_observation_space.shape)
            b_logprobs = self.logprobs.reshape(-1)
            if self.is_discrete:
                b_actions = self.actions.reshape(-1)
            else:
                b_actions = self.actions.reshape(-1, self.envs.single_action_space.shape[0])
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = self.values.reshape(-1)

            clipfracs = []
            # Policy and value function parameter updates.
            # We split the batch into minibatches and perform multiple epochs (update_epochs)
            # as suggested in PPO.
            for epoch in range(self.args.update_epochs):
                inds = np.arange(self.args.batch_size)
                np.random.shuffle(inds)
                for start in range(0, self.args.batch_size, self.args.minibatch_size):
                    end = start + self.args.minibatch_size
                    mb_inds = inds[start:end]

                    # Recalculates logprob and value for the minibatch.
                    if self.is_discrete:
                        mb_actions = b_actions[mb_inds].long()
                    else:
                        mb_actions = b_actions[mb_inds]
                    
                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                        b_obs[mb_inds], mb_actions if self.is_discrete else mb_actions
                    )
                    
                    # For continuous actions, logprob is summed over dimensions.
                    if self.is_continuous:
                        newlogprob = newlogprob
                        entropy = entropy
                    else:
                        newlogprob = newlogprob
                        entropy = entropy

                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    # Approximate KL for diagnostics (not directly used in the standard loss).
                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()]

                    # Normalization of advantage, as suggested, for stability.
                    mb_advantages = b_advantages[mb_inds]
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Calculation of policy loss (Policy Loss).
                    # PPO: we maximize L^CLIP, which uses the min between ratio*advantage and ratio clamped.
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Calculation of value loss (Value Loss).
                    # Clipping the value, as per the PPO approach to reduce instabilities.
                    newvalue = newvalue.view(-1)
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.args.clip_coef,
                        self.args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                    # Entropy to encourage exploration.
                    entropy_loss = entropy.mean()

                    # Total loss (Policy Loss + c1 * Value Loss + c2 * Entropy Loss).
                    # As described in the PPO paper, the total loss combines these components.
                    loss = pg_loss - self.args.ent_coef * entropy_loss + v_loss * self.args.vf_coef

                    # Updates network parameters.
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()

            # Calculation of "explained_variance" to assess the quality of the value function.
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # Logging metrics.
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.metrics["learning_rate"].append((self.global_step, current_lr))
            self.metrics["value_loss"].append((self.global_step, v_loss.item()))
            self.metrics["policy_loss"].append((self.global_step, pg_loss.item()))
            self.metrics["entropy_loss"].append((self.global_step, entropy_loss.item()))
            self.metrics["old_approx_kl"].append((self.global_step, old_approx_kl.item()))
            self.metrics["approx_kl"].append((self.global_step, approx_kl.item()))
            self.metrics["clipfrac"].append((self.global_step, np.mean(clipfracs)))
            self.metrics["explained_variance"].append((self.global_step, explained_var))

            # Steps per second (SPS) to assess training speed.
            sps = int(self.global_step / (time.time() - self.start_time))
            self.metrics["SPS"].append((self.global_step, sps))
            if self.args.printOn:
                print("SPS:", sps)

        self.envs.close()

    def get_metrics(self):
        return self.metrics
