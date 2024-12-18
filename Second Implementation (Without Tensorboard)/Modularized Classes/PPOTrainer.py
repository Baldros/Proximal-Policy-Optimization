# Libraries Used:
import time
import torch
import random
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.optim as optim
import torch.backends.cudnn
from torch.distributions import Categorical, Normal

# Class:
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
