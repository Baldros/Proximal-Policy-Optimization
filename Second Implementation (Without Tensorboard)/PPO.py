import os
import time
import torch
import random
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.optim as optim
import torch.backends.cudnn
from torch.distributions.categorical import Categorical

class PPOTrainer:
    def __init__(self, agent, envs, args):
        self.agent = agent
        self.envs = envs
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.args.cuda else "cpu")
        self.agent.to(self.device)

        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.args.learning_rate, eps=1e-5)

        # Buffers
        self.obs = torch.zeros((self.args.num_steps, self.args.num_envs) + envs.single_observation_space.shape).to(self.device)
        self.actions = torch.zeros((self.args.num_steps, self.args.num_envs) + envs.single_action_space.shape).to(self.device)
        self.logprobs = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        self.rewards = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        self.dones = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        self.values = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)

        # Dicionário para armazenar métricas
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

        # Estado inicial do ambiente
        next_obs, _ = envs.reset(seed=None)
        self.next_obs = torch.Tensor(next_obs).to(self.device)
        self.next_done = torch.zeros(self.args.num_envs).to(self.device)

    def anneal_lr(self, iteration):
        if self.args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / self.args.num_iterations
            lrnow = frac * self.args.learning_rate
            self.optimizer.param_groups[0]["lr"] = lrnow

    def compute_advantages(self, next_value):
        advantages = torch.zeros_like(self.rewards).to(self.device)
        lastgaelam = 0
        for t in reversed(range(self.args.num_steps)):
            if t == self.args.num_steps - 1:
                nextnonterminal = 1.0 - self.next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - self.dones[t + 1]
                nextvalues = self.values[t + 1]
            delta = self.rewards[t] + self.args.gamma * nextvalues * nextnonterminal - self.values[t]
            advantages[t] = lastgaelam = delta + self.args.gamma * self.args.gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + self.values
        return advantages, returns

    def train_loop(self):
        for iteration in range(1, self.args.num_iterations + 1):
            self.anneal_lr(iteration)

            # Coleta de dados
            for step in range(self.args.num_steps):
                self.global_step += self.args.num_envs
                self.obs[step] = self.next_obs
                self.dones[step] = self.next_done

                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(self.next_obs)
                    self.values[step] = value.flatten()

                self.actions[step] = action
                self.logprobs[step] = logprob

                next_obs, reward, terminations, truncations, infos = self.envs.step(action.cpu().numpy())
                self.next_done = np.logical_or(terminations, truncations)
                self.rewards[step] = torch.tensor(reward).to(self.device).view(-1)
                self.next_obs = torch.Tensor(next_obs).to(self.device)
                self.next_done = torch.Tensor(self.next_done).to(self.device)

                # Verifica se há informações de episódio no `infos`
                if "episode" in infos:
                    for idx, episode_finished in enumerate(infos["_episode"]):
                        if episode_finished:
                            reward_sum = infos["episode"]["r"][idx]
                            episode_length = infos["episode"]["l"][idx]
                            episode_time = infos["episode"].get("t", [None])[idx]

                            print(f"global_step={self.global_step}, episodic_return={reward_sum}")

                            # Armazena métricas no dicionário
                            self.metrics["episodic_return"].append((self.global_step, reward_sum))
                            self.metrics["episodic_length"].append((self.global_step, episode_length))
                            if episode_time is not None:
                                self.metrics["episodic_time"].append((self.global_step, episode_time))

            # Atualização dos parâmetros
            with torch.no_grad():
                next_value = self.agent.get_value(self.next_obs).reshape(1, -1)
            advantages, returns = self.compute_advantages(next_value)

            b_obs = self.obs.reshape((-1,) + self.envs.single_observation_space.shape)
            b_logprobs = self.logprobs.reshape(-1)
            b_actions = self.actions.reshape((-1,) + self.envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = self.values.reshape(-1)

            clipfracs = []
            for epoch in range(self.args.update_epochs):
                inds = np.arange(self.args.batch_size)
                np.random.shuffle(inds)
                for start in range(0, self.args.batch_size, self.args.minibatch_size):
                    end = start + self.args.minibatch_size
                    mb_inds = inds[start:end]

                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                        b_obs[mb_inds], b_actions.long()[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if True:  # Normalização da vantagem assumida como sempre ligada aqui
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if True: # Mantemos clipping do valor
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.args.clip_coef,
                            self.args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.args.ent_coef * entropy_loss + v_loss * self.args.vf_coef

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()

            # Métricas
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.metrics["learning_rate"].append((self.global_step, current_lr))
            self.metrics["value_loss"].append((self.global_step, v_loss.item()))
            self.metrics["policy_loss"].append((self.global_step, pg_loss.item()))
            self.metrics["entropy_loss"].append((self.global_step, entropy_loss.item()))
            self.metrics["old_approx_kl"].append((self.global_step, old_approx_kl.item()))
            self.metrics["approx_kl"].append((self.global_step, approx_kl.item()))
            self.metrics["clipfrac"].append((self.global_step, np.mean(clipfracs)))
            self.metrics["explained_variance"].append((self.global_step, explained_var))

            sps = int(self.global_step / (time.time() - self.start_time))
            self.metrics["SPS"].append((self.global_step, sps))
            print("SPS:", sps)

        self.envs.close()

    def get_metrics(self):
        return self.metrics

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PPOAgent(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        obs_shape = observation_space.shape
        n_actions = action_space.n

        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.prod(obs_shape), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.prod(obs_shape), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, n_actions), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

class Args:
    def __init__(self, num_steps=5, num_envs=1, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                 update_epochs=4, batch_size=64, minibatch_size=32, clip_coef=0.2, ent_coef=0.01,
                 vf_coef=0.5, max_grad_norm=0.5, anneal_lr=True, cuda=False, exp_name="ppo_cartpole", 
                 num_iterations=1000):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.anneal_lr = anneal_lr
        self.cuda = cuda
        self.exp_name = exp_name
        self.num_iterations = num_iterations

def make_env(env_id, idx):
    # Função auxiliar simples para criar ambientes sem captura de vídeo.
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk