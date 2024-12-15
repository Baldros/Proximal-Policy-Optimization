import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

class PPOTrainer:
    def __init__(self, agent, envs, args):
        self.agent = agent
        self.envs = envs
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.args.cuda else "cpu")
        self.agent.to(self.device)

        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.args.learning_rate, eps=1e-5)

        # Buffers de armazenamento
        self.obs = torch.zeros((self.args.num_steps, self.args.num_envs) + envs.single_observation_space.shape).to(self.device)
        self.actions = torch.zeros((self.args.num_steps, self.args.num_envs) + envs.single_action_space.shape).to(self.device)
        self.logprobs = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        self.rewards = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        self.dones = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        self.values = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)

        self.writer = SummaryWriter(f"runs/{self.args.run_name}")
        self.global_step = 0
        self.start_time = time.time()

        # Estado inicial do ambiente
        next_obs, _ = envs.reset(seed=self.args.seed)
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
                            self.writer.add_scalar("charts/episodic_return", reward_sum, self.global_step)
                            self.writer.add_scalar("charts/episodic_length", episode_length, self.global_step)
                            if episode_time is not None:
                                self.writer.add_scalar("charts/episodic_time", episode_time, self.global_step)

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
                    if self.args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.args.clip_vloss:
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

                if self.args.target_kl is not None and approx_kl > self.args.target_kl:
                    break

            # Métricas
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            self.writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], self.global_step)
            self.writer.add_scalar("losses/value_loss", v_loss.item(), self.global_step)
            self.writer.add_scalar("losses/policy_loss", pg_loss.item(), self.global_step)
            self.writer.add_scalar("losses/entropy", entropy_loss.item(), self.global_step)
            self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), self.global_step)
            self.writer.add_scalar("losses/approx_kl", approx_kl.item(), self.global_step)
            self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), self.global_step)
            self.writer.add_scalar("losses/explained_variance", explained_var, self.global_step)
            print("SPS:", int(self.global_step / (time.time() - self.start_time)))
            self.writer.add_scalar("charts/SPS", int(self.global_step / (time.time() - self.start_time)), self.global_step)

        self.envs.close()
        self.writer.close()
