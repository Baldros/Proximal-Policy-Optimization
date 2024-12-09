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


# Função para inicializar as camadas com pesos ortogonais e viés constante.
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)  # Inicializa os pesos da camada com uma distribuição ortogonal.
    torch.nn.init.constant_(layer.bias, bias_const)  # Inicializa o viés com um valor constante.
    return layer


# Definição da classe do agente, que implementa o ator e o crítico.
class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        # Critic: Rede neural para estimar o valor do estado.
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),  # Camada de entrada para o crítico.
            nn.Tanh(),  # Função de ativação Tanh.
            layer_init(nn.Linear(64, 64)),  # Camada oculta do crítico.
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),  # Camada de saída do crítico, saída de valor do estado.
        )
        
        # Actor: Rede neural para determinar a política (probabilidade de ações).
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),  # Camada de entrada para o ator.
            nn.Tanh(),  # Função de ativação Tanh.
            layer_init(nn.Linear(64, 64)),  # Camada oculta do ator.
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),  # Camada de saída do ator, probabilidade de ações.
        )

    # Função para obter o valor de um estado.
    def get_value(self, x):
        return self.critic(x)

    # Função para obter a ação e o valor associado, além de log-probabilidade e entropia.
    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)  # Calcula os logits das ações.
        probs = Categorical(logits=logits)  # Converte os logits em probabilidades de ações.
        if action is None:  # Se nenhuma ação foi passada, amostra uma ação.
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


# Definição da classe PPO (Proximal Policy Optimization).
class PPO:
    def __init__(self, args, envs):
        self.args = args
        # Configura o dispositivo para CPU ou GPU, dependendo da disponibilidade.
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        self.envs = envs
        # Criação do agente (actor-critic).
        self.agent = Agent(envs).to(self.device)
        # Otimizador Adam para atualizar os parâmetros do agente.
        self.optimizer = optim.Adam(self.agent.parameters(), lr=args.learning_rate, eps=1e-5)
        # Inicializa o TensorBoard para monitoramento de experimentos.
        self.writer = SummaryWriter(f"runs/{args.exp_name}__{int(time.time())}")

    # Função de treinamento do agente.
    def train(self):
        # Inicialização de tensores para armazenar as observações, ações, log-probabilidades, recompensas, etc.
        obs = torch.zeros((self.args.num_steps, self.args.num_envs) + self.envs.single_observation_space.shape).to(self.device)
        actions = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        logprobs = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        rewards = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        dones = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        values = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)

        global_step = 0
        start_time = time.time()
        next_obs, _ = self.envs.reset()  # Resetando o ambiente.
        next_obs = torch.Tensor(next_obs).to(self.device)
        next_done = torch.zeros(self.args.num_envs).to(self.device)

        # Loop de iterações de treinamento.
        for update in range(1, self.args.num_iterations + 1):
            # Annealing da taxa de aprendizado (diminuir a taxa de aprendizado ao longo do tempo).
            if self.args.anneal_lr:
                frac = 1.0 - (update - 1.0) / self.args.num_iterations
                lrnow = frac * self.args.learning_rate
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lrnow

            # Loop de passos no ambiente para coletar dados de experiência.
            for step in range(self.args.num_steps):
                global_step += 1 * self.args.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(next_obs)  # Calcula a ação e o valor.
                actions[step] = action
                logprobs[step] = logprob
                values[step] = value.flatten()

                # Executa a ação no ambiente e coleta a recompensa, terminação e informações de cada ambiente.
                next_obs, reward, terminated, truncated, infos = self.envs.step(action.cpu().numpy())
                rewards[step] = torch.tensor(reward).to(self.device).view(-1)
                next_done = torch.tensor(terminated | truncated).to(self.device)

            # Calculando as vantagens usando GAE (Generalized Advantage Estimation).
            with torch.no_grad():
                next_value = self.agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.args.num_steps)):
                    nextnonterminal = 1.0 - dones[t]
                    nextvalues = next_value if t == self.args.num_steps - 1 else values[t + 1]
                    delta = rewards[t] + self.args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + self.args.gamma * self.args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # Flattening dos tensores para otimização do modelo.
            b_obs = obs.reshape((-1,) + self.envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Otimização do modelo: Atualiza os parâmetros do ator e do crítico.
            for _ in range(self.args.update_epochs):
                indices = np.arange(self.args.batch_size)
                np.random.shuffle(indices)
                for start in range(0, self.args.batch_size, self.args.minibatch_size):
                    end = start + self.args.minibatch_size
                    mb_inds = indices[start:end]

                    # Calcula as novas log-probabilidades, entropia e valor.
                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    # Cálculo da perda de política (PG loss).
                    mb_advantages = b_advantages[mb_inds]
                    if self.args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Cálculo da perda de valor (Value loss).
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

                    # Cálculo da perda de entropia e do total de perdas.
                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.args.ent_coef * entropy_loss + v_loss * self.args.vf_coef

                    # Backpropagation para otimizar o modelo.
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()

            # Log do progresso de treinamento a cada 10 atualizações.
            if update % 10 == 0:
                print(f"Update {update} completed.")
