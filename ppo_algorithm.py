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


# Função para inicializar as camadas da rede neural de forma adequada (orthogonal initialization)
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)  # Inicialização ortogonal dos pesos
    torch.nn.init.constant_(layer.bias, bias_const)  # Inicialização dos biases
    return layer


# Definição da classe PPO (Proximal Policy Optimization)
class PPO(nn.Module):
    def __init__(self, envs, hidden_size=64, gamma=0.99, lam=0.95, clip_epsilon=0.2, lr=3e-4, entropy_coef=0.01):
        super(PPO, self).__init__()

        # Parâmetros de configuração
        self.gamma = gamma  # Fator de desconto
        self.lam = lam  # Fator de GAE (Generalized Advantage Estimation)
        self.clip_epsilon = clip_epsilon  # Epsilon para clipping no PPO
        self.entropy_coef = entropy_coef  # Coeficiente de entropia para a regularização

        # Definição das dimensões de entrada (observação) e saída (ação)
        obs_dim = np.array(envs.single_observation_space.shape).prod()  # Tamanho da observação
        action_dim = envs.single_action_space.n  # Número de ações possíveis

        # Definição da rede neural para o ator (Policy Network)
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_size)),  # Primeira camada (entradas para a rede oculta)
            nn.Tanh(),  # Função de ativação
            layer_init(nn.Linear(hidden_size, hidden_size)),  # Segunda camada oculta
            nn.Tanh(),  # Função de ativação
            layer_init(nn.Linear(hidden_size, action_dim), std=0.01)  # Camada de saída (logits para as ações)
        )

        # Definição da rede neural para o crítico (Value Network)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_size)),  # Primeira camada (entradas para a rede oculta)
            nn.Tanh(),  # Função de ativação
            layer_init(nn.Linear(hidden_size, hidden_size)),  # Segunda camada oculta
            nn.Tanh(),  # Função de ativação
            layer_init(nn.Linear(hidden_size, 1), std=1.0)  # Camada de saída (valor estimado do estado)
        )

        # Definição dos otimizadores para o ator e o crítico
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)  # Otimizador para o ator
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)  # Otimizador para o crítico

    # Função de forward (passagem para frente) que retorna logits e valores do crítico
    def forward(self, obs):
        logits = self.actor(obs)  # Saídas do ator (logits para a distribuição de probabilidades)
        value = self.critic(obs)  # Valor do estado (avaliação do crítico)
        return logits, value

    # Função para selecionar uma ação com base na observação
    def act(self, obs):
        logits, _ = self.forward(obs)  # Obtendo os logits do ator
        dist = Categorical(logits=logits)  # Definindo a distribuição de probabilidades
        action = dist.sample()  # Amostrando uma ação da distribuição
        return action, dist.log_prob(action), dist.entropy()  # Retornando a ação, log-probabilidade e entropia

    # Função para computar os retornos usando a fórmula de Generalized Advantage Estimation (GAE)
    def compute_returns(self, rewards, values, next_value, dones):
        returns = torch.zeros_like(rewards)  # Inicializando o tensor de retornos
        last_gae_lam = 0  # Inicializando o termo GAE
        for t in reversed(range(len(rewards))):  # Iterando de trás para frente
            if dones[t]:
                next_value = 0  # Se o episódio terminou, o valor seguinte é zero
            delta = rewards[t] + self.gamma * next_value - values[t]  # Cálculo do delta
            last_gae_lam = delta + self.gamma * self.lam * last_gae_lam  # Atualizando o GAE
            returns[t] = last_gae_lam + values[t]  # Armazenando os retornos
            next_value = values[t]  # Atualizando o próximo valor
        return returns

    # Função para atualizar os parâmetros da rede (ator e crítico)
    def update(self, observations, actions, log_probs_old, returns, advantages):
        logits, values = self.forward(observations)  # Passando as observações pela rede
        dist = Categorical(logits=logits)  # Definindo a distribuição de probabilidades
        log_probs_new = dist.log_prob(actions)  # Calculando as log-probabilidades novas

        ratio = torch.exp(log_probs_new - log_probs_old)  # Calculando o ratio do PPO (razão entre log-probs)

        # Cálculo da função de perda do ator (surrogate loss) com clipping
        surrogate_loss = torch.min(
            ratio * advantages, 
            torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        )
        actor_loss = -torch.mean(surrogate_loss)  # Perda do ator

        # Cálculo da perda do crítico (erro quadrático médio)
        critic_loss = F.mse_loss(values.squeeze(), returns)  # Perda quadrática média

        # Cálculo da perda de entropia para a regularização
        entropy_loss = torch.mean(dist.entropy()) * self.entropy_coef  # Regularização de entropia

        # A perda total é composta pela soma das perdas do ator, crítico e a penalização de entropia
        loss = actor_loss + critic_loss - entropy_loss

        # Passo de otimização para o ator e o crítico
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()  # Backpropagation
        self.actor_optimizer.step()  # Atualização dos parâmetros do ator
        self.critic_optimizer.step()  # Atualização dos parâmetros do crítico

    # Função de treinamento
    def train(self, envs, total_timesteps=100000, num_steps=128, batch_size=64, epochs=4):
        obs = torch.tensor(envs.reset(), dtype=torch.float32)  # Inicialização das observações

        # Loop de treinamento para a quantidade de timesteps desejados
        for step in range(total_timesteps // num_steps):
            observations, actions, rewards, log_probs, dones, values = [], [], [], [], [], []

            # Coleta de dados para treinamento
            for _ in range(num_steps):
                logits, value = self.forward(obs)  # Obtendo logits e valor do estado
                dist = Categorical(logits=logits)  # Criando a distribuição de probabilidades
                action = dist.sample()  # Amostrando a ação

                observations.append(obs)  # Armazenando a observação
                actions.append(action)  # Armazenando a ação
                log_probs.append(dist.log_prob(action))  # Armazenando o log-probabilidade
                values.append(value)  # Armazenando o valor do estado

                # Realizando a ação no ambiente e obtendo a recompensa
                next_obs, reward, done, _ = envs.step(action.cpu().numpy())  # Executando a ação no ambiente
                obs = torch.tensor(next_obs, dtype=torch.float32)  # Atualizando a observação

                rewards.append(torch.tensor(reward, dtype=torch.float32))  # Armazenando a recompensa
                dones.append(torch.tensor(done, dtype=torch.float32))  # Armazenando o status de fim de episódio

            # Empacotando os dados coletados
            observations = torch.stack(observations)
            actions = torch.stack(actions)
            log_probs = torch.stack(log_probs)
            values = torch.stack(values).squeeze()
            rewards = torch.stack(rewards).squeeze()
            dones = torch.stack(dones).squeeze()

            # Computando os retornos e as vantagens
            next_value = self.critic(obs).detach()  # Valor do próximo estado
            returns = self.compute_returns(rewards, values, next_value, dones)  # Computando os retornos
            advantages = returns - values  # Calculando as vantagens

            # Atualizando o modelo por várias épocas
            for _ in range(epochs):
                for i in range(0, num_steps, batch_size):
                    # Pegando lotes de dados
                    batch_obs = observations[i:i + batch_size]
                    batch_actions = actions[i:i + batch_size]
                    batch_log_probs = log_probs[i:i + batch_size]
                    batch_returns = returns[i:i + batch_size]
                    batch_advantages = advantages[i:i + batch_size]

                    # Atualizando os parâmetros do modelo
                    self.update(batch_obs, batch_actions, batch_log_probs, batch_returns, batch_advantages)
