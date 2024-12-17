# Bibliotecas Utilizadas:
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

# Funções Utilizadas ##################################################################################################

def make_env(env_id, idx):
    # Função auxiliar para criar o ambiente Gym.
    # Usamos RecordEpisodeStatistics para obter recompensas e comprimentos de episódio.
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk
    
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    # Inicialização ortogonal dos pesos, conforme boas práticas sugeridas em PPO e outros trabalhos
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# Classes #############################################################################################################

class Args:
    # Classe simples para armazenar hiperparâmetros e configurações.
    # Esses parâmetros seguem o padrão PPO: lr, gamma, lambda (GAE), clipping, etc.
    def __init__(self, num_steps=5, num_envs=1, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                 update_epochs=4, batch_size=64, minibatch_size=32, clip_coef=0.2, ent_coef=0.01,
                 vf_coef=0.5, max_grad_norm=0.5, anneal_lr=True, cuda=False, exp_name="ppo_cartpole",
                 num_iterations=1000,printOn=True):
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
        self.printOn = printOn

class PPOAgent(nn.Module):
    # Agente PPO: Redes ator e crítico
    # Conforme descrito no paper, usamos duas redes compartilhando camadas iniciais ou separadas:
    # Aqui temos um ator (para logitos de ação) e um crítico (para valor), ambos MLPs.
    def __init__(self, observation_space, action_space):
        super().__init__()
        obs_shape = observation_space.shape
        n_actions = action_space.n

        # Rede crítica
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.prod(obs_shape), 64)), # np.prod((i,j)) = i*j
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        # Rede ator
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.prod(obs_shape), 64)), # np.prod((i,j)) = i*j
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, n_actions), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        # Atores PPO geralmente produzem uma distribuição de probabilidade sobre ações.
        logits = self.actor(x)
        probs = Categorical(logits=logits) # Aqui usamos a distribuição Categorical (Multinomial) para ações discretas.
        if action is None:
            action = probs.sample()  # Amostra aleatóriamente uma ação da política atual
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)



class PPOTrainer:
    def __init__(self, agent, envs, args):
        self.agent = agent
        self.envs = envs
        self.args = args
        # Define o dispositivo (GPU ou CPU) conforme disponibilidade e preferência
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.args.cuda else "cpu")
        self.agent.to(self.device)

        # Otimizador Adam para atualização dos parâmetros do agente
        # PPO usa um otimizador como o Adam para atualizar a rede ator-crítico
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.args.learning_rate, eps=1e-5)

        # Criação de tensores para armazenamento dos dados de rollout
        # Aqui armazenamos observações, ações, log-probs, recompensas e valores
        # em buffers para computar as vantagens (GAE) e returns ao final.
        self.obs = torch.zeros((self.args.num_steps, self.args.num_envs) + envs.single_observation_space.shape).to(self.device)
        self.actions = torch.zeros((self.args.num_steps, self.args.num_envs) + envs.single_action_space.shape).to(self.device)
        self.logprobs = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        self.rewards = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        self.dones = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        self.values = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)

        # Dicionário para armazenar métricas de desempenho e diagnóstico
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

        # Reset inicial do ambiente
        # Obtém a primeira observação do ambiente
        next_obs, _ = envs.reset(seed=None)
        self.next_obs = torch.Tensor(next_obs).to(self.device)
        self.next_done = torch.zeros(self.args.num_envs).to(self.device)

    def anneal_lr(self, iteration):
        # Ajuste da taxa de aprendizado (annealing)
        # PPO sugere a redução gradual da learning rate ao longo do treinamento.
        # Aqui, lrnow é a learning rate atual, calculada de forma linear decrescente
        # a partir da learning_rate inicial. Isso ajuda na estabilidade do treino.
        if self.args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / self.args.num_iterations
            lrnow = frac * self.args.learning_rate
            self.optimizer.param_groups[0]["lr"] = lrnow

    def compute_advantages(self, next_value):
        # Inicializa um tensor para armazenar as vantagens (GAE)
        # As vantagens medem quão boa foi uma ação comparada ao valor esperado.
        advantages = torch.zeros_like(self.rewards).to(self.device)
        
        # Inicializa a variável lastgaelam, que armazena o GAE para o próximo passo
        lastgaelam = 0
    
        # Percorre os passos na ordem reversa (do final para o início do rollout)
        # Isso é necessário porque o cálculo da vantagem depende do próximo valor.
        for t in reversed(range(self.args.num_steps)):
    
            # Se estamos no último passo da trajetória (rollout)
            if t == self.args.num_steps - 1:
                # Se o episódio terminou no próximo estado, o nextnonterminal será 0.
                nextnonterminal = 1.0 - self.next_done  # 1 se não terminou, 0 se terminou.
                # Valor estimado do próximo estado
                nextvalues = next_value
            else:
                # Caso contrário, usamos os valores do próximo passo na trajetória
                nextnonterminal = 1.0 - self.dones[t + 1]  # 1 se não terminou, 0 se terminou.
                nextvalues = self.values[t + 1]  # Valor do próximo estado já armazenado no rollout
    
            # 1. Cálculo do **delta**, que mede a vantagem imediata
            '''
            delta captura o **erro temporal** (TD Error), ou seja:
            - A diferença entre a recompensa recebida + valor futuro descontado e o valor atual.
            - Se delta > 0: a ação foi melhor do que o valor estimado.
            - Se delta < 0: a ação foi pior do que o valor estimado.
            '''
            delta = self.rewards[t] + self.args.gamma * nextvalues * nextnonterminal - self.values[t]
            
    
            # 2. Cálculo da vantagem (GAE) acumulada
            # GAE combina o delta atual com a vantagem acumulada futura, amortizada pelo fator lambda.
            advantages[t] = lastgaelam = delta + self.args.gamma * self.args.gae_lambda * nextnonterminal * lastgaelam
            '''
            Interpretação:
            - `delta` contribui para a vantagem atual.
            - `lastgaelam` acumula as vantagens futuras, descontadas por gamma e amortizadas por lambda.
            - Isso cria uma estimativa "suavizada" das vantagens, reduzindo a variância.
            '''
        
        # 3. Cálculo dos retornos
        # O retorno é simplesmente a soma da vantagem (GAE) com o valor do estado.
        # Retorno = Vantagem (GAE) + Valor estimado
        returns = advantages + self.values
        
        return advantages, returns


    def train_loop(self):
        # Loop de treinamento principal do PPO
        # Cada iteração corresponde a coletar 'num_steps' de experiências em cada um dos 'num_envs' ambientes
        # e depois realizar as atualizações de parâmetro.
        for iteration in range(1, self.args.num_iterations + 1):
            # Ajusta a learning rate conforme a iteração (annealing)
            self.anneal_lr(iteration)

            # Coleta de dados (rollout)
            for step in range(self.args.num_steps):
                self.global_step += self.args.num_envs
                self.obs[step] = self.next_obs
                self.dones[step] = self.next_done

                # Obtém ação e valor a partir da política atual (rede ator-crítico)
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(self.next_obs)
                    self.values[step] = value.flatten()

                # Armazena ação e log-probabilidade da ação atual
                self.actions[step] = action
                self.logprobs[step] = logprob

                # Avança no ambiente com a ação escolhida
                # Recebe nova observação, recompensa e flags de término (terminations, truncations)
                next_obs, reward, terminations, truncations, infos = self.envs.step(action.cpu().numpy())
                self.next_done = np.logical_or(terminations, truncations)
                self.rewards[step] = torch.tensor(reward).to(self.device).view(-1)
                self.next_obs = torch.Tensor(next_obs).to(self.device)
                self.next_done = torch.Tensor(self.next_done).to(self.device)

                # Checagem de final de episódio e registro de métricas
                if "episode" in infos:
                    for idx, episode_finished in enumerate(infos["_episode"]):
                        if episode_finished:
                            reward_sum = infos["episode"]["r"][idx]
                            episode_length = infos["episode"]["l"][idx]
                            episode_time = infos["episode"].get("t", [None])[idx]
                            
                            if self.args.printOn:
                                print(f"global_step={self.global_step}, episodic_return={reward_sum}")

                            # Armazena métricas de episódio
                            self.metrics["episodic_return"].append((self.global_step, reward_sum))
                            self.metrics["episodic_length"].append((self.global_step, episode_length))
                            if episode_time is not None:
                                self.metrics["episodic_time"].append((self.global_step, episode_time))

            # Cálculo do valor de bootstrap no próximo estado (para GAE)
            with torch.no_grad():
                next_value = self.agent.get_value(self.next_obs).reshape(1, -1)
            advantages, returns = self.compute_advantages(next_value)

            # "Flatten" dos batchs para preparar para atualização em minibatches
            b_obs = self.obs.reshape((-1,) + self.envs.single_observation_space.shape)
            b_logprobs = self.logprobs.reshape(-1)
            b_actions = self.actions.reshape((-1,) + self.envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = self.values.reshape(-1)

            clipfracs = []
            # Atualização de parâmetros da política e da função de valor
            # Dividimos o batch em minibatches e realizamos várias epochs (update_epochs)
            # conforme sugerido no PPO.
            for epoch in range(self.args.update_epochs):
                inds = np.arange(self.args.batch_size)
                np.random.shuffle(inds)
                for start in range(0, self.args.batch_size, self.args.minibatch_size):
                    end = start + self.args.minibatch_size
                    mb_inds = inds[start:end]

                    # Recalcula logprob e valor para o minibatch
                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                        b_obs[mb_inds], b_actions.long()[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    # KL aproximada para diagnóstico (não diretamente usada na perda padrão)
                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()]

                    # Normalização da vantagem, conforme sugerido, para estabilidade
                    mb_advantages = b_advantages[mb_inds]
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Cálculo da perda da política (Policy Loss)
                    # PPO: maximizamos L^CLIP, que usa a min entre ratio*advantage e ratio clamped.
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Cálculo da perda do valor (Value Loss)
                    # Clipping do valor, conforme a abordagem PPO para reduzir instabilidades.
                    newvalue = newvalue.view(-1)
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.args.clip_coef,
                        self.args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                    # Entropia para incentivar a exploração
                    entropy_loss = entropy.mean()

                    # Soma total da perda (Policy Loss + c1 * Value Loss + c2 * Entropy Loss)
                    # Conforme descrito no paper do PPO, a loss total combina essas componentes.
                    loss = pg_loss - self.args.ent_coef * entropy_loss + v_loss * self.args.vf_coef

                    # Atualiza parâmetros da rede
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()

            # Cálculo do "explained_variance" para avaliar a qualidade da função de valor
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # Registro de métricas
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.metrics["learning_rate"].append((self.global_step, current_lr))
            self.metrics["value_loss"].append((self.global_step, v_loss.item()))
            self.metrics["policy_loss"].append((self.global_step, pg_loss.item()))
            self.metrics["entropy_loss"].append((self.global_step, entropy_loss.item()))
            self.metrics["old_approx_kl"].append((self.global_step, old_approx_kl.item()))
            self.metrics["approx_kl"].append((self.global_step, approx_kl.item()))
            self.metrics["clipfrac"].append((self.global_step, np.mean(clipfracs)))
            self.metrics["explained_variance"].append((self.global_step, explained_var))

            # Steps per second (SPS) para avaliar velocidade de treinamento
            sps = int(self.global_step / (time.time() - self.start_time))
            self.metrics["SPS"].append((self.global_step, sps))
            if self.args.printOn:
                print("SPS:", sps)

        self.envs.close()

    def get_metrics(self):
        return self.metrics
