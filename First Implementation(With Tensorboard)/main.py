import os
import time
import random
import numpy as np
import torch
import torch.backends.cudnn
import gymnasium as gym
import argparse
from dataclasses import dataclass

from ppo_agent import PPOAgent
from ppo_trainer import PPOTrainer

def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="ppo_experiment", help="Experiment name")
    parser.add_argument("--seed", type=int, default=1, help="Seed")
    parser.add_argument("--torch_deterministic", action="store_true", default=True)
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument("--track", action="store_true", default=False)
    parser.add_argument("--wandb_project_name", type=str, default="cleanRL")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--capture_video", action="store_true", default=False)

    parser.add_argument("--env_id", type=str, default="CartPole-v1")
    parser.add_argument("--total_timesteps", type=int, default=500000)
    parser.add_argument("--learning_rate", type=float, default=2.5e-4)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument("--anneal_lr", action="store_true", default=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--num_minibatches", type=int, default=4)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--norm_adv", action="store_true", default=True)
    parser.add_argument("--clip_coef", type=float, default=0.2)
    parser.add_argument("--clip_vloss", action="store_true", default=True)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--target_kl", type=float, default=None)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    # Calcular parâmetros derivados
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    args.run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Criar ambientes
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, args.run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "Apenas espaços de ação discretos são suportados."

    # Criar agente
    agent = PPOAgent(envs.single_observation_space, envs.single_action_space)

    # Treinador
    trainer = PPOTrainer(agent, envs, args)
    trainer.train_loop()
