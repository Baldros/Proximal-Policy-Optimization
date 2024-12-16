# Libs:
import gymnasium as gym
from PPO import *

if __name__ == "__main__":
    args = Args(
        num_steps=128,
        num_envs=4,
        learning_rate=2.5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        update_epochs=4,
        batch_size=4*128, # num_envs * num_steps
        minibatch_size=(4*128)//4,
        clip_coef=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        anneal_lr=True,
        cuda=torch.cuda.is_available(),
        exp_name="ppo_cartpole",
        num_iterations=500 # Exemplo
    )

    envs = gym.vector.SyncVectorEnv(
        [make_env("CartPole-v1", i) for i in range(args.num_envs)]
    )

    agent = PPOAgent(envs.single_observation_space, envs.single_action_space)
    trainer = PPOTrainer(agent, envs, args)
    trainer.train_loop()

    # Acesso às métricas
    metrics = trainer.get_metrics()
    print(metrics)
  
