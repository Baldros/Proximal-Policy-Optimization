# Libs:
import os
import gymnasium as gym

# modules:
from Args import *
from PPOAgent import *
from PPOTrainver import *

# Implementation:
if __name__ == "__main__":
    # Path to the JSON configuration file
    config_path = r"Estudo\Deepmath\config.json"
    # Checks if the configuration file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"The configuration file {config_path} was not found.")
    # Loads arguments from the JSON file
    args = Args(config_path)

    # Creates environments, for example, using AsyncVectorEnv for parallelization
    from gymnasium.vector import AsyncVectorEnv
    envs = AsyncVectorEnv([make_env(args.gym_task, i) for i in range(args.num_envs)])
    # Creates the PPO trainer
    agent = PPOAgent(envs.single_observation_space, envs.single_action_space, args)
    trainer = PPOTrainer(agent, envs, args)
    # Starts the training loop
    trainer.train_loop()

    # Retrieves and optionally saves the metrics
    metrics = trainer.get_metrics()
    with open("metrics.json", "w") as f:
        # To save the metrics in a readable way, converts tensors to lists
        serializable_metrics = {k: [(step, float(val)) for step, val in v] for k, v in metrics.items()}
        json.dump(serializable_metrics, f, indent=4)
    
    print("Training completed and metrics saved to 'metrics.json'.")
