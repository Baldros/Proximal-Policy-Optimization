

import os
import gymnasium as gym
from PPOv3 import *

if __name__ == "__main__":
    # Caminho para o arquivo de configuração JSON
    config_path = r"Estudo\Deepmath\config.json"
    # Verifica se o arquivo de configuração existe
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"O arquivo de configuração {config_path} não foi encontrado.")
    # Carrega os argumentos do arquivo JSON
    args = Args(config_path)

    # Cria os ambientes, por exemplo, usando AsyncVectorEnv para paralelização
    from gymnasium.vector import AsyncVectorEnv
    envs = AsyncVectorEnv([make_env(args.gym_task, i) for i in range(args.num_envs)])
    # Cria o treinador PPO
    agent = PPOAgent(envs.single_observation_space, envs.single_action_space, args)
    trainer = PPOTrainer(agent, envs, args)
    # Inicia o loop de treinamento
    trainer.train_loop()
    ## Obtém e possivelmente salva as métricas
    #metrics = trainer.get_metrics()

    # Obtém e possivelmente salva as métricas
    metrics = trainer.get_metrics()
    with open("metrics.json", "w") as f:
        # Para salvar as métricas de maneira legível, converte os tensores para listas
        serializable_metrics = {k: [(step, float(val)) for step, val in v] for k, v in metrics.items()}
        json.dump(serializable_metrics, f, indent=4)
    
    print("Treinamento concluído e métricas salvas em 'metrics.json'.")