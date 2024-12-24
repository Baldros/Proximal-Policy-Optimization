# Presentation:

**Reinforcement Learning (RL)** is an iterative machine learning paradigm in which an **agent** learns to make decisions within an **environment** by interacting with it. For each **action** taken, the agent receives a **reward** (positive or negative), and the goal is to **maximize** the sum of these rewards over time. There is a wide range of learning methods in the context of RL, such as policy-based methods, value-based methods, and hybrid methods (like actor-critic), among others. Here, the **Proximal Policy Optimization (PPO)** algorithm is implemented—a policy optimization method created by researchers at [OpenAI](https://openai.com/index/openai-baselines-ppo/)—which I am using to solve gamified tasks in the [Gymnasium](https://gymnasium.farama.org/) environment. 

## Proximal Policy Optmization:

O **PPO** é considerado um **método de Otimização de Política** que complementa sua estrutura de aprendizado com uma **arquitetura de ator-critico**, para estimar vantagens (**Advantage Function**) e estabilizar o gradiente, cuja a estratégia principal é buscar manter a nova política em uma região próxima da política anterior, reduzindo o risco de grandes saltos de desempenho negativo.

$$A_t = \delta_t + (\gamma \lambda)\delta_{t+1} + \cdots + (\gamma \lambda)^{T-t+1}\delta_{T-1}, \quad \text{where } \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

Para isso, ele utiliza uma função de perda com **clipping**, limitando a diferença entre a probabilidade de uma ação sob a nova política e a probabilidade sob a política antiga. Dessa forma, evita atualizações excessivas e promove maior estabilidade durante o treinamento.

$$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left(r_t(\theta)\hat{A}_t, \text{clip}\left(r_t(\theta), 1 - \epsilon, 1 + \epsilon\right)\hat{A}_t\right)\right]$$

onde,

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$$


# Repository Organization:
Each folder corresponds to a different implementation. The first folder maintains the original code's functionality, focusing on organizing the model's implementation in a more functional way. The second implementation uses only essential dependencies, aiming to simplify the dependency versioning process.
