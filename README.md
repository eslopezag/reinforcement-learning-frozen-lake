# Reinforcement Learning Educational Project: Frozen Lake

Ths is an educational project consisting in applying Reinforcement Learning to OpenAI Gym's [Frozen Lake environment](https://gym.openai.com/envs/FrozenLake8x8-v0/).

To train the agent using SARSA, run one of the following:

```shell
make train_sarsa
```

```shell
python sarsa_training.py
```

To train the agent using Q-learning, run one of the following:

```shell
make train_q_learning
```

```shell
python q_learning_training.py
```

To see a simulation where actions are taken by the agent trained with SARSA, run one of the following:

```shell
make simulate_sarsa
```

```shell
python simulate_agent.py "sarsa_agent.dill"
```

To see a simulation where actions are taken by the agent trained with Q-learning, run one of the following:

```shell
make simulate_q_learning
```

```shell
python simulate_agent.py "q_learning_agent.dill"
```

To see a simulation where actions are taken randomly for comparison, run one of the following:

```shell
make simulate_random_policy
```

```shell
python simulate_random_policy.py
```


