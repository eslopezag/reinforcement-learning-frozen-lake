# Reinforcement Learning Educational Project: Frozen Lake

Ths is an educational project consisting in applying Reinforcement Learning to OpenAI Gym's [Frozen Lake environment](https://gym.openai.com/envs/FrozenLake8x8-v0/).

In the commands described in the following, `<GRIDWORLD-SIDE-LENGTH>` can be either 4 or 8, and `<NUMBER-OF-STEPS>` is the number of steps for which the training algorithm is run.

To train the agent using SARSA, run one of the following:

```shell
make train_sarsa size=<GRIDWORLD-SIDE-LENGTH> steps=<NUMBER-OF-STEPS>
```

```shell
python sarsa_training.py <GRIDWORLD-SIDE-LENGTH> <NUMBER-OF-STEPS>
```

To train the agent using Expected SARSA, run one of the following:

```shell
make train_expected_sarsa size=<GRIDWORLD-SIDE-LENGTH> steps=<NUMBER-OF-STEPS>
```

```shell
python expected_sarsa_training.py <GRIDWORLD-SIDE-LENGTH> <NUMBER-OF-STEPS>
```

To train the agent using Q-learning with fully exploratory behavior, run one of the following:

```shell
make train_q_learning size=<GRIDWORLD-SIDE-LENGTH> steps=<NUMBER-OF-STEPS>
```

```shell
python q_learning_training.py <GRIDWORLD-SIDE-LENGTH> <NUMBER-OF-STEPS>
```

To train the agent using Q-learning with epsilon-greedy behavior policy, run one of the following:

```shell
make train_q_learning_eps size=<GRIDWORLD-SIDE-LENGTH> steps=<NUMBER-OF-STEPS>
```

```shell
python q_learning_eps_training.py <GRIDWORLD-SIDE-LENGTH> <NUMBER-OF-STEPS>
```

To see a simulation where actions are taken by a given agent, run one of the following:

```shell
make simulate_agent agent_name=<AGENT-NAME>
```

```shell
python simulate_agent.py <AGENT-NAME>
```

Where `<AGENT-NAME>` is one of:

- `sarsa_agent`
- `q_learning_agent` (fully exploratory behavior)
- `expected_sarsa_agent`
- `q_learning_eps_agent` (epsilon-greedy behavior policy)

To see a simulation where actions are taken randomly for comparison, run one of the following:

```shell
make simulate_random_policy size=<GRIDWORLD-SIDE-LENGTH>
```

```shell
python simulate_random_policy.py <GRIDWORLD-SIDE-LENGTH>
```

To generate a GIF with the simulation of a given agent, run one of the following:

```shell
make generate_gif agent_name=<AGENT-NAME>
```

```shell
python generate_gif_from_agent.py <AGENT-NAME>
```
