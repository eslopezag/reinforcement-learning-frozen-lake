from typing import Callable

import numpy as np
import gym

from rl_agents.q_approximators import TabularQApproximator
from rl_agents.policies import EpsGreedyPolicy, SoftmaxPolicy
from rl_agents.schedulers import (
    cosine_decay_scheduler,
    exp_decay_scheduler,
    constant_scheduler,
)
from rl_agents.agents import (
    TabularSarsaAgent,
    TabularQLearningAgent,
    TabularExpectedSarsaAgent
)
from custom_agent_history import CustomAgentHistory


def get_frozen_lake_agent(
    name: str,
    map_size: int = 4,
    discount: float = 1.,
    initial_Q_mean: float = 1.,
    initial_Q_std: float = 0.,
    training_steps: int = 100000,
    step_size_scheduler: Callable = constant_scheduler(0.01)
):
    """
    Returns the specified agent with the given characteristics.
    """
    if map_size == 4:
        map_name = '4x4'
    elif map_size == 8:
        map_name = '8x8'
    else:
        raise ValueError('The map size must be 4 or 8.')

    env = gym.make(
        'FrozenLake-v1',
        desc=None,
        map_name=map_name,
        is_slippery=True,
    )
    env._max_episode_steps = float('inf')

    terminal_states = [
        int(np.sqrt(env.observation_space.n) * i + j)
        for i, j in zip(
            *((env.desc == b'H') + (env.desc == b'G')).nonzero()
        )
    ]

    Q = TabularQApproximator.from_normal_dist(
        scheduler=step_size_scheduler,
        initial_Q_mean=initial_Q_mean,
        initial_Q_std=initial_Q_std,
        env=env,
        terminal_states=terminal_states,
    )

    if name == 'sarsa':
        agent = TabularSarsaAgent(
            env=env,
            Q=Q,
            history=CustomAgentHistory(discount, env=env),
            target_policy=EpsGreedyPolicy(
                scheduler=cosine_decay_scheduler(0.3, 1e-6, training_steps),
            ),
            discount=discount,
            output_filename='sarsa_agent',
        )

    elif name == 'q_learning':
        agent = TabularQLearningAgent(
            env=env,
            Q=Q,
            history=CustomAgentHistory(discount, env=env),
            exploration_policy=EpsGreedyPolicy(eps=1.),
            discount=discount,
            output_filename='q_learning_agent',
        )

    elif name == 'q_learning_eps':
        agent = TabularQLearningAgent(
            env=env,
            Q=Q,
            history=CustomAgentHistory(discount, env=env),
            exploration_policy=EpsGreedyPolicy(
                scheduler=cosine_decay_scheduler(1, 0.4, training_steps)
            ),
            discount=discount,
            output_filename='q_learning_eps_agent',
        )

    elif name == 'expected_sarsa':
        agent = TabularExpectedSarsaAgent(
            env=env,
            Q=Q,
            history=CustomAgentHistory(discount, env=env),
            target_policy=EpsGreedyPolicy(
                scheduler=cosine_decay_scheduler(
                    0.1, 0, training_steps // 2
                ),
            ),
            exploration_policy=SoftmaxPolicy(
                scheduler=exp_decay_scheduler(
                    10, 0.1, training_steps
                ),
            ),
            discount=discount,
            output_filename='expected_sarsa_agent',
        )

    return agent
