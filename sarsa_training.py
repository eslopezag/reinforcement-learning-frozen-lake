import gym

from agent_training import Agent
from schedulers import cosine_decay_scheduler


if __name__ == '__main__':

    import sys

    map_size, training_steps = sys.argv[1:3]

    if map_size == '4':
        map_name = '4x4'
    elif map_size == '8':
        map_name = '8x8'
    else:
        raise ValueError('The map size must be 4 or 8.')

    training_steps = int(training_steps)

    env = gym.make(
        'FrozenLake-v1',
        desc=None,
        map_name=map_name,
        is_slippery=True,
    )

    agent = Agent(
        training_alg='sarsa',
        environment=env,
        initial_Q_mean=0.8,
        initial_Q_std=0.05,
        discount=1.,
    )

    agent.train(
        training_steps=training_steps,
        step_size_scheduler=cosine_decay_scheduler(
            0.01,
            0.0005,
            training_steps,
        ),
        epsilon_scheduler=cosine_decay_scheduler(0.3, 1e-6, training_steps),
    )
