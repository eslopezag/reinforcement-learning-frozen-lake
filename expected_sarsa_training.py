from frozen_lake_agents import get_frozen_lake_agent
from rl_agents.schedulers import cosine_decay_scheduler


if __name__ == '__main__':

    import sys

    map_size, training_steps = tuple(map(int, sys.argv[1:3]))

    agent = get_frozen_lake_agent(
        name='expected_sarsa',
        map_size=map_size,
        discount=1.,
        initial_Q_mean=0.8,
        initial_Q_std=0.05,
        training_steps=training_steps,
        step_size_scheduler=cosine_decay_scheduler(
            0.1,
            0.002,
            training_steps,
        ),
    )

    agent.train(training_steps)
    agent.save('saved_agents/expected_sarsa_agent')
    agent.show_training_results()
