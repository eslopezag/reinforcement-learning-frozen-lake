from typing import List

import numpy as np
import gym
from tqdm import tqdm
import dill

from analytical_optimal_policy import AnalyticalOptimalPolicyFinder


class AgentHistory:
    def __init__(self, map_size: int, discount: float) -> None:
        apf = AnalyticalOptimalPolicyFinder(map_size, discount)
        initial_policy = [2 for _ in range(map_size ** 2)]
        self.optimal_policies, self.optimal_action_values = (
            apf.policy_iteration(initial_policy)
        )
        self.optimal_state_values = np.max(self.optimal_action_values, axis=1)

        # Initialize the list that will contain the indices of the time steps
        # where episodes ended:
        self.episode_ends = []

        # Initialize the list that will contain the average fo the RMSE across
        # the states:
        self.V_RMSE = []

        # Initialize the list that will contain the distance of the policy to
        # the optimal one, found as the number of states for which the action
        # is non-optimal:
        self.policy_distances_to_optimal = []

    def register_Q(self, Q: np.ndarray) -> None:
        V = np.max(Q, axis=1)  # get the state-values assuming greedy policy
        self.V_RMSE.append(
            np.sqrt(np.square(self.optimal_state_values - V).mean())
        )

    def register_policy(self, policy: List[int]) -> None:
        self.policy_distances_to_optimal.append(
            sum(
                a not in optimal_action
                for a, optimal_action in zip(policy, self.optimal_policies)
            )
        )

    def register_episode_end(self):
        self.episode_ends.append(len(self.V_RMSE))


class Agent:
    def __init__(
        self,
        environment: gym.Env,
        eps: float,
        discount: float,
        mode: str = 'training',
    ) -> None:
        self.env = environment
        self.eps = eps
        self.discount = discount

        if mode == 'training' or mode == 'inference':
            self.mode = mode
        else:
            raise ValueError(
                'The agent\'s mode can only be "training" or "inference."'
            )

        self.last_state = None
        self.last_action = None

        self.terminal_states = [
            int(np.sqrt(self.env.observation_space.n) * i + j)
            for i, j in zip(
                *((self.env.desc == b'H') + (self.env.desc == b'G')).nonzero()
            )
        ]

        self.Q = np.full(
            (self.env.observation_space.n, self.env.action_space.n),
            1.,
            dtype='float64',
        )
        self.Q[self.terminal_states] = 0

        self.greedy_policy = np.argmax(self.Q, axis=1)

        self.history = AgentHistory(
            int(self.env.observation_space.n ** (1/2)),
            self.discount,
        )

    def get_action(self, state) -> int:
        if self.mode == 'inference':
            return self.greedy_policy[state]
        elif self.mode == 'training':
            if np.random.random() <= self.eps:
                return np.random.randint(self.env.action_space.n)
            else:
                return self.greedy_policy[state]

    def set_mode(self, new_mode: str) -> None:
        if new_mode == 'training' or new_mode == 'inference':
            self.mode = new_mode
        else:
            raise ValueError(
                'The agent\'s mode can only be "training" or "inference."'
            )

    def train_step(
        self,
        state: int,
        reward: float,
        done: bool,
        step_size: float
    ):
        if self.mode != 'training':
            raise Exception('The agent cannot be trained in inference mode.')

        action = self.get_action(state)

        if self.last_state is not None and self.last_action is not None:

            greedy_action = np.argmax(self.Q[state])

            actions_prob = np.zeros(self.env.action_space.n)
            actions_prob[greedy_action] = 1 - self.eps
            actions_prob += np.full(
                self.env.action_space.n,
                self.eps / self.env.action_space.n,
            )

            self.Q[self.last_state, self.last_action] += step_size * (
                reward + self.discount * np.dot(actions_prob, self.Q[state]) -
                self.Q[self.last_state, self.last_action]
            )

        if done:
            self.history.register_episode_end()
            self.last_state = None
            self.last_action = None
        else:
            self.last_state = state
            self.last_action = action

        self.greedy_policy = np.argmax(self.Q, axis=1)

        self.history.register_Q(self.Q)
        self.history.register_policy(self.greedy_policy)

        return action


if __name__ == '__main__':

    import sys
    from matplotlib import pyplot as plt

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
        environment=env,
        eps=0.3,
        discount=1,
    )

    observation = env.reset()
    reward = 0
    done = False
    for step, _ in enumerate(tqdm(range(training_steps))):
        # render = env.render(mode='human')

        initial_learning_rate = 0.01

        # Minimum fraction of initial learning rate reached:
        min_fraction = 0.0005 / initial_learning_rate

        cosine_decay = 0.5 * (1 + np.cos(np.pi * step / (training_steps - 1)))
        decayed = (1 - min_fraction) * cosine_decay + min_fraction
        learning_rate = initial_learning_rate * decayed

        action = agent.train_step(observation, reward, done, learning_rate)

        if done:
            observation = env.reset()
            reward = 0
            done = False
            action = agent.train_step(observation, reward, done, learning_rate)

        observation, reward, done, info = env.step(action)

    env.close()

    print(agent.Q)
    print(agent.greedy_policy)

    with open('expected_sarsa_agent.dill', 'wb') as fopen:
        dill.dump(agent, fopen)

    # fig, ax = plt.subplots(2, 1, figsize=(10, 14))

    # ax[0].set_xlabel('Step')
    # ax[0].set_ylabel('Number of states where policy is non-optimal')
    # ax[0].plot(agent.history.policy_distances_to_optimal)

    # ax[1].set_xlabel('Step')
    # ax[1].set_ylabel('RMSE of the estimated state-values')
    # ax[1].plot(agent.history.V_RMSE)

    fig, ax = plt.subplots(2, 1, figsize=(10, 14))

    ax[0].set_xlabel('Episode')
    ax[0].set_ylabel('Number of states where policy is non-optimal')
    ax[0].plot([
        agent.history.policy_distances_to_optimal[i - 1]
        for i in agent.history.episode_ends
    ])

    ax[1].set_xlabel('Episode')
    ax[1].set_ylabel('RMSE of the estimated state-values')
    ax[1].plot([
        agent.history.V_RMSE[i - 1]
        for i in agent.history.episode_ends
    ])

    plt.show(block=True)
