from typing import List, Optional, Union, Callable
from itertools import chain

import numpy as np
import gym
from tqdm import tqdm
import dill
from matplotlib import pyplot as plt

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

        # Initialize the list that will keep a history of the rewards:
        self.reward_history = []

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

    def register_reward(self, reward: Union[int, float]):
        self.reward_history.append(reward)


class Agent:
    def __init__(
        self,
        training_alg: str,
        environment: gym.Env,
        initial_fill_value: float,
        discount: float,
        mode: str = 'training',
        eps: Optional[float] = None,
        output_filename: Optional[str] = None,
    ) -> None:
        if training_alg not in ('q_learning', 'sarsa', 'expected_sarsa'):
            raise ValueError(
                "The only values `training_alg` can take are 'q_learning', "
                "'sarsa', or 'expected_sarsa'."
            )
        self.training_alg = training_alg

        self.env = environment
        self.env._max_episode_steps = float('inf')

        self.eps = eps
        self.discount = discount

        self.output_filename = output_filename

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
            initial_fill_value,
            dtype='float64',
        )
        self.Q[self.terminal_states] = 0

        self.greedy_policy = np.argmax(self.Q, axis=1)

        self.history = AgentHistory(
            int(self.env.observation_space.n ** (1/2)),
            self.discount,
        )

        self.steps = 0
        self.episodes = 0

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

            if self.training_alg == 'sarsa':
                self.Q[self.last_state, self.last_action] += step_size * (
                    reward + self.discount * self.Q[state, action] -
                    self.Q[self.last_state, self.last_action]
                )

            elif self.training_alg == 'expected_sarsa':
                greedy_action = np.argmax(self.Q[state])

                actions_prob = np.zeros(self.env.action_space.n)
                actions_prob[greedy_action] = 1 - self.eps
                actions_prob += np.full(
                    self.env.action_space.n,
                    self.eps / self.env.action_space.n,
                )

                self.Q[self.last_state, self.last_action] += step_size * (
                    reward +
                    self.discount * np.dot(actions_prob, self.Q[state]) -
                    self.Q[self.last_state, self.last_action]
                )

            elif self.training_alg == 'q_learning':
                self.Q[self.last_state, self.last_action] += step_size * (
                    reward + self.discount * np.max(self.Q[state]) -
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

        self.history.register_reward(reward)
        self.history.register_Q(self.Q)
        self.history.register_policy(self.greedy_policy)

        return action

    def train(
        self,
        training_steps: int,
        step_size_scheduler: Callable,
        epsilon_scheduler: Optional[Callable] = None,
    ) -> None:
        observation = self.env.reset()
        reward = 0
        done = False
        self.episodes += 1
        for _ in tqdm(range(training_steps)):

            self.steps += 1

            step_size = step_size_scheduler(self.steps, self.episodes)

            if epsilon_scheduler:
                self.eps = epsilon_scheduler(self.steps, self.episodes)
            else:
                if self.eps is None:
                    raise ValueError(
                        'The agent\'s epsilon value can only be `None` if an '
                        '`epsilon_scheduler` is defined for training.'
                    )

            action = self.train_step(observation, reward, done, step_size)

            if done:
                self.episodes += 1
                observation = self.env.reset()
                reward = 0
                done = False
                action = self.train_step(
                    observation,
                    reward,
                    done,
                    step_size,
                )

            observation, reward, done, info = self.env.step(action)

        self.env.close()

        print(self.Q)
        print(self.greedy_policy)

        if not self.output_filename:
            self.output_filename = self.training_alg + '_agent'
        with open(f'{self.output_filename}.dill', 'wb') as fopen:
            dill.dump(self, fopen)

        fig, ax = plt.subplots(3, 1, figsize=(10, 21))

        ax[0].set_xlabel('Episode')
        ax[0].set_ylabel('Number of states where policy is non-optimal')
        episode_policy_distances_to_optimal = [
            self.history.policy_distances_to_optimal[i - 1]
            for i in self.history.episode_ends
        ]
        ax[0].plot(
            range(1, len(episode_policy_distances_to_optimal) + 1),
            episode_policy_distances_to_optimal,
        )

        ax[1].set_xlabel('Episode')
        ax[1].set_ylabel('RMSE of the estimated state-values')
        episode_V_RMSE = [
            self.history.V_RMSE[i - 1] for i in self.history.episode_ends
        ]
        ax[1].plot(range(1, len(episode_V_RMSE) + 1), episode_V_RMSE)

        ax[2].set_xlabel('Episode')
        ax[2].set_ylabel(
            'Running average of episode reward (window size = 100)'
        )
        episode_rewards = [
            sum(self.history.reward_history[i:j])
            for i, j in zip(
                chain([0], self.history.episode_ends),
                self.history.episode_ends,
            )
        ]
        cumsum = np.cumsum(np.insert(episode_rewards, 0, 0))
        running_avg = (cumsum[100:] - cumsum[:-100]) / 100
        ax[2].plot(range(100, len(running_avg) + 100), running_avg)

        plt.show(block=True)
