from typing import Optional
from itertools import chain

import numpy as np
from matplotlib import pyplot as plt
import gym

from q_approximators import TabularQApproximator
from analytical_optimal_policy import AnalyticalOptimalPolicyFinder
from agent_history import TabularAgentHistory


class CustomAgentHistory(TabularAgentHistory):
    def __init__(
        self,
        discount: float,
        *,
        env: Optional[gym.Env] = None,
        n_states: Optional[int] = None,
    ) -> None:
        super().__init__()

        if env is not None:
            self.n_states = env.observation_space.n
        elif n_states is not None:
            self.n_states = n_states
        else:
            raise ValueError(
                'Either `env` or `n_states` must be specified.'
            )

        apf = AnalyticalOptimalPolicyFinder(
            int(self.n_states ** (1/2)),
            discount,
        )
        initial_policy = [2 for _ in range(self.n_states)]
        self.optimal_policies, self.optimal_action_values = (
            apf.policy_iteration(initial_policy)
        )

        self.optimal_state_values = np.max(self.optimal_action_values, axis=1)

        # Initialize the list that will contain the average of the RMSE across
        # the states:
        self.V_RMSE = []

        # Initialize the list that will contain the distance of the policy to
        # the optimal one, found as the number of states for which the action
        # is non-optimal:
        self.policy_distances_to_optimal = []

    def register_Q(self, Q: TabularQApproximator) -> None:
        V = np.max(Q, axis=1)  # get the state-values assuming greedy policy
        self.V_RMSE.append(
            np.sqrt(np.square(self.optimal_state_values - V).mean())
        )

        policy = np.argmax(Q, axis=1)

        self.policy_distances_to_optimal.append(
            sum(
                a not in optimal_action
                for a, optimal_action in zip(policy, self.optimal_policies)
            )
        )

    def show_training_results(self) -> None:
        fig, ax = plt.subplots(3, 1, figsize=(10, 21))

        ax[0].set_xlabel('Episode')
        ax[0].set_ylabel('Number of states where policy is non-optimal')
        episode_policy_distances_to_optimal = [
            self.policy_distances_to_optimal[i]
            for i in self.episode_ends
        ]
        ax[0].plot(
            range(len(episode_policy_distances_to_optimal)),
            episode_policy_distances_to_optimal,
        )

        ax[1].set_xlabel('Episode')
        ax[1].set_ylabel('RMSE of the estimated state-values')
        episode_V_RMSE = [
            self.V_RMSE[i] for i in self.episode_ends
        ]
        ax[1].plot(range(len(episode_V_RMSE)), episode_V_RMSE)

        ax[2].set_xlabel('Episode')
        ax[2].set_ylabel(
            'Running average of episode reward (window size = 100)'
        )
        episode_rewards = [
            sum(r for r in self.reward_history[i + 1: j + 1] if r is not None)
            for i, j in zip(
                chain([-1], self.episode_ends),
                self.episode_ends,
            )
        ]
        cumsum = np.cumsum(np.insert(episode_rewards, 0, 0))
        running_avg = (cumsum[100:] - cumsum[:-100]) / 100
        ax[2].plot(range(100, len(running_avg) + 100), running_avg)

        plt.show(block=True)
