"""
This script finds the optimal policy analytically
"""

from typing import List, Tuple

import numpy as np


class AnalyticalOptimalPolicyFinder():
    def __init__(self, map_size: int, discount: float) -> None:
        self.map_size = map_size
        self.discount = discount

        if self.map_size == 4:
            self.map = [
                "SFFF",
                "FHFH",
                "FFFH",
                "HFFG"
            ]
        elif self.map_size == 8:
            self.map = [
                "SFFFFFFF",
                "FFFFFFFF",
                "FFFHFFFF",
                "FFFFFHFF",
                "FFFHFFFF",
                "FHHFFFHF",
                "FHFFHFHF",
                "FFFHFFFG",
            ]

    def matrix_idx(self, list_idx: int) -> Tuple[int]:
        """
        Converts a state from list index to matrix indices.
        """
        return divmod(list_idx, self.map_size)

    def list_idx(self, matrix_idx: Tuple[int]) -> int:
        """
        Converts a state from matrix indices to list index.
        """
        return matrix_idx[0] * self.map_size + matrix_idx[1]

    def move(self, i: int, j: int, direction: str) -> Tuple[int]:
        """
        Deterministic movement function.
        """
        if direction == 'left':
            j = max(0, j - 1)
        elif direction == 'down':
            i = min(self.map_size - 1, i + 1)
        elif direction == 'right':
            j = min(self.map_size - 1, j + 1)
        elif direction == 'up':
            i = max(0, i - 1)

        return i, j

    def transition(self, i: int, j: int, action: int) -> List[Tuple[int]]:
        """
        Transition function given the matrix indices. Returns the list of all
        possible next states.
        """
        if action == 0:  # left
            possible_directions = ('left', 'up', 'down')
        elif action == 1:  # down
            possible_directions = ('down', 'left', 'right')
        elif action == 2:  # right
            possible_directions = ('right', 'up', 'down')
        elif action == 3:  # up
            possible_directions = ('up', 'left', 'right')

        next_states = [self.move(i, j, dir) for dir in possible_directions]
        return next_states

    def policy_evaluation(self, policy: List[int]) -> List[float]:
        # Initialize the matrix and vector that will be used to solve the
        # system of Bellman equations:
        M = np.zeros((self.map_size ** 2, self.map_size ** 2))
        b = np.zeros(self.map_size ** 2)

        # Fill M and b:
        for s in range(self.map_size ** 2):
            i, j = self.matrix_idx(s)
            M[s, s] += 1
            if self.map[i][j] not in ('H', 'G'):  # if state is not terminal
                for next_s in map(
                    self.list_idx, self.transition(i, j, policy[s])
                ):
                    M[s, next_s] -= self.discount / 3
                    if next_s == self.map_size ** 2 - 1:  # next state is goal
                        b[s] += 1 / 3

        # Solve the system of Bellman equations to find the state-values
        state_values = np.linalg.solve(M, b)
        return state_values

    def V_to_Q(self, state_values: List[float]) -> np.ndarray:
        """
        State-values to Action-values.
        """
        # Initialize the action values array:
        action_values = np.zeros((self.map_size ** 2, 4))

        for s in range(self.map_size ** 2):
            i, j = self.matrix_idx(s)
            if self.map[i][j] not in ('H', 'G'):  # if state is not terminal
                for action in range(4):
                    for next_s in map(
                        self.list_idx, self.transition(i, j, action)
                    ):
                        # Reward:
                        R = 1 if next_s == (self.map_size ** 2 - 1) else 0

                        action_values[s, action] += (
                            R + self.discount * state_values[next_s]
                        ) / 3

        return action_values

    def policy_iteration(self, initial_policy: List[int]) -> List[int]:
        policy = None
        new_policy = initial_policy
        while policy != new_policy:
            policy = new_policy
            state_values = self.policy_evaluation(policy)
            action_values = self.V_to_Q(state_values)
            new_policy = np.argmax(action_values, axis=1).tolist()

        optimal_policies = [
            [i for i in range(4) if q[i] == np.max(q)]
            for q in action_values
        ]

        return optimal_policies, action_values


if __name__ == '__main__':
    import sys

    map_size, discount = sys.argv[1:3]

    map_size = int(map_size)
    discount = float(discount)

    apf = AnalyticalOptimalPolicyFinder(map_size, discount)

    print(apf.policy_iteration([2 for _ in range(map_size ** 2)]))
