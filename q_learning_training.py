import numpy as np
import gym
from tqdm import tqdm
import dill

env = gym.make('FrozenLake-v1', desc=None, map_name='8x8', is_slippery=True)


class Agent:
    def __init__(
        self,
        environment: gym.Env,
        eps: float,
        discount: float,
        mode: str = 'training',
    ) -> None:
        self.env = environment
        self.Q = np.zeros(
            (self.env.observation_space.n, self.env.action_space.n)
        )
        self.greedy_policy = np.random.randint(
            0,
            self.env.action_space.n,
            self.env.observation_space.n,
        )
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
        self.terminal_states = set()

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

        if done:
            if state not in self.terminal_states:
                self.terminal_states.add(state)
                self.Q[state] = np.zeros(self.env.action_space.n)

        if self.last_state is not None and self.last_action is not None:
            self.Q[self.last_state, self.last_action] += step_size * (
                reward + self.discount * np.max(self.Q[state]) -
                self.Q[self.last_state, self.last_action]
            )

        if done:
            self.last_state = None
            self.last_action = None
        else:
            self.last_state = state
            self.last_action = action

        self.greedy_policy = np.argmax(self.Q, axis=1)

        return action


agent = Agent(
    environment=env,
    eps=1,
    discount=0.8,
)

if __name__ == '__main__':
    observation = env.reset()
    reward = 0
    done = False
    for _ in tqdm(range(30000000)):
        # render = env.render(mode='human')
        action = agent.train_step(observation, reward, done, 0.02)

        if done:
            observation = env.reset()
            reward = 0
            done = False
            action = agent.train_step(observation, reward, done, 0.02)

        observation, reward, done, info = env.step(action)

    env.close()

    print(agent.Q)
    print(agent.greedy_policy)

    with open('q_learning_agent.dill', 'wb') as fopen:
        dill.dump(agent, fopen)
