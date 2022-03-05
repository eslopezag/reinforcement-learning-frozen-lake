import gym
from time import sleep

env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=True)

observation = env.reset()
for _ in range(100):
    sleep(0.2)
    render = env.render(mode='human')
    action = env.action_space.sample()  # uniformly random actions
    observation, reward, done, info = env.step(action)

    if done:
        observation = env.reset()
env.close()
