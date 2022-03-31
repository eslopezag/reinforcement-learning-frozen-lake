from time import sleep
import sys

import gym


map_size = sys.argv[1]

if map_size == '4':
    map_name = '4x4'
elif map_size == '8':
    map_name = '8x8'

env = gym.make('FrozenLake-v1', desc=None, map_name=map_name, is_slippery=True)

observation = env.reset()
for _ in range(100):
    sleep(0.2)
    render = env.render(mode='human')
    action = env.action_space.sample()  # uniformly random actions
    observation, reward, done, info = env.step(action)

    if done:
        observation = env.reset()
env.close()
