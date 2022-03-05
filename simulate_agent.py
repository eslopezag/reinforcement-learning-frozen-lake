import sys
import gym
import dill
from time import sleep


agent_file_path = sys.argv[1]

env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=True)

with open(agent_file_path, 'rb') as fopen:
    agent = dill.load(fopen)

agent.set_mode('inference')

observation = env.reset()
for _ in range(100):
    render = env.render(mode='human')
    sleep(0.2)
    action = agent.get_action(observation)
    observation, reward, done, info = env.step(action)

    if done:
        observation = env.reset()
env.close()
