import sys
import gym
import dill
from time import sleep


agent_file_path = sys.argv[1]

with open(agent_file_path, 'rb') as fopen:
    agent = dill.load(fopen)

env = agent.env

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
