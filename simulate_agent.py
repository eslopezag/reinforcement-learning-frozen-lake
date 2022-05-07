import sys
from time import sleep

from rl_agents.agents import load_agent


agent_folder_path = sys.argv[1]

agent = load_agent(agent_folder_path)

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
