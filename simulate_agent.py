import sys
import os
from time import sleep

from rl_agents.agents import load_agent


agent_name = sys.argv[1]
agent_folder_path = os.path.join('saved_agents', agent_name)
agent = load_agent(agent_folder_path)
env = agent.env
agent.set_mode('inference')

observation = env.reset()
done = False
terminate = False
for _ in range(100):
    render = env.render(mode='human')
    sleep(0.2)

    if not done:
        action = agent.get_action(observation)
        observation, reward, done, info = env.step(action)
    else:
        terminate = True

    if terminate:
        print(f'The episode has ended at state {observation}.')
        break

env.close()
