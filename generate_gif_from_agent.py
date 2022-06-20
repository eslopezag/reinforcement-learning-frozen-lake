import sys
import os
from PIL import Image

from rl_agents.agents import load_agent

agent_name = sys.argv[1]
agent_folder_path = os.path.join('saved_agents', agent_name)
agent = load_agent(agent_folder_path)
env = agent.env
agent.set_mode('inference')

frames = []
observation = env.reset()
done = False
terminate = False
for i in range(200):
    render = env.render(mode='rgb_array')
    image = Image.fromarray(render)
    # image.save(f'gifs/{agent_name}_img_{i:0>3}.png')
    frames.append(image)

    if not done:
        action = agent.get_action(observation)
        observation, reward, done, info = env.step(action)
    else:
        terminate = True

    if terminate:
        print(f'The episode has ended at state {observation}.')
        print(f'The GIF will have {len(frames)} frames.')
        break

env.close()

frames[0].save(
    f'gifs/{agent_name}.gif',
    format='GIF',
    append_images=frames[1:],
    save_all=True,
    duration=200,  # Each frame will be shown for 200 milliseconds
    #  Not specifying the `loop` parameter means the GIF will only play once
)
