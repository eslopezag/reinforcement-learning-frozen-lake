size = 4
steps = 200000

train_sarsa:
	python sarsa_training.py $(size) $(steps)
train_expected_sarsa:
	python expected_sarsa_training.py $(size) $(steps)
train_q_learning:
	python q_learning_training.py $(size) $(steps)
train_q_learning_eps:
	python q_learning_eps_training.py $(size) $(steps)
simulate_agent:
	python simulate_agent.py $(agent_name)
simulate_random_policy:
	python simulate_random_policy.py $(size)
generate_gif:
	python generate_gif_from_agent.py $(agent_name)
