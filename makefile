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
simulate_sarsa:
	python simulate_agent.py "saved_agents/sarsa_agent"
simulate_expected_sarsa:
	python simulate_agent.py "saved_agents/expected_sarsa_agent"
simulate_q_learning:
	python simulate_agent.py "saved_agents/q_learning_agent"
simulate_q_learning_eps:
	python simulate_agent.py "saved_agents/q_learning_eps_agent"
simulate_random_policy:
	python simulate_random_policy.py $(size)
