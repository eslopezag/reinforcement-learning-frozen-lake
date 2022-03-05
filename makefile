train_sarsa:
	python sarsa_training.py
train_q_learning:
	python q_learning_training.py
simulate_sarsa:
	python simulate_agent.py "sarsa_agent.dill"
simulate_q_learning:
	python simulate_agent.py "q_learning_agent.dill"
simulate_random_policy:
	python simulate_random_policy.py
