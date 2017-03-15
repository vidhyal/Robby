import numpy as np

epsilon = 0.1
Q = np.array([[1,2,4,3,1,3],[3,4,5,3,5,5],[1,4,2,7,3,4]])
actions = {0,1,2,3,4}

def chooseAction(state):
	m = np.max(Q[state,:])
	best_actions = [i for i in range(len(Q[state,:])) if Q[state,i]==m]
	print(best_actions)
	action = np.random.choice(np.random.choice([best_actions,range(len(actions))], p = [1-epsilon, epsilon] ))
	print(action)
	return action

chooseAction(1)