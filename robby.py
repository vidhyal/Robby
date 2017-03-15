import numpy as np
import array
from operator import add
import time

states =  {0,1,2,3,4}  #{'N', 'S', 'E', 'W', 'C'}
actions = {0,1,2,3,4}  #{'N', 'S', 'E', 'W', 'P'}
rewards = [-5, -1, 10]
base3 = np.array([81,27,9,3,1])
epsilon = 0.1
sensors = [(-1,0),(1,0),(0,-1),(0,1),(0,0)]
step_cost = -0.05

gridStr = {1 : ' ', 2 : '*', 0 : '-'}

loc=()
grid =[]
grid1=[]

def generateGrid():
	n,m = 11,11
	global grid, grid1, loc
	grid = np.empty(shape = (n,m), dtype = 'object')
	grid1 = np.empty(shape = (n,m), dtype = 'object')
	grid.fill(1)
	grid1.fill(' ')
	for i in range(grid.shape[0]):
		for j in range(grid.shape[1]):
			if (np.random.uniform(0,1.0) < 0.1):
				grid[i,j] = 2
				grid1[i,j] = '*'

	for i in [0,n-1]:
		for j in range(grid.shape[1]):
			grid[i,j] = 0
			grid1[i,j] = '-'

	for i in [0,m-1]:
		for j in range(grid.shape[0]):
			grid[j,i] = 0
			grid1[j,i] = '|'
	
	loc = (np.random.randint(1,n-1), np.random.randint(1,m-1))
	print(loc)



def printGrid():
	prev = grid1[loc]
	grid1[loc] = '#'
	for i in range(grid.shape[0]):
		print(' '.join(grid1[i]))
	grid1[loc] = prev


def getState(loc, grid):
	state =0
	print(loc)
	for i in range(len(sensors)):
		inc = sensors[i]
		state += base3[i]*grid[loc[0]+inc[0], loc[1]+inc[1]]
	print(state)
	return state



def updateGrid(pos, val):
	grid[pos] = val
	grid1[pos] = gridStr[val]


def chooseAction(state):
	m = np.max(Q[state,:])
	best_actions = np.random.choice([i for i in range(len(Q[state,:])) if Q[state,i]==m])
	#print("ba = ", best_actions)
	action = np.random.choice([best_actions,np.random.choice(range(len(actions)))], p = [1-epsilon, epsilon] )
	#print(action)
	return action

def getReward(state, action):
	global loc
	print("loc =", loc)
	if action== 4:
		r = rewards[grid[loc]] + step_cost
		updateGrid(loc,1)
		
	else:
		new_loc = tuple(map(add, loc, sensors[action]))
		if (grid[new_loc] ==0):
			r = rewards[grid[new_loc]] +step_cost
		else:
			r = step_cost
			loc = new_loc
	print("new-loc = ", loc)	
	new_state = getState(loc, grid)		
	return new_state, r


Q = np.zeros(shape = (3**len(states), len(actions)))

def Q_Train(Q, eta =0.1, gamma = 0.9):
	reward_list = []
	for j in range(0,5000):
		generateGrid()
		
		episode_reward = 0
		for i in range(0,200):
			state = getState(loc, grid)
			action = chooseAction(state)#np.random.choice(actions)
			new_state, r = getReward(state, action)
			Q[state, action] += eta*(r+ gamma*np.argmax(Q[new_state,:]) - Q[state, action])
			print(state, "act = ", action, new_state, "rew = ", r)
			print("iter = "+ str(i) +" ep_rew = "+str(episode_reward))
			printGrid()
			#raw_input(" enter")
			time.sleep(.300)
			state = new_state
			episode_reward += r
	
			
		reward_list.append(episode_reward)
		print("rewards = " ,reward_list)
		print("episode = ", j)
        
	return Q


Q_Train(Q)