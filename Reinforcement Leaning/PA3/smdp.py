import gym
import numpy as np
import gym_four
import matplotlib.pyplot as plt
import matplotlib
import numpy.random as rndm


# Function to make an epsilon-greedy move
def eps_greedy(env, s, Q, eps=0.1):
	if rndm.uniform(low = 0, high = 1) < eps:
		a = env.action_space.sample()
	else:
		a = np.argmax(Q[:,s])
	return a

# Function to make a Q-Learning Update
def q_update(s, k, o, r, s_, Q, alpha, gamma):
	delta = r + gamma**k*np.max(Q[:,s_]) - Q[o,s]
	Q[o,s] = Q[o,s] + alpha*delta
	return Q

# Function to do a single run of Q-Learning
def q_learn(gamma, alpha, n_episodes, env, goal, eps=0.1):
	Q = rndm.rand(env.option_space.n, env.observation_space.n)
	goal_pos = env.encode(goal)
	steps_list = np.zeros([episodes])
	r_list = np.zeros([episodes])
	for n in range(episodes):
		s = env._reset()
		a = eps_greedy(env, s, Q, eps)
		k = 0
		for steps in range(1000):
			s_, r, done, boo = env.step(a)
			k+=1
			a_ = eps_greedy(env, s, Q, eps)
			if env.in_hallway_index(s_):
				Q = q_update(s, k, o, r, s_ , Q, alpha, gamma)
				k=0
				s = s_
			steps_list[n] += 1
			r_list[n] += r
			a = a_
			if (s == goal_pos): # If the goal state is reached:
				break
	return r_list, steps_list, Q

# Hyperparameters
gamma = 0.9
alpha = 0.1
eps = 0.1
episodes = 100

# Main Function to run the 50 parallel runs for both goals: G1 and G2
def main():

	goal_list = ['G1']#,'G2']
	goal_positions = [[1, [6, 2]]]#,[2, [1, 2]]]
	
	for ind, goalpos in enumerate(goal_positions):
		goal = goal_list[ind]
		print("SMDP Q-LEARNING: Learning for goal " + goal+"...")
		runs = np.array([(q_learn(gamma, alpha, episodes, gym.make('gym_four:FourRooms-v0'),goalpos,eps)) for i in range(50)])

		average_steps = []
		average_rewards = []
		averages = 0
		
		for i in range(len(runs)): # i iterates through all runs
			averages += runs[i][0]	# The [i][0] index of 'runs' contains the lists of total rewards at the end of each episode
		average_rewards = averages/len(runs)
		
		averages = 0
		for i in range(len(runs)):
			averages += runs[i][1]  # The [i][1] index of 'runs' contains the lists of steps of each episode
		average_steps = averages/len(runs)

		averages = 0
		for i in range(len(runs)):
			averages += runs[i][2]  # The [i][2] index of 'runs' contains the lists of Q-values after each episode (for each run i)
		average_Q = averages/len(runs)
		

	print(average_Q.shape)	
	""" # Selects the most frequently chosen action across all runs according to policy, in each state
		for i in range(12):
			for j in range(12):
				for k in range(len(runs)):
					acts_freq[np.argmax(runs[k][2], axis=0)[i,j]]+=1
				best_act = np.argmax(acts_freq)  # The most chosen action at the current state by the learned policies across runs
				acts_freq = np.zeros(4)
				pi[i,j] = best_act  # The best action is assigned to the policy at the current state """
		
if __name__== "__main__":
	main()

