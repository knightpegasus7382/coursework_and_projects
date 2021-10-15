import numpy as np
import gym
import gym_puddle
import matplotlib.pyplot as plt
import matplotlib
import numpy.random as rndm

# Function to make an epsilon-greedy move
def eps_greedy(env, eps, s, Q):
	if rndm.uniform(0,1) < eps:
		a = env.random_act()
	else:
		a = np.argmax(Q[:,s[0],s[1]])
	return a
	
# Function to make a SARSA Update
def sarsa_update(s, a, r, s_, a_, Q, alpha, gamma):
	delta = r + gamma*Q[a_,s_[0],s_[1]] - Q[a,s[0],s[1]]
	Q[a,s[0],s[1]] = Q[a,s[0],s[1]] + alpha*delta
	return Q

# Function to do a single run of SARSA
def sarsa_learn(gamma, alpha, eps, n_episodes, env, goal):
	Q = rndm.rand(env.action_space.n, env.observation_space.shape[0], env.observation_space.shape[1])
	goal_pos = env.set_goal(goal)
	steps_list = np.zeros([episodes])
	r_list = np.zeros([episodes])
	for n in range(episodes):
		s = env.reset()
		a = eps_greedy(env, eps, s, Q)
		for steps in range(1000):
			s_, r = env.step(s, a)
			a_ = eps_greedy(env, eps, s_, Q)
			Q = sarsa_update(s, a, r, s_, a_, Q, alpha, gamma)
			steps_list[n] += 1
			r_list[n] += r
			s = s_
			a = a_
			if (s == goal_pos).all():
				break
	return r_list, steps_list, Q

# Function to plot the average reward and average steps plots, and policy	    
def plots(avg_r, avg_steps, episodes, goal, goalpos, pi):
	fig1=plt.figure(figsize=(8,5))
	plt.plot(range(episodes), avg_r, 'b' , label = "Average Reward" )
	plt.title('SARSA: Average reward vs Episodes: Goal '+ goal)
	plt.ylabel('Average Reward \u279d')
	plt.xlabel('Episodes \u279d')
	plt.legend(loc = 'lower right')
	fig2=plt.figure(figsize=(8,5))
	plt.plot(range(episodes), avg_steps, 'r', label = "Steps")
	plt.title('SARSA: Average steps vs Episodes: Goal '+ goal)
	plt.ylabel('Steps \u279d')
	plt.xlabel('Episodes \u279d')
	plt.legend(loc = 'upper right')
	#plt.show()
	plt.rcParams['figure.figsize'] = [6,6]
	fig, ax = plt.subplots()
	ax.set_title('SARSA: Optimal Policy for Goal '+goal)
	for i in range(12):
		for j in range(12):
			if [j,i] == goalpos:
				ax.text(i, j, goal, va='center', ha='center')
			else:
				c = int(pi[j,i])
				direcs = ['↑', '➜', '←', '↓']
				ax.text(i, j, direcs[c], va='center', ha='center')
	ax.matshow(pi, cmap = plt.cm.Pastel1)
	plt.show()

# Hyperparameters
gamma = 0.9
alpha = 0.1
eps = 0.01
episodes = 500

# Main Function to run the 50 parallel runs for all goals: A, B, and C
def main():

	goal_list = ['A','B','C']
	goal_positions = [[0,11],[2,9],[7,8]]
	
	for ind, goalpos in enumerate(goal_positions):
		goal = goal_list[ind]
		print("SARSA: Learning for goal " + goal+"...")
		runs = np.array([(sarsa_learn(gamma, alpha, eps, episodes, gym.make('gym_puddle:puddle-v0'),goal)) for i in range(50)])

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
		
		pi = np.zeros([12,12]) # Policy at each cell initialised to 0

		acts_freq = np.zeros(4)  # Used to store the frequency of each action across all runs according to policy

        # Selects the most frequently chosen action across all runs according to policy, in each state
		for i in range(12):
			for j in range(12):
				for k in range(len(runs)):
					acts_freq[np.argmax(runs[k][2], axis=0)[i,j]]+=1
				best_act = np.argmax(acts_freq)  # The most chosen action at the current state by the learned policies across runs
				acts_freq = np.zeros(4)
				pi[i,j] = best_act  # The best action is assigned to the policy at the current state

        # Plot all required plots
		plots(average_rewards, average_steps, episodes, goal, goalpos, pi)
		
if __name__== "__main__":
	main()
