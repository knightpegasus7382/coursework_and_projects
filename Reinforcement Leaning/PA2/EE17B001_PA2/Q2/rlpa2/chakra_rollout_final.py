#!/usr/bin/env python

import click
import numpy as np
import numpy.linalg as npl
import gym
import chakra
import matplotlib.pyplot as plt


def chakra_get_action(theta, ob, rng=np.random):
	ob_1 = include_bias(ob)
	mean = np.matmul(theta, ob_1)
	return rng.normal(loc=mean, scale=1.)

# Function that adds a bias constant to the state, so that theta can be learnt along with a bias in linear parametrisation    
def include_bias(inp):
	return np.append(inp, 1)
	
# Function that gets the mean of the multivariate Gaussian action at a state
def get_action_mean(theta, ob):
	ob_1 = include_bias(ob)
	mean = np.matmul(theta, ob_1)	 
	return mean

# Function to calculate log of the policy at a state
def log_pi(ob, theta):
	s_ = np.expand_dims(include_bias(ob), axis = 1)
	mean = np.squeeze(np.matmul(theta, s_))
	cov = np.array([[1, 0], [0, 1]])
	a = np.squeeze(np.random.normal(loc=mean, scale = 1.))
	pi = 1/np.sqrt(2*np.pi)*np.exp(-1/2*np.matmul(np.transpose(a-mean), (a-mean)))
	logpi = np.log(pi)
	return logpi, a, mean

# Function to calculate gradient of log of the policy at a state    
def grad_log_pi(a, mean, ob):
	s_ = np.expand_dims(include_bias(ob), axis = 1)
	a = np.expand_dims(a, axis = 1)
	mean = np.expand_dims(mean, axis = 1)
	gradlogpi = np.matmul((a-mean),np.transpose(s_))
	return gradlogpi

# Function to normalise gradient	
def norm_grad(gradlogpi):
	gradlogpi = gradlogpi / (npl.norm(gradlogpi) + 1e-8)
	return gradlogpi

# Function to run an episode
def episode(env, theta, ob, rng, get_action, batch_size = 50, T= 100, gamma=0.9):
	done = False
	start_ob = ob  # Store the start state of 
	gradlogpi_f = 0
	trajrews = []

	for n_sample in range(batch_size):  # Outer loop over batch size
		ob = start_ob
		trans_list = []  # List to store transitions
		mu_list = []  # List to store means of actions
		t=0
		traj_rew = 0

		for t in range(T):  # Inner loop over time steps of a trajectory
			done = False
			logpi, a, mu = log_pi(ob, theta)
			if npl.norm(a) > 0.025:  # Clip the actions at 0.025 if they are over 0.025 in norm
				mu = mu / npl.norm(a) * 0.025
				a = a / npl.norm(a) * 0.025
			mu_list.append(mu)
			next_ob, r, done, _ = env.step(a)
			trans_list.append((ob, a, r))
			traj_rew += r
			if done:  # if the trajectory is done:
				break
			ob = next_ob
       
		for ind, trans in enumerate(trans_list):
			st, act, r = trans
			mu = mu_list[ind]
			discount_return = 0
			for k in range(ind, len(trans_list)):  # Calculating the cumulative discounted return for the transition
				gamma_pow = np.power(gamma, k - ind)
				s_k, a_k, r_k = trans_list[k]
				discount_return += gamma_pow*r_k
			gradlogpi = grad_log_pi(act, mu, st)
			gradlogpi = norm_grad(gradlogpi)
			gradlogpi_f += gradlogpi*(discount_return-(-0.5))  # Policy gradient formula
		trajrews.append(traj_rew)
	avg_reward = sum(trajrews)/batch_size    
	done = True
	return gradlogpi_f/batch_size, avg_reward	

# Hyperparameters
alpha = 0.03
batch_size = 200
max_itr = 200
T = 100 # Trajectory length
gamma = 0.95

@click.command()
@click.argument("env_id", type=str, default="chakra")
def main(env_id):
	rng = np.random.RandomState(42)
	thetas00, thetas01, thetas02, thetas10, thetas11, thetas12 = [], [], [], [], [], []
	if env_id == 'chakra':
		env = gym.make('chakra-v0')
		get_action = chakra_get_action
		obs_dim = env.observation_space.shape[0]
		action_dim = env.action_space.shape[0]
	else:
		raise ValueError("Unsupported environment: must be 'chakra' ")

	env.seed(42)
	# Initialize parameters
	theta = rng.normal(scale=0.01, size=(action_dim, obs_dim + 1))

	done = False
	rewards,episodes, traj_mn = [], [], []
	print("Itr | Average Trajectory Reward | Theta[0][0]")
	for itr in range(max_itr):
		ob = env.reset()
		grad, avg_reward = episode(env, theta, ob, rng, get_action, batch_size, T, gamma)
		gradstep = np.squeeze(np.array(grad))
		theta += alpha*gradstep  # Parameter update
		thetas00.append(theta[0][0])
		thetas01.append(theta[0][1])
		thetas02.append(theta[0][2])
		thetas10.append(theta[1][0])
		thetas11.append(theta[1][1])
		thetas12.append(theta[1][2])
		rewards.append(avg_reward)
		print(str(itr) + "\t" +  str(avg_reward) + "\t" + str(theta[0][0]))
    
    # Plotting the average trajetory reward and theta values vs episodes
	fig1 = plt.figure(figsize=(8,5))
	plt.plot(rewards, 'b', label = 'Average Trajectory Reward')
	plt.ylabel('Average Reward \u279d')
	plt.xlabel('Episodes \u279d')
	plt.legend(loc = 'lower right')
	plt.title('Average Trajectory Rewards across Batches vs Episodes')
	
	fig2 = plt.figure(figsize=(8,5))
	plt.plot(thetas00, 'g', label = 'theta[0][0]')
	plt.plot(thetas01, 'm', label = 'theta[0][1]')
	plt.plot(thetas02, 'k', label = 'theta[0][2]')
	plt.ylabel('theta[0][i] \u279d')
	plt.xlabel('Episodes \u279d')
	plt.legend(loc = 'lower right')
	plt.title('Coefficients of x component of parametrised action (theta [0][i] values) vs Episodes')
	
	fig3 = plt.figure(figsize=(8,5))
	plt.plot(thetas10, 'g', label = 'theta[1][0]')
	plt.plot(thetas11, 'm', label = 'theta[1][1]')
	plt.plot(thetas12, 'k', label = 'theta[1][2]')
	plt.ylabel('theta[1][i] \u279d')
	plt.xlabel('Episodes \u279d')
	plt.legend(loc = 'lower right')
	plt.title('Coefficients of y component of parametrised action (theta [1][i] values) vs Episodes')
	plt.show()
	          
	# Plotting the policy trajectories
	print('Plotting the policy trajectories...')          
	x=np.arange(-1,1.05,0.05)
	y=np.arange(-1,1.05,0.05)
	acts = np.array([0.025*get_action_mean(theta, [i,j])/npl.norm(get_action_mean(theta, [i,j])) if (npl.norm(get_action_mean(theta, [i,j]))>0.025) else get_action_mean(theta, [i,j]) for j in y for i in x])  # Clipping the action at 0.025 wherever needed
	ax = np.reshape(acts[:,0], (41,41))
	ay = np.reshape(acts[:,1], (41,41))
	plt.quiver(x, y, ax, ay, headwidth = 2, headlength = 2.5, headaxislength = 2.25)
	plt.title('Policy Trajectories for chakra')	
	plt.show()
    
    # Plotting the rough value function visualisation
	x=np.arange(-1,1.1,0.1)
	y=np.arange(-1,1.1,0.1)    
	N = 300
	v_s = np.zeros((len(y),len(x)))
	print('Plotting the value function visualisation...')
	for j_count, j in enumerate(y):
		for i_count, i in enumerate(x):  # Going through the whole grid
			ob = np.array([i,j])
			start_ob = ob
			gt_list = []
			for n in range(N):  # N trajectories from each state
				ob = start_ob
				gt = 0
				for t in range(T):
					done = False
					logpi, a, mu = log_pi(ob, theta)
					if npl.norm(a) > 0.025:
						mu = mu / npl.norm(a) * 0.025
						a = a / npl.norm(a) * 0.025
					next_ob, r, done, _ = env.step(a)
					gamma_pow = np.power(gamma, t)
					gt += gamma_pow*r
					if np.linalg.norm(next_ob) < 0.025:  # End the trajectory when the agent reaches close to the origin (to prevent accumuluting negative rewards)
						done = True
					if done:
						break
					ob = next_ob
				gt_list.append(gt)
			gt_avg = sum(gt_list)/N  # Averaging the returns (estimating the value function) at one point
			#print(gt_avg)
			v_s[-1-j_count, i_count] = gt_avg
	cp=plt.contourf(x,y,v_s, cmap=plt.cm.hot)
	plt.title('Value Function Visualisation')  
	plt.colorbar(format='%.3f')
	plt.show()
           
if __name__ == "__main__":
	main()    
