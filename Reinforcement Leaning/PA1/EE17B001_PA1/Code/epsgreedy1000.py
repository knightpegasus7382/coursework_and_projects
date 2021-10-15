########################################
#       PROGRAMMING ASSIGNMENT 1       #
########################################
# Author: Akash Reddy A                #
# Roll No: EE17B001                    #
########################################

# Importing required packages
import numpy as np
import math
import matplotlib.pyplot as plt
import time


# Parellelly setting up the means of the distributions for 2000 1000-armed bandits
q_star = np.random.randn(1000, 2000)


# Function to implement epsilon-greedy for 1000 arms
def eps_greedy1000(epsilon):
    q = np.zeros((1000,2000))    # Estimates initialised at 0 for values
    times_sampled = np.zeros((1000,2000))    # Keeps track of number of times each arm has been sampled
    avg_reward = []    # List to store average rewards
    optim_percent = []    # List to store percentage optimum pick
    best_acts = np.argmax(q_star, axis = 0)
    for t in range(1000):
        sampler = np.random.uniform(0,1,2000)
        greedy_bool = sampler>epsilon
        act = np.argmax(q, axis = 0)*greedy_bool + np.random.randint(0,1000,2000)*(np.invert(greedy_bool))   # Epsilon-greedily picking an action
        optim_percent.append(np.sum(act == best_acts)/2000*100)
        indices = (act, np.arange(2000))
        times_sampled[indices] = times_sampled[indices] + 1
        rt = np.random.normal(q_star[indices], 1)   # Evaluating the reward of the epsilon-greedy action
        avg_reward.append(np.mean(rt))
        q[indices] = q[indices] + (rt - q[indices])/times_sampled[indices]    # Stochastic averaging update of values
    return q, avg_reward, optim_percent
    
    
# Running epsilon-greedy on 1000 arms
q1, avgr1, optim1 = eps_greedy1000(epsilon=0)
q2, avgr2, optim2 = eps_greedy1000(epsilon=0.01)
start = time.time()
q3, avgr3, optim3 = eps_greedy1000(epsilon=0.1)
end = time.time()

plt.plot(avgr1, label = '$\epsilon$ = 0 (greedy)')
plt.plot(avgr2, label = '$\epsilon$ = 0.01')
plt.plot(avgr3, label = '$\epsilon$ = 0.1')
plt.legend(loc='lower right')
plt.suptitle('$\epsilon$-greedy Average Reward over Time', fontweight = 'bold', fontsize = 14)
plt.xlabel('Steps \u279d', fontweight = 'bold')
plt.ylabel('Average Reward \u279d', fontweight = 'bold')
plt.grid(True)
plt.show()

plt.plot(optim1, label = '$\epsilon$ = 0 (greedy)')
plt.plot(optim2, label = '$\epsilon$ = 0.01')
plt.plot(optim3, label = '$\epsilon$ = 0.1')
plt.legend(loc='lower right')
plt.suptitle('$\epsilon$-greedy % Optimal Pick over Time', fontweight = 'bold', fontsize = 14)
plt.xlabel('Steps \u279d', fontweight = 'bold')
plt.ylabel('% Optimal Action \u279d', fontweight = 'bold')
plt.grid(True)
plt.show()

eps_greedy1000_time = end-start
print("Time taken for eps-greedy (1000 arms) = "+str(eps_greedy1000_time))
