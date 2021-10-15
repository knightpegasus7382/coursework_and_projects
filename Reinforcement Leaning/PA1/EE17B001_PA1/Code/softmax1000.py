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


# Function to implement softmax on 1000 arms
def softmax1000(temp):
    q = np.zeros((1000,2000))    # Estimates initialised at 0 for values
    times_sampled = np.zeros((1000,2000))    # Keeps track of number of times each arm has been sampled
    avg_reward = []
    optim_percent = []
    best_acts = np.argmax(q_star, axis = 0)
    for t in range(1000):
        gibbs = np.exp(q/temp)/np.sum(np.exp(q/temp), axis = 0)
        gibbs = np.cumsum(gibbs, axis = 0)
        sampler = np.random.uniform(0,1,2000)
        act_bool = sampler < gibbs
        act = np.argmax(act_bool, axis = 0)    # Using a uniform distribution and the cumulative probability values to sample an action
        optim_percent.append(np.sum(act == best_acts)/2000*100)
        indices = (act, np.arange(2000))
        times_sampled[indices] = times_sampled[indices] + 1
        rt = np.random.normal(q_star[indices], 1)
        avg_reward.append(np.mean(rt))
        q[indices] = q[indices] + (rt - q[indices])/times_sampled[indices]    # Stochastic averaging update
    return q, avg_reward, optim_percent
    
    
# Running softmax with various temperature values on 1000 arms
q1, avgr1, optim1 = softmax1000(temp = 0.01)
start = time.time()
q2, avgr2, optim2 = softmax1000(temp = 0.1)
end = time.time()
q3, avgr3, optim3 = softmax1000(temp = 1)
q4, avgr4, optim4 = softmax1000(temp = 5)

plt.plot(avgr1, label = '$\\beta$ = 0.01')
plt.plot(avgr2, label = '$\\beta$ = 0.1')
plt.plot(avgr3, label = '$\\beta$ = 1')
plt.plot(avgr4, label = '$\\beta$ = 5')
plt.legend(loc='lower right')
plt.suptitle('Softmax Average Reward over Time for Varying Temperature', fontweight = 'bold')
plt.xlabel('Steps \u279d', fontweight = 'bold')
plt.ylabel('Average Reward \u279d', fontweight = 'bold')
plt.grid(True)
plt.show()

plt.plot(optim1, label = '$\\beta$ = 0.01')
plt.plot(optim2, label = '$\\beta$ = 0.1')
plt.plot(optim3, label = '$\\beta$ = 1')
plt.plot(optim4, label = '$\\beta$ = 5')
plt.legend(loc='lower right')
plt.suptitle('Softmax % Optimal Pick over Time for Varying Temperature', fontweight = 'bold')
plt.xlabel('Steps \u279d', fontweight = 'bold')
plt.ylabel('% Optimal Action \u279d', fontweight = 'bold')
plt.grid(True)
plt.show()

softmax1000_time = end-start
print("Time taken for softmax (1000 arms) = "+str(softmax1000_time))
