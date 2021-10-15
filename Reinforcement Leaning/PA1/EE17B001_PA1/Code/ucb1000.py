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


# Function to implement UCB1 on 1000 arms
def ucb1000():
    q = np.random.normal(q_star, 1)    # All arms sampled once
    times_sampled = np.ones((1000,2000))    # Initialised at ones and not zeros because all arms have been picked once
    avg_reward = np.mean(q, axis = 1).tolist()
    optim_percent = []
    best_acts = np.argmax(q_star, axis = 0)
    for t in range(1000):
        act = np.argmax(q + np.sqrt(2*math.log(t+1000)/times_sampled), axis = 0)
        optim_percent.append(np.sum(act == best_acts)/2000*100)
        indices = (act, np.arange(2000))
        times_sampled[indices] = times_sampled[indices] + 1
        rt = np.random.normal(q_star[indices], 1)
        avg_reward.append(np.mean(rt))
        q[indices] = q[indices] + (rt - q[indices])/times_sampled[indices]    # Stochastic averaging update
    return q, avg_reward, optim_percent
    
    
# Running UCB1 on 1000 arms
start = time.time()
q1, avgr1, optim1 = ucb1000()
end = time.time()

plt.plot(avgr1)
plt.suptitle('UCB1 Average Reward over Time', fontweight = 'bold', fontsize = 16)
plt.xlabel('Steps \u279d', fontweight = 'bold')
plt.ylabel('Average Reward \u279d', fontweight = 'bold')
plt.grid(True)
plt.show()

plt.plot(optim1)
plt.suptitle('UCB1 % Optimal Pick over Time', fontweight = 'bold', fontsize = 16)
plt.xlabel('Steps \u279d', fontweight = 'bold')
plt.ylabel('% Optimal Action \u279d', fontweight = 'bold')
plt.grid(True)
plt.show()

ucb1000_time = end-start
print("Time taken for UCB1 (1000 arms) = "+str(ucb1000_time))
