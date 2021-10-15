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

# Parellelly setting up the means of the distributions for 2000 10-armed bandits
q_star = np.random.randn(10, 2000)


# Function to implement Median Elimination
# Here, the matrix of rewards and runs has been used in the transposed form, because it is easier to reshape the matrix into the form that we need (2000 runs * half of the actions) after the lower half of arms has been parallelly removed from each of the 2000 bandits
def median_elim(epsilon, delta):
    eps_l = epsilon/4
    del_l = delta/2
    avg_reward = []
    q = np.zeros((2000,10))
    best_q_star = np.transpose(q_star)
    count = 0
    while True:
        times = (4/eps_l**2) * math.log(3/del_l)
        c = 0
        for _ in range(math.floor(times)):
            c = c + 1
            rt = np.random.normal(best_q_star, 1)
            q = q + (rt-q)/c    # Stochastic averaging update: each time step is treated as one pull of one arm
            avg_reward.append(np.mean(rt, axis = 0).tolist())
        if q.shape[1] == 1:
            break
        med = np.median(q, axis = 1)
        indices = np.transpose(np.transpose(q)>=med)    # Lower half of arms removed, but all the remaining arms of all bandits are in a 1-D array now
        best_q_star = best_q_star[indices].reshape(-1,math.ceil(q.shape[1]/2))    
        q = q[indices].reshape(-1,math.ceil(q.shape[1]/2))    # Reshaping the matrix into (no. of bandits * half of the arms)
        eps_l = 0.75*eps_l
        del_l = 0.5*del_l
        print("Round " + str(count) + " done...")
    avg_reward = [reward for sublist in avg_reward for reward in sublist]   # Flattening list of average rewards
    return best_q_star, avg_reward
    
    
# Running Median Elimination for epsilon and delta = 0.1
start = time.time()
best_q, avgr = median_elim(epsilon = 0.1, delta = 0.1)
end = time.time()

plt.plot(avgr, label = "$\epsilon$ = 0.1, $\delta$ = 0.1")
plt.legend(loc = "lower right")
plt.suptitle('Median Elimination Avg Reward over Time for $\epsilon = 0.1$ and $\delta = 0.1$', fontweight = 'bold')
plt.xlabel('Steps \u279d', fontweight = 'bold')
plt.ylabel('Average Reward \u279d', fontweight = 'bold')
plt.grid(True)
plt.show()

me_time = end-start
print("Time taken for Median Elimination (10 arms) = "+str(me_time))
