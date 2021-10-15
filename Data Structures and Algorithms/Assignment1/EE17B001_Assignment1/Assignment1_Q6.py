from random import choices
import matplotlib.pyplot as plt
import numpy as np

from math import factorial

def permutation(d, n):
    return factorial(d)//factorial(d-n)                     # Flooor division used to prevent overflow of result of division for large integers

# Probability is estimated by taking the fraction of occurrence of common birthdays in 20000 sampled sequences of n birthdays
samples = 20000

xaxis_sim = []
yaxis_sim = []                                             # List to store values of experimental probabilities from simulation
yaxis_exp = []                                             # List to store the theoretical/expected/calculated probabilities

net_error = 0                                              # Variable to store the sum of squared errors (L2 error) between simulated and theoretical probabilities

for n in range(5, 201, 5):
    count = 0
    for i in range(samples):
        bdays = np.random.randint(1, 365, size = n)        # Generating a sequence of sampled birthdays (from 365 days of the year) with replacement
        if len(bdays) != len(set(bdays)):                  # We increment count each time the sequence of birthdays has more elements than number of unique elements (i.e, there is atleast 1 repeated birthday)
            count +=1
    prob = count/samples                                   # Calculation of empirical / sample probability
    xaxis_sim.append(n)
    yaxis_sim.append(prob)                                 # Appending experimental / simulated probability
    calc_prob = 1 - permutation(365, n)/365**n
    yaxis_exp.append(calc_prob)                            # Appending expected probability
    net_error += (prob-calc_prob)**2                       # Adding squared error

print("Sum squared error = ", net_error)                   # Printing sum squared error to console

# Plotting the expected and simulation curves for n = 5, 10, 15 ... 200
plt.figure(figsize = (12,9))
plt.plot(xaxis_sim, yaxis_sim, 'go-', label = "Simulation probabilities")
plt.plot(xaxis_sim, yaxis_exp, 'r--', label = "Expected probabilities")
plt.xlabel("No of birthdays n \u279d", fontsize = 12, fontweight = 'bold')
plt.ylabel("Probability \u279d",  fontsize = 12, fontweight = 'bold')
plt.title("Plot for the Probability of Occurrence of a Pair of Common Birthdays vs Number of Birthdays Sampled", fontsize = 12, fontweight = 'bold')
plt.legend()
plt.show()