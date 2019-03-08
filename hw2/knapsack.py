import numpy as np
import random
import math
import matplotlib.pyplot as plt

m = 5
w = 10

worth = np.array([6, 3, 5, 4, 6])
weight = np.array([2, 2, 6, 4, 5])

# helper function, note that state is represent as a interger, 
# we convert it to binary to indicate choice
def get_state(v):
    f = np.binary_repr(v, width=m)
    f = np.array(list(f)) == '1'
    return f

def state_to_int(f):
    f = f.astype(np.int32)
    return f.dot(2**np.arange(f.size)[::-1])

def get_value_and_weight(f):
    current_worth = np.sum(worth[f])
    current_weight = np.sum(weight[f])
    return current_worth, current_weight


# brute force solution
max_worth = 0
best_solution = None
for i in range(2**m):
    f = get_state(i)
    current_worth, current_weight = get_value_and_weight(f)
    if current_worth > max_worth and current_weight <= w:
        max_worth = current_worth
        best_solution = f

print('optimal solution found by brute force search:', max_worth)
print(best_solution)

# MCMC solution
trial = 10000
beta = 1
init = 0
states = [init]

for i in range(trial):
    state = states[-1]
    f = get_state(state)
    current_worth, _ = get_value_and_weight(f)

    # choose a treasure uniformly from f and replace it
    index = random.sample(range(m), 1)[0]
    f[index] = 1 - f[index]

    # calculate worth and weight of new choice
    new_worth, new_weight = get_value_and_weight(f)
    new_state = state_to_int(f)

    # if new choice is invalid, stay in same state
    if new_weight > w:
        states.append(state)
    else:
        # jump to new state with probability min(1, e^(beta(worth1, worth2)))
        p = min(1, math.exp(beta * (new_worth - current_worth)))
        new = np.random.choice([state, new_state], p=[1-p, p])
        states.append(new)

unique, counts = np.unique(states, return_counts=True)

state = [ get_state(l) for l in unique]
worths = [get_value_and_weight(f)[0] for f in state]
counts = np.array(counts) / np.sum(counts)
print('MCMC simulation results, choice, worth, probability:')
print(np.asarray((state, worths, counts)).T)

# hist and convergence
plt.plot(unique, counts)
plt.title("probability distribution in states")
plt.xlabel("state represent in interger")
plt.ylabel("probability")
plt.savefig("knapsack_distribution.png", dpi=300)
plt.show()

# convergence rate
error = []
step = trial // 1000
for i in range(1000):
    end = step * (i+1)
    # select the best solution for each break points 
    unique, counts = np.unique(states[:end], return_counts=True)
    best_solution = unique[np.array(counts).argmax()]
    best_worth = get_value_and_weight(get_state(best_solution))[0]
    error.append(best_worth)

plt.plot(range(1000), (max_worth-np.array(error))/max_worth)
plt.title("convergence rate")
plt.xlabel("iteration (* %d)"%(step))
plt.ylabel("relative approximation error")
plt.savefig("knapsack_convergence_rate.png", dpi=300)
plt.show()
