import numpy as np
import random
import math
import matplotlib.pyplot as plt

trial = 100000

init = 0.0
states = [init]

for i in range(trial):
    state = states[-1]
    # sample from uniform(state-1, state+1)
    y = random.uniform(state - 1.0, state + 1.0)
    # accept the proposal with probability min(1.0, e^(-0.5*(y^2 - s^2)))
    p = min(1.0, math.exp(-0.5 * (y**2 - state**2)))
    new = np.random.choice([state, y], p=[1-p, p])

    states.append(new)

plt.hist(states, 50, density=True)
plt.title('simulation results of standard normal distribution')
plt.savefig('standard_normal_distribution.png', dpi=300)
plt.show()

means, variances = [], []
step = trial // 1000
states = np.array(states)
for i in range(1000):
    end = step * (i+1)
    mean = np.mean(states[:end])
    variance = np.var(states[:end])
    means.append(mean)
    variances.append(variance)
means = np.array(means)
variances = np.array(variances)

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.plot(range(1000), np.abs(0.0-means))
plt.title('convergence rate of mean')
plt.xlabel("iteration (* %d)"%(step))
plt.ylabel("approximation error")

plt.subplot(122)
plt.plot(range(1000), np.abs(1.0-variances))
plt.title("convergence rate of variance")
plt.xlabel("iteration (* %d)"%(step))

plt.ylabel("approximation error")
plt.savefig("standard_normal_distribution_convergence_rate.png", dpi=300)
plt.show()

