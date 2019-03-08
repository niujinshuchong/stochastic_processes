import numpy as np
import random
import math
import matplotlib.pyplot as plt

trial = 100000
a = 5.0
b = 5.0

init = 0.5
states = [init]

for i in range(trial):
    s = states[-1]
    # sample from uniform(0.0, 1.0)
    y = random.uniform(0.0, 1.0)
    # accept the proposal with probability 
    # min(1.0, (y^(a-1) * (1.0-y)^(b-1)) / (s^(a-1) * (1-s)^(b-1)) )
    p = min(1.0, (y**(a-1) * (1.0-y)**(b-1)) / (s**(a-1) * (1-s)**(b-1)))
    new = np.random.choice([s, y], p=[1-p, p])

    states.append(new)

plt.hist(states, 50, density=True)
plt.title('simulation results of beta distribution')
plt.savefig('beta_distribution.png', dpi=300)
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
m = a / (a + b)
plt.plot(range(1000), np.abs(m - means))
plt.title('convergence rate of mean')
plt.xlabel("iteration (* %d)"%(step))
plt.ylabel("approximation error")

plt.subplot(122)
v = a*b / ((a+b)**2 * (a+b+1))
plt.plot(range(1000), np.abs(v-variances))
plt.title("convergence rate of variance")
plt.xlabel("iteration (* %d)"%(step))

plt.ylabel("approximation error")
plt.savefig("beta_distribution_convergence_rate.png", dpi=300)
plt.show()


