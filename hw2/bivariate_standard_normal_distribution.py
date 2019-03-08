import numpy as np
import random
import math
import matplotlib.pyplot as plt

trial = 2000
correlations = [-0.6, 0.6]

def simulate(correlation):
    init = [0.0, 0.0]
    states = [init]

    for i in range(trial):
        x, y = states[-1]
        # generate xm from conditional distribution of x given y
        new_x = np.random.normal(correlation*y, 1.0 - correlation**2)
        # generate ym from conditional distribution of y given new_x
        new_y = np.random.normal(correlation*new_x, 1.0-correlation**2)

        states.append([new_x, new_y])
    return states


plt.figure(figsize=(12, 6))
for i, c in zip(range(2), correlations):
    states = simulate(c)
    states = np.array(states)
    plt.subplot(1, 2, i+1)
    plt.scatter(states[:, 0], states[:, 1])
    plt.title('bivariate standard normal distribution, correlation=%.1lf'%(c))
plt.savefig("bivariate_standard_normal_distribution.png", dpi=300)
plt.show()


