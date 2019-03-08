import numpy as np
import random
import math
import matplotlib.pyplot as plt

trial = 5000

x0 = 1
p0 = 0.5
n0 = 2
init = [x0, p0, n0]
states = [init]

for i in range(trial):
    x, p, n = states[-1]
    # generate xm from a binomial distribution with parameters n and p
    new_x = np.random.binomial(n, p)
    # generate pm from a beta distribution with parameters xm+1 and n_m-1 -xm + 1
    new_p = np.random.beta(new_x + 1, n - new_x + 1)
    # generate n_m = z + xm, z ~ pois(4*(1.-pm))
    z = np.random.poisson(4.0*(1.0 - new_p))
    new_n = z + new_x

    states.append([new_x, new_p, new_n])

states = np.array(states, dtype=np.float32)

plt.subplot(231)
plt.scatter(states[:, 0], states[:, 1], s=1)

plt.subplot(232)
plt.scatter(states[:, 0], states[:, 2], s=1)

plt.subplot(233)
plt.scatter(states[:, 1], states[:, 2], s=1)

plt.subplot(235)
plt.hist(states[:, 0])

plt.subplot(236)
plt.hist(states[:, 1])

plt.show()
