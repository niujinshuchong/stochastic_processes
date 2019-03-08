import numpy as np
import random
import math
import matplotlib.pyplot as plt

trial = 10000
rate = 10
a = 1
b = 1
x = 7

p = 0.5
n = 14
init = [p, n]
states = [init]

for i in range(trial):
    p, n = states[-1]
    # conditional on N = n, X = x, draw a new guess from p from beta(x+a, n-x+b)
    new_p = np.random.beta(x+a, n-x+b)
    # conditional on new_p and X = x, the number of unhatched eggs is Y ~ pois(rate(1-p))
    # so we draw new guess of n = x + y
    y = np.random.poisson(rate * (1.0 - p))
    new_n = x + y
    states.append([new_p, new_n])

states = np.array(states)

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.hist(states[:, 0], 20)
plt.title('Histogram of p')
plt.subplot(122)
ns = states[:, 1].astype(np.int32)
plt.hist(ns, len(np.unique(ns)))
plt.title('Histogram of N')
plt.savefig("chichen_egg_distribution.png")
plt.show()
