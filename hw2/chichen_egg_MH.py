import numpy as np
import random
import math
import matplotlib.pyplot as plt

trial = 100000
rate = 10.
a = 1.
b = 1.
x = 7.

init_p = 0.5
n = 14
states = [init_p]

for i in range(trial):
    p = states[-1]
    # generate proposal given X=x
    new_p = np.random.beta(a, b)
    f_p = math.exp(-rate * p) * ((rate * p) ** x) * (p**(a-1)) * ((1-p)**(b-1))
    f_new_p = math.exp(-rate * new_p) * ((rate * new_p) ** x) * (new_p**(a-1)) * ((1-new_p)**(b-1))

    a_p_new_p = min(1.0, f_new_p/f_p)
    new = np.random.choice([p, new_p], p=[1-a_p_new_p, a_p_new_p])
    states.append(new)

states = np.array(states)

plt.hist(states, 20)
plt.title('Histogram of p')
plt.savefig("chichen_egg_MH_distribution.png")
plt.show()
