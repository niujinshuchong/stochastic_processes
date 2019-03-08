import numpy as np
import random
import math
import matplotlib.pyplot as plt

trial = 10000
y = 3
mu = 0
sigm = 1
tau = 4
ds = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 15., 20., 50., 80., 100.]

def simulate(d):
    init = 3.0
    states = [init]

    for i in range(1, trial):
        s = states[-1]
        # propose a new state x = s + normal(0, d^2)
        x = s + np.random.normal(0, d**2)

        # acceptance probability
        p = min(1.0, (math.exp((-1.0/(2*sigm))*(y-x)**2) * math.exp((-1.0/(2*tau))*(x-mu)**2) ) / \
                     (math.exp((-1.0/(2*sigm))*(y-s)**2) * math.exp((-1.0/(2*tau))*(s-mu)**2) ) )

        new = np.random.choice([s, x], p=[1-p, p])
        states.append(new)
    return states


plt.figure(figsize=(15, 20))
for i, d in zip(range(len(ds)), ds):
    states = simulate(d)
    plt.subplot(3, 4, i+2)
    plt.plot(range(trial), states)
    plt.title('Trace plots of theta (d = %.2lf)'%(d))
    #plt.xlabel('n')
    #plt.ylabel('theta')

    if d == 1.0:
        plt.subplot(3, 4, 1)
        plt.hist(states, 50, density=True)
        plt.title('distribution of theta when d = 1.0')
        #plt.xlabel('theta')
        #plt.ylabel('frequency')
plt.savefig('normal_normal_distribution.png', dpi=300)
plt.show()
    

