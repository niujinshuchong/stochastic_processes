import numpy as np
import random
import math
import matplotlib.pyplot as plt

def simulate(init, n):
    '''
        init: numpy array with shape (m)
        n: maximum iteration of simulation
        return expected number of 1s in good sequences
    '''
    k = init.shape[0]
    total = 0
    new = np.concatenate((2.0*np.ones((1, 1)),
                          init.reshape(1, -1),
                          2.0*np.ones((1, 1))), axis=1).reshape(-1)
    states = np.zeros(n)
    for i in range(n):
        index = 1 + random.sample(range(k), 1)[0]
        newbit = 1 - new[index]
        if newbit == 0:
            new[index] = 0
            states[i] = np.sum(new)
            continue
        elif new[index-1]  == 1 or new[index+1] == 1:
            states[i] = np.sum(new)
            continue
        else:
            new[index] = 1
            states[i] = np.sum(new)

    return np.mean(states) - 4, states

n = 10000
m = 100
init = np.zeros(m)
e, states = simulate(init, n)
print('expected number of 1s in good sequences with length %d: '%(m), e)

exact = 27.7921
# convergence rate
values = []
step = 100
ends = []
for i in range(1, step+1):
   end = n // step * i
   v = np.mean(states[:end] - 4)
   print(v)
   ends.append(end)
   values.append(v)

fig = plt.figure()
ax = fig.gca()

plt.plot(np.array(ends), abs(exact - np.array(values)))

plt.xlabel('iteration T')
plt.ylabel('approximate error')
plt.title('good sequence convergence rate')
plt.grid()
plt.savefig('good_sequence_convergence_rate.png', dpi=600)
plt.show()

   

