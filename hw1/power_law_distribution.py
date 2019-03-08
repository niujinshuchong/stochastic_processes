import numpy as np
import random
import matplotlib.pyplot as plt

# calculate exact distribution
all_prob = 0.0
for i in range(1, 10000):
    prob = i ** (-3./2.)
    all_prob += prob

probs = []
for i in range(1, 9):
    prob = (i ** (-3./2.)) / all_prob
    probs.append(prob)
probs = np.array(probs)

print('exact results of power law distribution is:')
print(probs)

n = 1000000
sim_list = np.zeros(n, dtype=np.int32)
sim_list[0] = 2

for i in range(1, n):
    if sim_list[i-1] == 1:
        p = (1./2.) ** (5./2.)
        new = np.random.choice([1, 2], 1, p=[1-p, p])[0]
        sim_list[i] = new
    else:
        left_right = np.random.choice([-1, 1], 1)[0]
        if left_right == -1:
            sim_list[i] = sim_list[i-1] -1
        else:
            p = (float(sim_list[i-1]) / (sim_list[i-1]+1)) ** (3./2.)
            new = np.random.choice([sim_list[i-1], 1+sim_list[i-1]], 1, p=[1-p, p])[0]
            sim_list[i] = new

counts, _  = np.histogram(sim_list, bins=np.max(sim_list))
tab = counts/float(n)
print('The simulation results of power law distribution is:')
print(tab[:8])

counts, _  = np.histogram(sim_list[n//10:], bins=np.max(sim_list))
tab_burn_in = counts/float((n - n//10))
print('The burn-in simulation results of power law distribution is:')
print(tab_burn_in[:8])

print('approximate gaps is:')
print(np.abs(probs[:8] - tab[:8]))

print('approximate gaps of burn is:')
print(np.abs(probs[:8] - tab_burn_in[:8]))


# convergence rate
values = []
step = 1000
ends = []
for i in range(1, step+1):
   end = n // step * i
   counts, _  = np.histogram(sim_list[:end], bins=np.max(sim_list))
   tab = counts/float(end)
   v = np.mean(np.abs(probs[:8] - tab[:8]))
   ends.append(end)
   values.append(v)

fig = plt.figure()
ax = fig.gca()

plt.plot(np.array(ends), np.array(values))

plt.xlabel('iteration T')
plt.ylabel('approximate error')
plt.title('power law convergence rate')
plt.grid()
plt.savefig('power_law_convergence_rate.png', dpi=600)
plt.show()

