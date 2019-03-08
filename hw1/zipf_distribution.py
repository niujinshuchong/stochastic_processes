import numpy as np
import random
import matplotlib.pyplot as plt

M = 6
a = 0.5

def get_p(k):
    return 1. / (k ** a)

ps = np.zeros(1+M)
for i in range(1, M+1):
    ps[i] = get_p(i)

n = 1000000
sim_list = np.zeros(n, dtype=np.int32)
sim_list[0] = 2

for i in range(1, n):
    if sim_list[i-1] == 1:
        p1 = ps[1]
        p2 = ps[2]
        p = min(1, p2 / p1 * 0.5)
        new = np.random.choice([1, 2], 1, p=[1-p, p])[0]
        sim_list[i] = new
    elif sim_list[i-1] == M:
        p_m = ps[M]
        p_m_1 = ps[M-1]
        p = min(1, p_m_1 / p_m * 0.5)
        new = np.random.choice([M, M-1], 1, p=[1-p, p])[0]
        sim_list[i] = new
    else:
        left_right = np.random.choice([-1, 1], 1)[0]
        if left_right == -1:
            current = sim_list[i-1]
            p_c = ps[current]
            p_c_1 = ps[current-1]
            p = min(1, p_c_1 / p_c)
            if p == 1: 
                sim_list[i] = current-1
            else:
                new = np.random.choice([current, current-1], 1, p=[1-p, p])[0]
                sim_list[i] = new
        else:
            current = sim_list[i-1]
            p_c = ps[current]
            p_c_1 = ps[current+1]
            p = min(1, p_c_1 / p_c)
            if p == 1:
                sim_list[i] = current + 1
            else:
                new = np.random.choice([current, current+1], 1, p=[1-p, p])[0]
                sim_list[i] = new
                        

counts, _  = np.histogram(sim_list, bins=M)
tab = counts/float(n)
print('similation results of zipf distribution with M = %d, a = %.4lf:'%(M, a))
print(tab)


counts, _  = np.histogram(sim_list[n//10:], bins=M)
tab_burn_in = counts/float(n - n//10)
print('burn-in similation results of zipf distribution with M = %d, a = %.4lf:'%(M, a))
print(tab_burn_in)

probs = ps / ps.sum()
probs = probs[1:]
print('exact distribution is:')
print(probs)
print('approximate gaps is:')
print(np.abs(probs - tab))

print('approximate gaps of burn is:')
print(np.abs(probs - tab_burn_in))



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
plt.title('zipf convergence rate')
plt.grid()
plt.savefig('zipf_convergence_rate.png', dpi=600)
plt.show()

