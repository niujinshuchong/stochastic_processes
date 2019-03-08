import numpy as np
import random
import math
import matplotlib.pyplot as plt

# helper function, note that state is represent as a interger, 
# we convert it to binary to indicate choice
def get_state(v, m):
    f = np.binary_repr(v, width=m)
    f = np.array(list(f)) == '1'
    return f

def state_to_int(f):
    f = f.astype(np.int32)
    return f.dot(2**np.arange(f.size)[::-1])

# create graph
def create_petersen_graph(node_num, skip):
    adj_matrix = np.zeros((node_num+1, node_num+1), dtype=np.bool) 

    for i in range(1, node_num // 2 + 1):
        ends = []
        for j in range(1, 3):
            end = i + j * skip 
            if end > node_num // 2:
                end = end % (node_num // 2)
            ends.append(end)
        ends.append(i + skip * 3)
        for end in ends:
            adj_matrix[i, end] = True
            adj_matrix[end, i] = True
            #print(i, end)

    for i in range(node_num // 2 + 1, node_num):
        adj_matrix[i, i+1] = True        
        adj_matrix[i+1, i] = True        
        #print(i, i+1)
    adj_matrix[node_num, node_num // 2 + 1] = True
    adj_matrix[node_num // 2 + 1, node_num] = True
    #print(node_num, node_num // 2 + 1)    

    graph = {}
    for i in range(1, node_num + 1):
        graph[i-1] = []
        for j in range(1, node_num + 1):
            if adj_matrix[i, j]:
                graph[i-1].append(j-1)
    return graph

node_num = 24
skip = 4
graph = create_petersen_graph(node_num, skip)
print('petersen graph is')
print(graph)   

trial = 100000
beta = 2
init = 0
states = [init]
counts = [0]

maximum_independent_states = set([])

for i in range(trial):
    state = states[-1]
    f = get_state(state, node_num)
    count = np.sum(f)
    # if current state is maximum independent set, store it
    if count == 9:
        maximum_independent_states.add(state)

    # uniformly choose a position and replace it
    index = random.sample(range(node_num), 1)[0]
    f[index] = 1 - f[index]
    new_state = state_to_int(f)
    new_count = np.sum(f)

    # if new choice is invalid, stay in same state
    if f[index] == 1 and (True in set(f[graph[index]])):
        states.append(state)
        counts.append(count)
    else:
        # jump to new state with probability min(1, e^(beta(worth1 - worth2)))
        p = min(1, math.exp(beta * (new_count - count)))
        new = np.random.choice([state, new_state], p=[1-p, p])
        states.append(new)
        counts.append(np.sum(get_state(new, node_num)))

# print all feasible maximum independent size
print('all feasible maximum independent set is:')
for s in maximum_independent_states:
    f = get_state(s, node_num)   
    print(f)

unique, count = np.unique(counts, return_counts=True)
prob = count / np.sum(count)

# hist and convergence
plt.plot(unique, prob)
plt.title("probability distribution in size of independent set")
plt.xlabel("size of independent set")
plt.ylabel("probability")
plt.savefig("wireless_DTMC_distribution_beta_%d.png"%(beta), dpi=300)
plt.show()

