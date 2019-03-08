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

trial = 10000
beta = 3
init = 1
states = [init]
counts = [0]

for i in range(trial):
    state = states[-1]
    f = get_state(state, node_num)
    count = np.sum(f)

    # generate all feasible state with hamming distance 1
    feasible_states = [state]
    feasible_counts = [count]
    for index in range(node_num):
        f[index] = 1 - f[index]

        # if new choice is invalid, stay in same state
        if f[index] == 1 and (True in set(f[graph[index]])):
            f[index] = 1 - f[index]    # restore original state
            continue

        new_state = state_to_int(f)
        new_count = np.sum(f)

        # new state is feasible and add to feasible set
        feasible_states.append(new_state)
        feasible_counts.append(new_count)

        # restore original state
        f[index] = 1 - f[index]
            
    '''
    # jump to new state with probability proportional to value
    c = np.array(feasible_counts)
    p = np.exp(beta * c)
    p = p / p.sum()
    new = np.random.choice(feasible_states, p=p)
    states.append(new)
    counts.append(np.sum(get_state(new, node_num)))
    '''

    count_down_timer = []
    for c in feasible_counts:
        count_down_timer.append(np.random.exponential(1. / (beta * c + 1.)))
    index = np.array(count_down_timer).argmin()
    new = feasible_states[index]
    states.append(new)
    counts.append(np.sum(get_state(new, node_num)))
   
unique, count = np.unique(counts, return_counts=True)
prob = count / np.sum(count)

# hist and convergence
plt.plot(unique, prob)
plt.title("probability distribution in size of independent set")
plt.xlabel("size of independent set")
plt.ylabel("probability")
plt.savefig("wireless_CTMC_distribution_beta_%d.png"%(beta), dpi=300)
plt.show()

