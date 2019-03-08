import numpy as np
import random
import math
import matplotlib.pyplot as plt

colors = 7
nodes = 10
edges = [[0, 1], 
         [1, 2], [1, 4], 
         [2, 3], [2, 4],
         [3, 5], 
         [4, 5], [4, 7], [4, 9], 
         [5, 6], [5, 7], 
         [7, 8]]
# 0: white, 1: gray, 2: black
init_colors = [0, 2, 1, 2, 0, 1, 2, 2, 0, 1]

def create_graph(nodes, edges):
    # convert nodes and edges to adj matrix
    adj_matrix = np.zeros((nodes, nodes), dtype=np.bool)
    for s, e in edges:
        adj_matrix[s, e] = 1
        adj_matrix[e, s] = 1
    # convert adj_matrix to link_nodes
    graph = {}
    for i in range(nodes):
        graph[i] = []
        for j in range(nodes):
            if adj_matrix[i, j]:
                graph[i].append(j)
    return graph

def get_legal_colors(graph, state, index, colors):
    neighboor_colors = []
    for v in graph[index]:
        neighboor_colors.append(state[v])
    legal = set(range(colors)) - set(neighboor_colors) 
    return list(legal)

def check_legal(graph, f):
    for k, link_nodes in graph.items():
        for v in link_nodes:
            if f[k] == f[v]:
                 return False
    return True

graph = create_graph(nodes, edges)
print('graph:')
for k, v in graph.items():
    print(k, v)

# burte force search for optimal number of assignment
'''
all_states = []
optimal_count = 0
for i in range(colors**nodes):
    f = np.base_repr(i, base=colors)
    l = nodes - len(f)
    f = np.base_repr(i, base=colors, padding=l) 
    if check_legal(graph, f):
        #print(f)
        optimal_count += 1
        f = list(map(int, f))
        f = np.array(f)
        all_states.append(f)
all_states = np.stack(all_states)
print(all_states.shape)
print('number of all-feasible assignment:%d'%(optimal_count))
'''

trial = 10000000
counts_for_init = 0

init = np.array(init_colors, dtype=np.int32)

state = init.copy()
for i in range(trial):
    # random choice a node
    index = random.sample(range(nodes), 1)[0]
    # check legal colors
    legal_colors = get_legal_colors(graph, state, index, colors)

    # if there are more than one choice, random pick a color and jump to new state
    if len(legal_colors) > 1:
        new = random.sample(legal_colors, 1)[0]
        state[index] = new

    # since we want to find out the number of all-feasible assignment, 
    # we count the frequency to init state
    if np.sum(init == state) == nodes:
        counts_for_init += 1

prob = float(counts_for_init)/float(trial)
print('visit init state %d times in total %d trial'%(counts_for_init, trial))
print('approximate probability of visiting init state: %.5lf'%(prob))
print('approximate number of all-feasible assignment: %.5lf'%(1.0/prob))

