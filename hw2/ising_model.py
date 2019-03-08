import numpy as np
import random
import math
import matplotlib.pyplot as plt

trial = 100000
beta_list = [-1.5, -1.0, 0.0, 0.2, 0.4, 0.441, 0.5, 1, 2, 10]

def ising_model(beta, g=60):
    grid = np.random.choice([-1, 1], (g+2)**2)
    grid = np.asarray(grid).reshape(g+2, g+2)
    grid[:, [0, -1]] = 0
    grid[[0, -1], :] = 0

    for m in range(trial):
        # random sample a positon
        i = 1 + random.sample(range(g), 1)[0]
        j = 1 + random.sample(range(g), 1)[0]
        # check degree
        deg = grid[i, j+1] + grid[i, j-1] + grid[i+1, j] + grid[i-1, j]
        p = 1. / (1. + math.exp(-beta * 2.0 * deg))

        new = np.random.choice([1, -1], p=[p, 1-p])
        grid[i, j] = new
    return grid[1:-1, 1:-1]



plt.figure(figsize=(6, 15))
i = 0
for beta in beta_list:
    grid = ising_model(beta, 100)
    plt.subplot(5, 2, i+1)
    plt.imshow(grid)
    plt.title('beta=%.3lf'%(beta))
    plt.axis("off")
    i += 1
plt.savefig("Ising_model.png", dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()


