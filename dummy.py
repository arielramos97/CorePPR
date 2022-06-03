import numpy as np
import random


k = 32
half_k = int(k/2)

x = np.array([90,91,92,93,94,96,97,98,99,100])

idx = np.arange(half_k, k + half_k)

print(idx)

random_idx = np.random.choice(idx, 20, False)

print(random_idx)
random_nodes = x[random_idx]

print(random_nodes)