# %%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Let's get an intuition for how comparisons would look visually if some 3 models are self-similar.

grid = np.zeros((10, 10))

self_similar = [2, 5, 8, 7]

for i in range(10):
    for j in range(10):
        if i == j:
            grid[i, j] = 1
        elif i in self_similar and j in self_similar:
            grid[i, j] = 0.9
        else:
            grid[i, j] = 0.1


sns.heatmap(grid, vmin=0, vmax=1, annot=True, fmt=".3f", cmap="YlGnBu")
# %%
