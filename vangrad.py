"""
try code from
https://cs224d.stanford.edu/notebooks/vanishing_grad_example.html

2018-06-11 martin
"""

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)

N = 100  # points per class
D = 2  # dimensions
K = 3  # number of classes
X = np.zeros((N * K, D))
num_train_examples = X.shape[0]
y = np.zeros(N * K, dtype='uint8')


for j in range(K):
    ix = range(N * j, N * (j + 1))
    r = np.linspace(.0, 1, N)
    theta = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * .2
    X[ix] = np.c_[r * np.sin(theta), r * np.cos(theta)]
    y[ix] = j

plt.scatter(X[:, 0], X[:, 1], c=y, s=40)
plt.savefig('/dev/shm/vangrad_000_spiral_scatter.png')
