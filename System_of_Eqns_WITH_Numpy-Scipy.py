import numpy as np


a = np.array([[5, 3, 1],[3, 9, 4],[1, 3, 5]])
b = np.array([[9],[16],[9]])

x = np.linalg.solve(a, b)

print(x)
