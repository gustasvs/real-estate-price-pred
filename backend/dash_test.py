import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Data
x = np.random.uniform(1, 10, 100)
y = np.random.uniform(1, 10, 100)
z = np.random.uniform(1, 10, 100)
color = np.random.uniform(0.1, 1.0, 100)

# Plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(x, y, z, c=color, cmap='magma', s=400)
fig.colorbar(scatter, ax=ax, label='Validation Loss')
ax.set_xlabel('Hyperparameter 1')
ax.set_ylabel('Hyperparameter 2')
ax.set_zlabel('Hyperparameter 3')
ax.set_title('3D Scatter Plot of Hyperparameters and Validation Loss')
plt.show()
