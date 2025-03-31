import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load input X and output y
X = pd.read_csv("../../X.csv")
y = pd.read_csv("../../y.csv")

# Plot1: the Y1, Y2 distribution
x_plot = np.arange(y.shape[0])
y1_plot = y['Y1'].to_numpy()
y1_sort = np.sort(y1_plot)
y2_plot = y['Y2'].to_numpy()
y2_sort = np.sort(y2_plot)
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Plot Y1
axs[0].scatter(x_plot, y1_sort, c="goldenrod", alpha=0.5)
axs[0].set_title('Y1 Values Distribution')
axs[0].set_xlabel('Samples', fontsize=12)
axs[0].set_ylabel('Y1', fontsize=12)
axs[0].grid()

# Plot Y2
axs[1].scatter(x_plot, y2_sort, c="orangered", alpha=0.5)
axs[1].set_title('Y2 Values Distribution')
axs[1].set_xlabel('Samples', fontsize=12)
axs[1].set_ylabel('Y2', fontsize=12)
axs[1].grid()

plt.tight_layout()
plt.savefig("./distribution_Y1_Y2_values.svg", bbox_inches='tight')
plt.show()