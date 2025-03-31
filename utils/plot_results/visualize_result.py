import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load input X and output y
y_result = pd.read_csv("../../results/result_MLP.csv")
y = pd.read_csv("../../y.csv")

# Create 2 plots
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
plt.suptitle("MLP model", fontsize = 30)

# Plot Y1
y1_target_plot = y['Y1'].to_numpy()
y1_predict_plot = y['Y1'].to_numpy()

# Sort values
y1_pred_sort = np.sort(y1_predict_plot[0:100])
y1_sort_pred_indices = np.argsort(y1_predict_plot[0:100])
y1_target_sort = y1_target_plot[y1_sort_pred_indices]
x1_plot = np.arange(y1_pred_sort.shape[0])

axs[0].set_xlabel('Samples', fontsize=12)
axs[0].set_ylabel('Y1', fontsize=12)
axs[0].scatter(x1_plot, y1_target_sort, alpha=0.5, label='ground_true')
axs[0].plot(x1_plot, y1_pred_sort, label='prediction', color='r')
axs[0].grid()
axs[0].legend()


# Plot Y2
y2_target_plot = y['Y2'].to_numpy()
y2_predict_plot = y['Y2'].to_numpy()

# Sort values
y2_pred_sort = np.sort(y2_predict_plot[0:100])
y2_sort_pred_indices = np.argsort(y2_predict_plot[0:100])
y2_target_sort = y2_target_plot[y1_sort_pred_indices]
x2_plot = np.arange(y2_pred_sort.shape[0])

axs[1].set_xlabel('Samples', fontsize=12)
axs[1].set_ylabel('Y2', fontsize=12)
axs[1].scatter(x2_plot, y1_target_sort, alpha=0.5, label='ground_true')
axs[1].plot(x2_plot, y1_pred_sort, label='prediction', color='r')
axs[1].grid()
axs[1].legend()

plt.tight_layout()
plt.savefig("./result_MLP.svg", bbox_inches='tight')
plt.show()