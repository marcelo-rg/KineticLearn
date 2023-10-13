import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from src.config import dict as dictionary
scheme = 'O2_novib'
folder_path = "D:\\Marcelo\\github\\Dissertation\\Images\\"

true_data = np.loadtxt('data/true_pressure_curve.txt', comments='#')
predicted_data = np.loadtxt('data/predicted_pressure_curve.txt', comments='#')

# x data should be the 4th column and convert to pascal
x_true = true_data[:, 3]/133.332
x_pred = predicted_data[:, 3]/133.332

# y data should be the 5th column and onwards
y_true = true_data[:, 4:]
y_pred = predicted_data[:, 4:]

labels = dictionary[scheme]['species']

n_plots = len(labels)
n_plots_row = 4
n_plots_col = 3  # 4 plots per row

fig, axs = plt.subplots(n_plots_row, n_plots_col, figsize=(20, 16), sharex=True)
plt.rcParams.update({'font.size': 16, 'text.usetex': True})

for i in range(n_plots_row):
    for j in range(n_plots_col):
        index = i * n_plots_col + j
        if index >= n_plots:
            axs[i, j].axis('off')  # Turn off extra subplots
            continue

        axs[i, j].plot(x_true, y_true[:, index], '--', label='Model')
        axs[i, j].scatter(x_pred, y_pred[:, index], marker='v', label='Predicted', edgecolors='#FF8C00', facecolors='none')
        
        # Highlight the first and fourth points
        highlighted_indices = [0, 3]
        axs[i, j].scatter(x_pred[highlighted_indices], y_pred[highlighted_indices, index], marker='o', color=(0.8,0,0), label='Used in Training', zorder=5)

        # Set the tick marks for x-axis
        axs[i, j].set_xticks([1, 10, 20, 30, 40, 50])

        # Change the font size of tick labels
        axs[i, j].tick_params(axis='both', labelsize=12)

        # Standardize the number of ticks on the y-axis
        axs[i, j].yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))

        if index >= 8:
            axs[i, j].set_xlabel('Pressure (Torr)', fontsize=14)
        axs[i, j].set_ylabel(labels[index], fontsize=14)
        axs[i, j].grid(True)
        axs[i, j].legend(fontsize=14)

plt.tight_layout()
# plt.savefig(folder_path + 'physical_plots.pdf')
plt.show()







