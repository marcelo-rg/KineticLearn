import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import numpy as np

# folder_path = "D:\\Marcelo\\github\\Dissertation\\Images\\"


# Enable LaTeX formatting and set global font size
plt.rcParams.update({'font.size': 16, 'text.usetex': True})

# Load the grid search results from the CSV file
df_grid_results = pd.read_csv('SVM_grid_search_results_v2.csv')

# Convert the negative MSE to positive
df_grid_results['mean_test_score'] = df_grid_results['mean_test_score'].abs()

# Extract unique output names
outputs = df_grid_results['Unnamed: 0'].unique()


# Create the grid for subplots and colorbar with adjusted wspace
fig = plt.figure(figsize=(15, 6))
gs = GridSpec(1, len(outputs) + 1, width_ratios=[1] * len(outputs) + [0.05], wspace=0.08)

axes = [fig.add_subplot(gs[i]) for i in range(len(outputs))]

# Plot heatmap for each output
for i, (output, ax) in enumerate(zip(outputs, axes)):
    df_output = df_grid_results[df_grid_results['Unnamed: 0'] == output]

    # Filter the dataframe for rows where param_epsilon is 0.001
    df_output = df_output[df_output['param_epsilon'] == 0.001] # FOR ZOOM PLOT
    
    # Extracting the hyperparameters and scores into a pivot table
    # pivot = df_output.pivot_table(values='mean_test_score', columns='param_gamma', index=['param_epsilon', 'param_C'])
    pivot = df_output.pivot_table(values='mean_test_score', columns='param_gamma', index=['param_C']) # FOR ZOOM PLOT
    
    # Plot heatmap on the corresponding axis.
    # sns.heatmap(pivot, annot=True, cmap='Greens_r', fmt=".2e", linewidths=.5, cbar=(i == len(outputs) - 1), cbar_ax=None if i < len(outputs) - 1 else fig.add_subplot(gs[-1]), cbar_kws={'label': 'MSE'}, ax=ax, annot_kws={"size": 12})
    sns.heatmap(pivot, annot=True, cmap='Greens_r', fmt=".2e", linewidths=.5, cbar=False, ax=ax, annot_kws={"size": 14})
    
    # Set title for each subplot based on output
    ax.set_title(f"Output {i+1}: \(k_{i+1}\)", fontsize=14)

    ###### FOR ZOOM PLOT ######
    ###########################
    # Find the position of the minimum value directly
    row_num, col_num = np.unravel_index(pivot.values.argmin(), pivot.shape)
    min_val = pivot.values[row_num, col_num]
    min_val_row = pivot.index[row_num]
    min_val_index = pivot.columns[col_num]

    # Diagnostic print to ensure we're identifying the correct minimum value
    print(f"For output {output}, minimum MSE value: {min_val}, at C={min_val_row}, gamma={min_val_index}")

    # Add a rectangle around the square with the minimum value
    ax.add_patch(Rectangle((col_num, row_num), 1, 1, fill=False, edgecolor='goldenrod', lw=3))
    ###########################
    ###### END ZOOM PLOT ######

    # Highlight horizontal lines after every 5 rows to denote change in C values
    for j in range(5, len(pivot), 5):
        ax.hlines(j, *ax.get_xlim(), color='white', linewidth=5)

    # Update the x-axis label with LaTeX formatting
    ax.set_xlabel(r'\(\mathbf{\gamma}\)', fontsize=14)
    
    # Remove y-label for the second and third plots
    if i > 0:
        ax.set_ylabel('')
    else:
        # ax.set_ylabel(r'\(\mathbf{\epsilon} - \mathbf{C}\)', fontsize=14)
        ax.set_ylabel(r'\(\mathbf{C}\)', fontsize=14) # FOR ZOOM PLOT


# Manually synchronize the y-axis limits
ymin, ymax = axes[0].get_ylim()
for ax in axes[1:]:
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_ticks([])  # Hide the y-axis ticks for the second and third subplots


# Adjust the margins
plt.subplots_adjust(left=0.083, right=1.03)
plt.tight_layout()
plt.show()
# plt.savefig(folder_path + 'SVM_heatmap.pdf')
# plt.savefig(folder_path + 'SVM_heatmap_zoom.pdf') # FOR ZOOM PLOT
