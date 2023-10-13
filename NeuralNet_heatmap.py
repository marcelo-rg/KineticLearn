import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# folder_path = "D:\\Marcelo\\github\\Dissertation\\Images\\"

# Best hyperparameters: lr=0.0001, batch_size=16, hidden_size=(20, 20), activation_function=tanh, MSE: 6.111129550845362e-06
# The search space:
# # learning_rates = [0.01, 0.001, 0.0001] 
# # batch_sizes = [16, 32, 64, 128]
# # hidden_sizes = [(50,), (100,), (200,), (500,), \
# #                 (20, 20), (30, 30), (45, 45), (75,75)]

# Enable LaTeX formatting and set global font size
plt.rcParams.update({'font.size': 14, 'text.usetex': True})

scheme = ""

# Load the saved JSON files
with open('results'+scheme+'.json', 'r') as f:
    results = json.load(f)

with open('results_augmented'+scheme+'.json', 'r') as f:
    results_augmented = json.load(f)

# Convert dictionaries to DataFrames
results_df = pd.DataFrame(results)
results_augmented_df = pd.DataFrame(results_augmented)

# Concatenate vertically
final_results_df = pd.concat([results_df, results_augmented_df], ignore_index=True)

# Remove entries where 'lr' is 1e-06
final_results_df = final_results_df[final_results_df['lr'] != 1e-06]

# Optional: Assign the final dataframe back to results_df, if desired
# results_df = final_results_df

# replace mse value of worst model with 1
# results_df['mse'] = results_df['mse'].replace(results_df['mse'].max(), 0.0015)

# Specify the order of hidden sizes
hidden_sizes_order = ['(50,)', '(100,)', '(200,)', '(500,)', '(20, 20)', '(30, 30)', '(45, 45)', '(75, 75)']

# Convert the 'hidden_size' column to a category type with specified order
results_df['hidden_size'] = pd.Categorical(results_df['hidden_size'], categories=hidden_sizes_order, ordered=True)


# Create the pivot table for all results
pivot = results_df.pivot_table(values='mse', index='hidden_size', columns=['lr', 'batch_size'])


# Create a heatmap with specified aesthetics
fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(pivot, cmap='Blues_r', fmt=".2e", linewidths=.5, ax=ax, annot=True, annot_kws={"size": 14})

# Add vertical lines after every 4 columns to denote change in learning rates
num_columns = len(pivot.columns)
for i in range(4, num_columns, 4):
    ax.axvline(x=i, color='white', linewidth=4)

# Find the position of the minimum value directly
row_num, col_num = np.unravel_index(pivot.values.argmin(), pivot.shape)
min_val = pivot.values[row_num, col_num]
min_val_row = pivot.index[row_num]
min_val_col = pivot.columns[col_num]

# Diagnostic print to ensure we're identifying the correct minimum value
print(f"Minimum MSE value: {min_val}, at hidden size={min_val_row}, learning rate and batch size={min_val_col}")

# Add a rectangle around the square with the minimum value
ax.add_patch(Rectangle((col_num, row_num), 1, 1, fill=False, edgecolor='goldenrod', lw=3))

ax.set_xlabel(r'\(\mathbf{Learning Rate - Batch Size}\)', fontsize=14)
ax.set_ylabel(r'\(\mathbf{Hidden Size}\)', fontsize=14)

# Adjust the margins
plt.subplots_adjust(left=0.083, right=0.9)
plt.tight_layout()
# plt.savefig(folder_path+ 'NeuralNet_heatmap.pdf', bbox_inches='tight')
plt.show()
