import matplotlib.pyplot as plt
import numpy as np

# Number of folds
num_folds = 5

# Create an array to represent the entire dataset
data = np.ones(100)

# Create colors for training and validation
colors = ["#808080", "#FFFFFF"] # gray, white

fig, ax = plt.subplots(figsize=(10, 5))

# Adjust the data representation for horizontal bars with black contour
for i in range(num_folds):
    # Calculate the start and end indices for the validation set in the current fold
    val_start = i * (data.shape[0] // num_folds)
    val_end = val_start + (data.shape[0] // num_folds)
    
    # Create masks for training and validation sets
    train_mask = np.ones(data.shape, dtype=bool)
    train_mask[val_start:val_end] = 0
    val_mask = ~train_mask
    
    # Plot the training and validation bars horizontally with black contour
    ax.barh(i, np.sum(train_mask), color=colors[0], edgecolor='black', label="Training" if i == 0 else "")
    ax.barh(i, np.sum(val_mask), left=val_start, color=colors[1], edgecolor='black', label="Validation" if i == 0 else "")
    
    # Adding the training color to the end portion of bars 1 to 4
    if i < num_folds - 1:
        ax.barh(i, data.shape[0] - val_end, left=val_end, color=colors[0], edgecolor='black')

# add another bar, completely white
ax.barh(num_folds, data.shape[0], color=colors[1])

# st x axis limits
ax.set_xlim([-5, data.shape[0]+5])

# Setting the title and labels
# ax.set_title("5-Fold Cross Validation")
ax.set_yticks(range(num_folds))
ax.set_yticklabels([f"Fold {i+1}" for i in range(num_folds)])
ax.set_xlabel("Number of Samples")
ax.set_xticks([0, data.shape[0]])
ax.set_xticklabels([0, "N"])
ax.legend(loc="upper right", prop={'size': 12})  # Enlarged legend

# Display the plot
plt.tight_layout()
plt.show()