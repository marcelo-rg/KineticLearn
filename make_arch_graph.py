import os
import numpy as np
import matplotlib.pyplot as plt

# List of hidden sizes for different architectures
hidden_sizes = [(10,), (20,), (30,), (40,), (50,),
                (10, 10), (20, 20), (30, 30), (40, 40), (50, 50),
                (10, 10, 10), (20, 20, 20), (30, 30, 30), (40, 40, 40), (50, 50, 50)]

# Initialize lists to store mean and std
mean_val_loss = []
std_val_loss = []


# Loop over each hidden size
for hid_size in hidden_sizes:
    # Directory for the current architecture
    current_arch_dir = f'checkpoints/{hid_size}_checkpoints'

    # Open the log file and read the second column
    with open(f'{current_arch_dir}/log_table.txt', 'r') as f:
        lines = f.readlines()
        validation_loss = [float(line.split(',')[1]) for line in lines]

    # Calculate mean and std of the validation loss
    mean_val_loss.append(np.mean(validation_loss))
    std_val_loss.append(np.std(validation_loss) / np.sqrt(50))  # std error of the mean

# Define x values as strings of hidden sizes
x_values = [str(hid_size) for hid_size in hidden_sizes]

# Create plot
plt.errorbar(x_values, mean_val_loss, yerr=std_val_loss, fmt='o-', capsize=5)
plt.title('Mean Validation Loss as Function of Architecture Size')
plt.xlabel('Architecture (hidden size)')
plt.ylabel('Mean Validation Loss')
plt.grid(True)
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability

plt.tight_layout()

# Save the plot
plt.savefig('validation_loss_vs_architecture_size.png')
plt.clf()
