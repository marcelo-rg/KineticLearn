import numpy as np
import matplotlib.pyplot as plt

# List of hidden sizes for different architectures
hidden_sizes = [(10,), (20,), (30,), (40,), (50,),
                (10, 10), (20, 20), (30, 30), (40, 40), (50, 50),
                (10, 10, 10), (20, 20, 20), (30, 30, 30), (40, 40, 40), (50, 50, 50)]

hidden_sizes = [(10, 10)]

# Loop over each hidden size
for hid_size in hidden_sizes:
    # Directory for the current architecture
    current_arch_dir = f'checkpoints/{hid_size}_checkpoints'

    # Open the log file and read the second column
    with open(f'{current_arch_dir}/log_table.txt', 'r') as f:
        lines = f.readlines()
        validation_loss = [float(line.split(',')[1]) for line in lines]

    # Calculate mean and std of the validation loss
    mean_val_loss = np.mean(validation_loss)
    std_val_loss = np.std(validation_loss)

    # Create histogram
    plt.hist(validation_loss, bins=20, color='skyblue', edgecolor='black')
    plt.title(f'Validation Loss Histogram for {hid_size} Architecture')
    plt.xlabel('Validation Loss')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.text(0.6, 0.7, f'Mean: {mean_val_loss:.5f}\nStd: {std_val_loss:.5f}', 
             transform=plt.gca().transAxes)

    # Save the histogram
    plt.savefig(f'{current_arch_dir}/histogram.png')
    plt.clf()
