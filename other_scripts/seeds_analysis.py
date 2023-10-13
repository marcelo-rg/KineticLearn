import numpy as np
import matplotlib.pyplot as plt
import os

def create_histograms(hidden_sizes, folder_path):
    # Loop over each hidden size
    folder_path = os.path.join('checkpoints', folder_path)
    for hid_size in hidden_sizes:
        current_arch_dir = f'{folder_path}/{hid_size}_checkpoints'
        # Directory for the current architecture
        hid_dir = f'{folder_path}/{hid_size}_checkpoints'

        # Open the log file and read the second column
        with open(f'{hid_dir}/log_table.txt', 'r') as f:
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
        print(f"saving {current_arch_dir}/histogram.png")
        plt.clf()


def plot_mean_val_loss(hidden_sizes, folder_path):
    folder_path = os.path.join('checkpoints', folder_path)
    # Initialize lists to store mean and std
    mean_val_loss = []
    std_val_loss = []

    # Loop over each hidden size
    for hid_size in hidden_sizes:
        # Directory for the current architecture
        hid_dir = f'{folder_path}/{hid_size}_checkpoints'

        # Open the log file and read the second column
        with open(f'{hid_dir}/log_table.txt', 'r') as f:
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

    # plt.yscale('log')  # Use log scale for y-axis

    plt.tight_layout()

    # Save the plot
    plt.savefig('validation_loss_vs_architecture_size.png')
    plt.clf()


import seaborn as sns
import pandas as pd

def plot_heatmap(hidden_sizes, folder_path, fig_name='heatmap.png'):
    folder_path = os.path.join('checkpoints', folder_path)
    # Initialize dictionary to store mean validation loss
    val_loss_dict = {}

    for hid_size in hidden_sizes:
        # Directory for the current architecture
        hid_dir = f'{folder_path}/{hid_size}_checkpoints'

        # Open the log file and read the second column
        with open(f'{hid_dir}/log_table.txt', 'r') as f:
            lines = f.readlines()
            validation_loss = [float(line.split(',')[1]) for line in lines]

        # Calculate mean of the validation loss
        mean_val_loss = np.mean(validation_loss)

        # Number of neurons per layer and number of layers
        neurons = hid_size[0]  # assuming all layers have the same number of neurons
        layers = len(hid_size)

        if layers not in val_loss_dict:
            val_loss_dict[layers] = {}

        val_loss_dict[layers][neurons] = mean_val_loss

    # Convert dictionary to pandas DataFrame
    df = pd.DataFrame(val_loss_dict)

    # Sort DataFrame by index (Neurons per Layer)
    df.sort_index(inplace=True)

    # Create heatmap with 'viridis' color map
    ax = sns.heatmap(df, cmap='viridis')

    # Add labels and title
    plt.xlabel('Number of Layers')
    plt.ylabel('Neurons per Layer')
    plt.title('Mean Validation Loss as Function of Architecture Size')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90)

    # Invert the y-axis to have the highest number on top
    plt.gca().invert_yaxis()

    # Show plot
    plt.show()

    # Save the figure to the root directory
    fig = ax.get_figure()
    fig.savefig(fig_name)