import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import copy
import time
import random
import itertools
import json
import pandas as pd
import seaborn as sns
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.nn import MSELoss
from torch.optim import Adam
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import sys
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from src.NeuralNetworkModels import NeuralNet
from src.config import dict as dictionary

class LoadMultiPressureDatasetTorch(torch.utils.data.Dataset):

    def __init__(self, src_file, nspecies, num_pressure_conditions, react_idx=None, m_rows=None, columns=None,
                 scaler_input=None, scaler_output=None):
        self.num_pressure_conditions = num_pressure_conditions

        all_data = np.loadtxt(src_file, max_rows=m_rows,
                              usecols=columns, delimiter="  ",
                              comments="#", skiprows=0, dtype=np.float64)

        ncolumns = len(all_data[0])
        x_columns = np.arange(ncolumns - nspecies, ncolumns, 1) # densities
        y_columns = react_idx # k's
        if react_idx is None:
            y_columns = np.arange(0, ncolumns - nspecies, 1)

        x_data = all_data[:, x_columns]  # densities
        y_data = all_data[:, y_columns] * 1e30  # k's  # *10 to avoid being at float32 precision limit 1e-17

        # Reshape data for multiple pressure conditions
        x_data = x_data.reshape(num_pressure_conditions, -1, x_data.shape[1])
        y_data = y_data.reshape(num_pressure_conditions, -1, y_data.shape[1])

        # Create scalers
        self.scaler_input = scaler_input or [preprocessing.MaxAbsScaler() for _ in range(num_pressure_conditions)]
        self.scaler_output = scaler_output or [preprocessing.MaxAbsScaler() for _ in range(num_pressure_conditions)]
        
        for i in range(num_pressure_conditions):
            if scaler_input is None:
                self.scaler_input[i].fit(x_data[i])
            if scaler_output is None:
                self.scaler_output[i].fit(y_data[i])
            x_data[i] = self.scaler_input[i].transform(x_data[i])
            y_data[i] = self.scaler_output[i].transform(y_data[i])

        # Transpose x_data to move the pressure condition axis to the end, then flatten
        x_data = np.transpose(x_data, (1, 0, 2)).reshape(-1, self.num_pressure_conditions * x_data.shape[-1])
        
        # Flatten the output data to be of shape (2000,3)
        y_data = y_data[0]

        # Convert the data to PyTorch tensors
        self.x_data = torch.from_numpy(x_data).float()
        self.y_data = torch.from_numpy(y_data).float()

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return len(self.x_data)
    
    def get_data(self):
        return self.x_data, self.y_data
    

def train_model(model, criterion, optimizer, dataloader, num_epochs=100, patience=5, val_split=0.1):
    # Split the data into training and validation sets
    train_len = int((1.0 - val_split) * len(dataloader.dataset))
    val_len = len(dataloader.dataset) - train_len
    train_dataset, val_dataset = random_split(dataloader.dataset, [train_len, val_len])

    train_loader = DataLoader(dataloader.dataset, batch_size=dataloader.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=dataloader.batch_size, shuffle=False)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    # Early stopping details
    n_epochs_stop = patience
    min_val_loss = np.Inf
    epochs_no_improve = 0

    # To track loss history
    history = {
        'train_loss': [],
        'val_loss': [],
    }

    # Training loop
    for epoch in range(num_epochs):
        train_loss = 0.0
        val_loss = 0.0
        
        # Training phase
        model.train()
        for inputs, targets in train_loader:
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)

        # Validation phase
        model.eval()
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        # calculate average losses
        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        
        # record training/validation loss
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f"Epoch {epoch+1}, Training loss: {train_loss}, Validation loss: {val_loss}")

        # Early stopping check
        if val_loss < min_val_loss:
            epochs_no_improve = 0
            min_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            if epochs_no_improve == n_epochs_stop:
                print('Early stopping!')
                model.load_state_dict(best_model_wts)
                return model, history

    model.load_state_dict(best_model_wts)
    return model, history


def evaluate_model(model, test_data):
    model.eval()  # set the model to evaluation mode

    with torch.no_grad():
        for inputs, targets in test_data:
            outputs = model(inputs)

    # Calculate the Mean Squared Error (MSE) on the test data
    mse = mean_squared_error(targets.numpy(), outputs.numpy())
    print(f"Mean Squared Error (MSE) on the test data: {mse}")

    return targets, outputs, mse

def plot_results(targets, outputs, output_size):
    # Plot the predictions vs the true values
    fig, axs = plt.subplots(1, output_size, figsize=(15, 5), sharey=True)  # Share the same y-axis
    plt.rcParams.update({'font.size': 16, 'text.usetex': True} )

    for i in range(output_size):
        axs[i].scatter(targets[:, i], outputs[:, i], alpha=0.8, color=(0., 0., 0.9)) # blue
        # draw the y=x line
        axs[i].plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), '--', color='black')
        axs[i].set_xlabel('True Values', fontsize=14)
        # Only set the y-label for the first subplot since they share the same y-axis
        if i == 0:
            axs[i].set_ylabel('Predicted Values', fontsize=14)
        axs[i].set_title(f"$k_{{{i+1}}}$")

        # Calculate relative error
        denominator = outputs[:,i]
        denominator[np.abs(denominator) < 1e-9] = 1e-9  # Set small values to a small constant

        rel_err = np.abs(np.subtract(outputs[:,i], targets[:, i]) / denominator)

        textstr = '\n'.join((
            r'$Mean\ \delta_{rel}=%.2f\%%$' % (rel_err.mean() * 100,),
            r'$Max\ \delta_{rel}=%.2f\%%$' % (max(rel_err) * 100,)))

        # Colour point with max error
        max_index = np.argmax(rel_err)
        axs[i].scatter(targets[max_index, i], outputs[max_index, i], color="gold", zorder=2)

        # Define the text box properties
        props = dict(boxstyle='round', alpha=0.5)

        # Place a text box in upper left in axes coords
        axs[i].text(0.63, 0.25, textstr, fontsize=12, transform=axs[i].transAxes,
                verticalalignment='top', bbox=props)

         # Remove tick bars from non-first plots
        if i > 0:
            axs[i].tick_params(left=False)

    plt.tight_layout()
    plt.savefig(os.path.join('images', 'NeuralNet.pdf'))
    # plt.show()




def plot_loss_curves(history, log_scale=False):
    plt.rcParams.update({'font.size': 16, 'text.usetex': True})
    plt.figure(figsize=(9, 6))
    plt.plot(history['train_loss'],"-o", markersize=3, label='Training Loss')
    plt.plot(history['val_loss'], "-o", markersize=3, label='Validation Loss')
    # plt.title('Loss history')
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('MSE Loss', fontsize=16)
    # log y scale
    if log_scale:
        plt.yscale('log')
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.savefig(os.path.join('images', 'NeuralNet_loss_curves.pdf'))
    # plt.show()





def hyperparameter_random_search(n_samples):
    # Define the search space
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    batch_sizes = [16, 32, 64, 128]
    hidden_sizes = [(50,), (100,), (200,), (500,), \
                    (20, 20), (30, 30), (45, 45), (75,75)]

    activation_functions = ['tanh']

    best_mse = np.inf
    best_model = None
    best_hyperparameters = None

    for _ in range(n_samples):  # Number of samples
        # Sample hyperparameters
        lr = np.random.choice(learning_rates)
        batch_size = random.choice(batch_sizes)
        hidden_size = random.choice(hidden_sizes)
        activation_function = random.choice(activation_functions)

        # Create the model
        model = NeuralNet(input_size, output_size, hidden_size, activation_function)

        # Create the data loader
        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

        # Define the loss function and the optimizer
        criterion = MSELoss()
        optimizer = Adam(model.parameters(), lr=lr)

        # Train the model
        start = time.time()
        model, loss_history = train_model(model, criterion, optimizer, train_loader, num_epochs=1000, patience=15, val_split=0.1)
        end = time.time()
        print(f"Training time: {end - start}s")

        # Evaluate the model
        test_data = DataLoader(dataset_test, batch_size=len(dataset_test))
        targets, outputs, mse = evaluate_model(model, test_data)

        # Update best model
        if mse < best_mse:
            best_mse = mse
            best_model = model
            best_hyperparameters = (lr, batch_size, hidden_size, activation_function)

        print(f"Hyperparameters: lr={lr}, batch_size={batch_size}, hidden_size={hidden_size}, activation_function={activation_function}, MSE: {mse}")

    print(f"Best hyperparameters: lr={best_hyperparameters[0]}, batch_size={best_hyperparameters[1]}, hidden_size={best_hyperparameters[2]}, activation_function={best_hyperparameters[3]}, MSE: {best_mse}")
    return best_model, best_hyperparameters



def hyperparameter_grid_search():
    # Define the search space
    # learning_rates = [0.01, 0.001, 0.0001] 
    learning_rates = [0.00001, 0.000001]
    batch_sizes = [16, 32, 64, 128]
    hidden_sizes = [(50,), (100,), (200,), (500,), \
                    (20, 20), (30, 30), (45, 45), (75,75)]

    activation_functions = ['tanh']

    best_mse = np.inf
    best_model = None
    best_hyperparameters = None

    # Create a list to hold the results
    results = []

    total_iterations = len(learning_rates) * len(batch_sizes) * len(hidden_sizes) * len(activation_functions)

    # Grid search
    for lr, batch_size, hidden_size, activation_function in tqdm(
        itertools.product(learning_rates, batch_sizes, hidden_sizes, activation_functions),
        total=total_iterations,
        desc="Processing combinations"
    ):
        # Create the model
        model = NeuralNet(input_size, output_size, hidden_size, activation_function)

        # Create the data loader
        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

        # Define the loss function and the optimizer
        criterion = MSELoss()
        optimizer = Adam(model.parameters(), lr=lr)

        # Save the current stdout
        original_stdout = sys.stdout

        # Turn off stdout by redirecting it to os.devnull
        sys.stdout = open(os.devnull, 'w')

        # Train the model (output suppressed)
        model, loss_history = train_model(model, criterion, optimizer, train_loader, num_epochs=2000, patience=15, val_split=0.1)

        # Restore the original stdout
        sys.stdout.close()
        sys.stdout = original_stdout

        # Evaluate the model
        test_data = DataLoader(dataset_test, batch_size=len(dataset_test))
        targets, outputs, mse = evaluate_model(model, test_data)

        # Update best model
        if mse < best_mse:
            best_mse = mse
            best_model = model
            best_hyperparameters = (lr, batch_size, hidden_size, activation_function)

        print(f"Hyperparameters: lr={lr}, batch_size={batch_size}, hidden_size={hidden_size}, activation_function={activation_function}, MSE: {mse}")

        # Save the results
        results.append({
            'lr': lr,
            'batch_size': batch_size,
            'hidden_size': str(hidden_size),
            'activation_function': activation_function,
            'mse': mse,
        }) # remeber to save a flag for early stopping

    print(f"Best hyperparameters: lr={best_hyperparameters[0]}, batch_size={best_hyperparameters[1]}, hidden_size={best_hyperparameters[2]}, activation_function={best_hyperparameters[3]}, MSE: {best_mse}")
    return best_model, best_hyperparameters, results


# Custom JSON encoder to handle float32 values
class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return super().default(obj)


if __name__ == "__main__":
    torch.manual_seed(8) # for reproducibility

    # Select scheme
    scheme = 'O2_simple'

    # Parameters for data loading
    src_file_train = dictionary[scheme]['main_dataset']
    src_file_test = dictionary[scheme]['main_dataset_test']
    nspecies = dictionary[scheme]['n_densities']
    num_pressure_conditions = dictionary[scheme]['n_conditions']

    # Load the training data
    dataset_train = LoadMultiPressureDatasetTorch(src_file_train, nspecies, num_pressure_conditions, react_idx=dictionary[scheme]['k_columns'])
    x_train, y_train = dataset_train.get_data()

    # Load the test data
    dataset_test = LoadMultiPressureDatasetTorch(src_file_test, nspecies, num_pressure_conditions, 
                                                 react_idx=dictionary[scheme]['k_columns'], scaler_input=dataset_train.scaler_input, scaler_output=dataset_train.scaler_output)
    x_test, y_test = dataset_test.get_data()

    # Check the shape of the data
    print(f"Shape of x_data: {x_train.shape}") # (2000, 9)
    print(f"Shape of y_data: {y_train.shape}") # (2000, 3)


    # Define the network
    input_size = int(nspecies*num_pressure_conditions)  # 11 densities per each pressure condition
    output_size = len(dictionary[scheme]['k_columns'])  # 3 coefficients
    
    
    hidden_size = (30, 30)  # example value, can be tuned
    model = NeuralNet(input_size, output_size, hidden_size, activ_f='tanh')

    # Define the loss function and the optimizer
    criterion = MSELoss()
    optimizer = Adam(model.parameters(), lr=0.0001) 

    start = time.time()

    # # Hyperparameter search
    # model, hyperparameters, results = hyperparameter_grid_search()
    
    # # Save the results
    # with open('results_augmented'+scheme+'.json', 'w') as f:
    #     json.dump(results, f, cls=CustomEncoder)

    # print(f"Best hyperparameters: lr={hyperparameters[0]}, batch_size={hyperparameters[1]}, hidden_size={hyperparameters[2]}, activation_function={hyperparameters[3]}")
    
    # Create data loaders
    train_loader = DataLoader(dataset_train, batch_size=16, shuffle=True)  # batch size can be tuned

    # Train the model
    model, loss_history = train_model(model, criterion, optimizer, train_loader, num_epochs=5000, patience=100, val_split=0.1)

    end = time.time()
    print(f"Training time: {end - start}s")

    # Evaluate the model
    test_data = DataLoader(dataset_test, batch_size=len(dataset_test))
    targets, outputs, mse = evaluate_model(model, test_data)

    # Print mse value for the test data
    print(f"Mean Squared Error (MSE) on the test data: {mse}")
    
    # Plot results
    # plot_results(targets.numpy(), outputs.numpy(), output_size)

    # Plot loss curves
    plot_loss_curves(loss_history, log_scale=True)
    