import torch
from torch.nn import MSELoss
from torch.optim import Adam
import time

from src.Model import NSurrogatesModel
from src.Trainer import NSurrogatesModelTrainer
from src.DataHandler import LoadDataset, LoadMultiPressureDataset
from src.PlottingTools import PlottingTools

# make directory t save multiple models
import os
if not os.path.exists('checkpoints/seeds_checkpoints'):
    os.makedirs('checkpoints/seeds_checkpoints')


# recover reproducibility
torch.manual_seed(8)

# Specify device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Specify number of surrogate models and densities
n_surrog = 3 # number of surrogate models 
n_param = 3 # number of input densities
k_columns = [0,1,2]

# Define the model parameters
input_size = 3 # number of input densities
output_size = 3  # number of coefficients
hidden_size = (10,10)  # architecture of the main model
max_epoch = 200

# Initialize your model
plotter = PlottingTools()

# Initialize your model
model = NSurrogatesModel(input_size, output_size, hidden_size, n_surrog)

# Load surrogate datasets
datasets = [LoadDataset(src_file=f"data/datapoints_pressure_{i}.txt", nspecies=3, react_idx=k_columns) for i in range(n_surrog)]

# Load main net datasets
main_dataset = LoadMultiPressureDataset(src_file="data/datapoints_mainNet_2k.txt", nspecies=3, num_pressure_conditions=n_surrog, react_idx=k_columns,
                                         scaler_input=[datasets[i].scaler_input for i in range(n_surrog)], scaler_output=[datasets[i].scaler_output for i in range(n_surrog)])
main_dataset_test = LoadMultiPressureDataset(src_file="data/datapoints_mainNet_test.txt", nspecies=3, num_pressure_conditions=n_surrog, react_idx=k_columns,\
                            scaler_input=main_dataset.scaler_input, scaler_output=main_dataset.scaler_output, m_rows=3000)

# Specify loss function
criterion = MSELoss()

# Specify optimizer with parameters for main net (surrogate models are frozen)
optimizer = Adam(model.main_net.parameters(), lr=0.1)

# --------------------   Loading   -------------------- #

# Create trainer
trainer = NSurrogatesModelTrainer(model, datasets, device, criterion, optimizer)

# Load surrogate models
trainer.load_surrogate_models()

# Load main net
seed_idx = 7
trainer.model.main_net.load_model(f"seeds_checkpoints/main_model_seed{seed_idx}.pth")

# -------------------   Evaluation   ------------------- #

# Move model to cpu
if device.type == 'cuda':  
    model = model.to('cpu')

# Get main net
main_net = trainer.model.main_net

# Plot validation of main net
plotter.plot_predictions_main(model, main_dataset_test, filename="predictions_vs_true_values_main.png")

plotter.get_relative_error(model, main_dataset_test)