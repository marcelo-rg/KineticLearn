import torch
from torch.nn import MSELoss
from torch.optim import Adam

from src.Model import NSurrogatesModel
from src.DataHandler import LoadDataset
from src.Trainer import NSurrogatesModelTrainer
from src.PlottingTools import PlottingTools

# recover reproducibility
torch.manual_seed(0)

# Specify device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Specify number of surrogate models and densities
n_surrog = 1  # example
n_param = 3 # number of input densities
k_columns = [0,1,2]

# Define the model parameters
input_size = n_surrog*n_param  # example
output_size = 3  # number of coefficients
hidden_size = (10,10)  # architecture of the main model
max_epoch = 50  # example

# Initialize your model
model = NSurrogatesModel(input_size, output_size, hidden_size, n_surrog)

# Load surrogate datasets
datasets = [LoadDataset(src_file=f"data/datapoints_pressure_{i}.txt", nspecies=3, react_idx=k_columns) for i in range(n_surrog)]

# Specify loss function
criterion = MSELoss()

# Specify optimizer with parameters of all models
optimizer = Adam(model.parameters(), lr=0.0001)

# Create trainer
trainer = NSurrogatesModelTrainer(model, datasets, device, criterion, criterion, optimizer)

# Train surrogate models
training_losses, validation_losses = trainer.train_surrg_models(max_epoch)

# Further training ...

# for evaluation, move model to cpu
if device.type == 'cuda':  
    model = model.to('cpu')

# Plot training and validation loss histories
plotter = PlottingTools()
plotter.plot_loss_history(training_losses, validation_losses)

# Get first surrogate model
surrogate_model = model.surrog_nets[0]

# Plot validation using test dataset
test_dataset = LoadDataset(src_file="data/datapoints_pressure_0_test.txt", nspecies=3, react_idx=k_columns)
plotter.plot_predictions(surrogate_model, test_dataset)