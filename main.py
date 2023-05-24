import torch
from torch.nn import MSELoss
from torch.optim import Adam

from src.Model import NSurrogatesModel
from src.DataHandler import LoadDataset
from src.Trainer import NSurrogatesModelTrainer

# Specify device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Specify number of surrogate models and densities
n_surrog = 1  # example
n_param = 3 # number of densities

# Define the model parameters
input_size = n_surrog*n_param  # example
output_size = 3  # number of coefficients
hidden_size = (10,10)  # architecture of the main model

epochs = 100  # example

# Initialize your model
model = NSurrogatesModel(input_size, output_size, hidden_size, n_surrog)

# Load surrogate datasets
datasets = [LoadDataset(src_file=f"data/datapoints_pressure_{i}.txt", nspecies=3) for i in range(n_surrog)]

# Specify loss function and optimizer
criterion = MSELoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Create trainer
trainer = NSurrogatesModelTrainer(model, datasets, device, criterion, criterion, optimizer)

# Train surrogate models
trainer.train_surrg_models(epochs)

# Further training ...
