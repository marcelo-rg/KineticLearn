import torch
from torch.nn import MSELoss
from torch.optim import Adam

from Model import NSurrogatesModel
from DataHandler import LoadDataset
from Trainer import NSurrogatesModelTrainer

# Specify device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the model parameters
input_size = 3  # example
output_size = 3  # example
hidden_size = (100,)  # example
n_surrog = 2  # example
epochs = 10  # example

# Initialize your model
model = NSurrogatesModel(input_size, output_size, hidden_size, n_surrog)

# Load your datasets
datasets = [LoadDataset(src_file=f"data/datapoints_pressure_{i}.txt", nspecies=3) for i in range(n_surrog)]

# Specify loss function and optimizer
criterion = MSELoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Create trainer
trainer = NSurrogatesModelTrainer(model, datasets, device, criterion, criterion, optimizer)

# Train surrogate models
trainer.train_surrg_models(epochs)

# Further training can be done here
