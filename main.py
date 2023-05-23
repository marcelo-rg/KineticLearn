from src.Model import NSurrogatesModel
from src.DataHandler import LoadDataset
from src.Train import NSurrogatesModelTrainer

import torch 
import torch.nn as nn

# Specify device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize your model
model = NSurrogatesModel(input_size, output_size, hidden_size, n_surrog)

# Load your datasets
datasets = [LoadDataset(src_file=f"data/datapoints_pressure_{i}.txt", nspecies=3) for i in range(n_surrog)]

# Specify loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Create trainer
trainer = NSurrogatesModelTrainer(model, datasets, device, criterion, optimizer)

# Train model
trainer.train_model(epochs)
