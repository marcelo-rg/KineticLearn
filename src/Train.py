import torch
from torch.utils.data import DataLoader

class NSurrogatesModelTrainer:
    def __init__(self, model, datasets, device, main_criterion, surrog_criterion, optimizer, batch_size=20):
        self.model = model
        self.datasets = datasets
        self.device = device
        self.main_criterion = main_criterion
        self.surrog_criterion = surrog_criterion
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.dataloaders = [DataLoader(dataset, batch_size=batch_size, shuffle=True) for dataset in datasets]
    
    def train_surrg_models(self, epochs):
        for epoch in range(epochs):
            for surrog_net, dataloader in zip(self.model.surrog_nets, self.dataloaders):
                for x, y in dataloader:
                    # Move data to device
                    x = x.to(self.device)
                    y = y.to(self.device)

                    # Forward pass
                    output = surrog_net(x)

                    # Compute loss
                    loss = self.surrog_criterion(output, y)

                    # Backward pass and optimization
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
