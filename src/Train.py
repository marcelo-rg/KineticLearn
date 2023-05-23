import torch
from torch.utils.data import random_split, DataLoader

class NSurrogatesModelTrainer:
    def __init__(self, model, datasets, device, main_criterion, surrog_criterion, optimizer, batch_size=20, val_split=0.1):
        self.model = model
        self.device = device
        self.main_criterion = main_criterion
        self.surrog_criterion = surrog_criterion
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.train_dataloaders = []
        self.val_dataloaders = []
        
        for dataset in datasets:
            # Calculate split sizes
            train_size = int((1.0 - val_split) * len(dataset))
            val_size = len(dataset) - train_size
            
            # Split the dataset
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            
            # Create dataloaders
            train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)
            
            # Store dataloaders
            self.train_dataloaders.append(train_dataloader)
            self.val_dataloaders.append(val_dataloader)
    
    
    def train_surrg_models(self, epochs):
        for epoch in range(epochs):
            # Training phase
            for surrog_net, train_dataloader in zip(self.model.surrog_nets, self.train_dataloaders):
                surrog_net.train()  # Set the model to training mode
                for x, y in train_dataloader:
                    # Move data to device
                    x_batch = x.to(self.device)
                    y_batch = y.to(self.device)

                    # Forward pass
                    output = surrog_net(x)

                    # Compute loss
                    loss = self.surrog_criterion(output, y)

                    # Backward pass and optimization
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            # Validation phase
            with torch.no_grad():  # Disable gradient calculation
                for surrog_net, val_dataloader in zip(self.model.surrog_nets, self.val_dataloaders):
                    surrog_net.eval()  # Set the model to evaluation mode
                    for x, y in val_dataloader:
                        # Move data to device
                        x_batch = x.to(self.device)
                        y_batch = y.to(self.device)

                        # Forward pass
                        output = surrog_net(x)

                        # Compute loss
                        val_loss = self.surrog_criterion(output, y)
                        
            # Print the training and validation losses for this epoch
            print(f'Epoch {epoch+1}/{epochs}, Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}')
