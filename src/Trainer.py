import torch
from torch.utils.data import random_split, DataLoader

class NSurrogatesModelTrainer:
    """
    A class that represents a trainer for an NSurrogatesModel.

    This class is used to manage the training and validation of multiple surrogate models and a main model 
    for different datasets.

    Attributes:
        model (NSurrogatesModel): The model to train.
        device (str): The device to use for training (typically either "cpu" or "cuda").
        main_criterion (nn.Module): The loss function to use for training the main model.
        surrog_criterion (nn.Module): The loss function to use for training the surrogate models.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        batch_size (int): The batch size to use for training.
        train_dataloaders (list): List of DataLoader instances for the training data for each surrogate model.
        val_dataloaders (list): List of DataLoader instances for the validation data for each surrogate model.

    Args:
        model (NSurrogatesModel): The model to train.
        datasets (list): List of Dataset instances for each surrogate model.
        device (str): The device to use for training (typically either "cpu" or "cuda").
        main_criterion (nn.Module): The loss function to use for training the main model.
        surrog_criterion (nn.Module): The loss function to use for training the surrogate models.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        batch_size (int, optional): The batch size to use for training. Default is 20.
        val_split (float, optional): The proportion of the dataset to include in the validation split. Default is 0.1.
    """
    
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
        """
        Trains the surrogate models for a specified number of epochs.

        Each epoch consists of a training phase and a validation phase. In the training phase, the surrogate models 
        are updated to reduce the loss on the training data. In the validation phase, the loss on the validation data is 
        calculated, but the models are not updated. The training and validation losses for each epoch are printed out.

        Args:
            epochs (int): The number of epochs to train for.
        """

        for epoch in range(epochs):
            # Training phase
            for surrog_net, train_dataloader in zip(self.model.surrog_nets, self.train_dataloaders):
                surrog_net.train()  # Set the model to training mode
                for x_batch, y_batch in train_dataloader:
                    # Move data to device
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    # Forward pass
                    output = surrog_net(x_batch)

                    # Compute loss
                    loss = self.surrog_criterion(output, y_batch)

                    # Backward pass and optimization
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            # Validation phase
            with torch.no_grad():  # Disable gradient calculation
                for surrog_net, val_dataloader in zip(self.model.surrog_nets, self.val_dataloaders):
                    surrog_net.eval()  # Set the model to evaluation mode
                    for x_batch, y_batch in val_dataloader:
                        # Move data to device
                        x_batch = x_batch.to(self.device)
                        y_batch = y_batch.to(self.device)

                        # Forward pass
                        output = surrog_net(x_batch)

                        # Compute loss
                        val_loss = self.surrog_criterion(output, y_batch)
                        
            # Print the training and validation losses for this epoch
            print(f'Epoch {epoch+1}/{epochs}, Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}')




    def add_surrogate_and_dataset(self, surrogate_params, dataset):
        """
        Add a new surrogate network along with its corresponding dataset to the model.

        Args:
            surrogate_params (dict): A dictionary with the parameters required for the creation of the surrogate model.
                                     It should contain the following keys: 'input_size', 'output_size', 'hidden_size', 
                                     and 'activ_f' (optional).
            dataset (Dataset): The PyTorch Dataset object that contains the training data for the new surrogate model.
        """
        self.model.add_surrogate(**surrogate_params)
        self.add_dataset(dataset)
        
    def add_dataset(self, dataset):
        """
        Add a new dataset to the list of datasets used for training the surrogate models.

        Args:
            dataset (Dataset): The PyTorch Dataset object to be added.
        """
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.dataloaders.append(dataloader)
        self.datasets.append(dataset)

