import torch
from torch.nn import MSELoss
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
    
    def __init__(self, model, datasets, device, surrog_criterion, optimizer, batch_size=20, val_split=0.1):
        self.model = model
        self.device = device
        self.surrog_criterion = surrog_criterion
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.train_dataloaders = []
        self.val_dataloaders = []
        self.val_split = val_split
        
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
        Trains the surrogate models for a specified number of epochs and returns the training and validation loss histories.

        Each epoch consists of a training phase and a validation phase. In the training phase, the surrogate models 
        are updated to reduce the loss on the training data. In the validation phase, the loss on the validation data is 
        calculated, but the models are not updated. The training and validation losses for each epoch are printed out 
        and stored in dictionaries that are returned by this method.

        Args:
            epochs (int): The number of epochs to train for.

        Returns:
            dict: A dictionary mapping each surrogate model to a list of its training losses over all epochs.
            dict: A dictionary mapping each surrogate model to a list of its validation losses over all epochs.
        """

        # Initialize dictionaries to store the loss history
        training_losses = {f'surrogate_{i}': [] for i in range(self.model.n_surrog)}
        validation_losses = {f'surrogate_{i}': [] for i in range(self.model.n_surrog)}
        for epoch in range(epochs):

            # Training phase
            for i, (surrog_net, train_dataloader) in enumerate(zip(self.model.surrog_nets, self.train_dataloaders)):
                surrog_net.train()  # Set the model to training mode
                epoch_loss = 0.0  # Initialize epoch loss for each surrogate network
                for x_batch, y_batch in train_dataloader:
                    # Move data to device
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    # Forward pass
                    output = surrog_net(x_batch)

                    # Compute loss
                    loss = self.surrog_criterion(output, y_batch)
                    epoch_loss += loss.item()  # accumulate avgs

                    # Backward pass and optimization
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # Add loss to history
                training_losses[f'surrogate_{i}'].append(epoch_loss)

            # Validation phase
            with torch.no_grad():  # Disable gradient calculation
                for i, (surrog_net, val_dataloader) in enumerate(zip(self.model.surrog_nets, self.val_dataloaders)):
                    surrog_net.eval()  # Set the model to evaluation mode
                    epoch_val_loss = 0.0  
                    for x_batch, y_batch in val_dataloader:
                        # Move data to device
                        x_batch = x_batch.to(self.device)
                        y_batch = y_batch.to(self.device)

                        # Forward pass
                        output = surrog_net(x_batch)

                        # Compute loss
                        val_loss = self.surrog_criterion(output, y_batch)
                        epoch_val_loss += val_loss.item()  # accumulate avgs

                    # Add loss to history
                    validation_losses[f'surrogate_{i}'].append(epoch_val_loss*(1-self.val_split)/self.val_split)

            # Print the training and validation losses for this epoch
            print(f'Epoch {epoch+1}/{epochs}, Training Loss: {epoch_loss}, Validation Loss: {epoch_val_loss*(1-self.val_split)/self.val_split}')

        # Return the loss history
        return training_losses, validation_losses

    

    def train_main_model(self, main_dataset, epochs):
        """
        Trains the main neural network model for a specified number of epochs and returns the training and validation loss histories.

        Each epoch consists of a training phase and a validation phase. In the training phase, the main model is updated
        to reduce the loss on the training data. In the validation phase, the loss on the validation data is calculated,
        but the model is not updated. The training and validation losses for each epoch are printed out and stored in
        dictionaries that are returned by this method.

        Args:
            main_dataset (LoadDataset): The main dataset containing input (densities) and target (coefficients) data.
            epochs (int): The number of epochs to train for.

        Returns:
            dict: A dictionary mapping the main model to a list of its training losses over all epochs.
            dict: A dictionary mapping the main model to a list of its validation losses over all epochs.
        """
        # Calculate split sizes
        train_size = int((1.0 - self.val_split) * len(main_dataset))
        val_size = len(main_dataset) - train_size

        # Split the dataset
        train_dataset, val_dataset = random_split(main_dataset, [train_size, val_size])

        # Create dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size=1000, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=1000, shuffle=True)

        def main_criterion(output, target):
            """
            Calculates the main criterion loss as the sum of mean squared errors (MSE) between the surrogate outputs and targets.

            Args:
                output (torch.Tensor): The surrogate outputs of shape (batch_size, num_pressure_conditions, output_size).
                target (torch.Tensor): The targets of shape (batch_size, num_pressure_conditions, output_size).

            Returns:
                torch.Tensor: The main criterion loss.
            """
            loss_func = MSELoss()
            loss = 0.0
            num_pressure_conditions = output.shape[1]
            # print(num_pressure_conditions)
            # print(output.shape)
            # print(target.shape)
            # exit()

            for i in range(num_pressure_conditions):
                surrogate_output = output[:, i, :]  # Surrogate output for the i-th pressure condition
                target_i = target[:, i, :]  # Target for the i-th pressure condition
                loss += loss_func(surrogate_output, target_i)

            return loss

        # Initialize dictionaries to store the loss history
        training_losses = {'main_model': []}
        validation_losses = {'main_model': []}

        for epoch in range(epochs):
            main_model = self.model.main_net

            # Training phase
            main_model.train()  # Set the main model to training mode
            epoch_loss = 0.0  # Initialize epoch loss for the main model

            for _, y_batch in train_dataloader:
                # Move data to device
                y_batch = y_batch.to(self.device)


                # Forward pass through the main model
                main_input = y_batch.flatten(start_dim=1)
                main_output = main_model(main_input)   

                # Compute loss
                surrogate_outputs = []

                for surrogate_net in self.model.surrog_nets:
                    surrogate_net.eval()
                    # Forward pass through the surrogate model
                    surrogate_output = surrogate_net(main_output)
                    surrogate_outputs.append(surrogate_output)  # Set requires_grad to True

                # Stack surrogate outputs along the second dimension
                surrogate_outputs = torch.stack(surrogate_outputs, dim=1)

                loss = main_criterion(surrogate_outputs, y_batch)

                # Accumulate loss
                epoch_loss += loss

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                # freeze surrogate models
                # self.freeze_surrogate_models()
                self.optimizer.step()

            # Add loss to history
            training_losses['main_model'].append(epoch_loss.item())


            # Validation phase
            main_model.eval()  # Set the main model to evaluation mode
            epoch_val_loss = 0.0

            with torch.no_grad():
                for _, y_batch in val_dataloader:
                    # Move data to device
                    y_batch = y_batch.to(self.device)

                    # Forward pass through the main model
                    main_input = y_batch.flatten(start_dim=1)
                    main_output = main_model(main_input)

                    # Compute surrogate outputs for all surrogate models
                    surrogate_outputs = []
                    for surrogate_net in self.model.surrog_nets:
                        surrogate_net.eval()  
                        surrogate_output = surrogate_net(main_output)
                        surrogate_outputs.append(surrogate_output)

                    # Stack surrogate outputs along the second dimension
                    surrogate_outputs = torch.stack(surrogate_outputs, dim=1)  

                    # Compute loss
                    val_loss = main_criterion(surrogate_outputs, y_batch)

                    # Accumulate loss
                    epoch_val_loss += val_loss.item()

            # Add loss to history
            validation_losses['main_model'].append(epoch_val_loss * (1 - self.val_split) / self.val_split)

            # Print the training and validation losses for this epoch
            print(f'Epoch {epoch+1}/{epochs}, Training Loss: {epoch_loss}, Validation Loss: {epoch_val_loss*(1-self.val_split)/self.val_split}')

        return training_losses, validation_losses

    def freeze_surrogate_models(self):
        for surrog_net in self.model.surrog_nets:
            for param in surrog_net.parameters():
                param.requires_grad = False
    
    def unfreeze_surrogate_models(self):
        for surrog_net in self.model.surrog_nets:
            for param in surrog_net.parameters():
                param.requires_grad = True


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

