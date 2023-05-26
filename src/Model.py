import torch.nn as nn

from src.NeuralNetworkModels import NeuralNet

class NSurrogatesModel(nn.Module):
    """
    A class that represents a model composed of multiple surrogate neural networks and a main neural network.
    
    This class is used to manage multiple surrogate models and a main model. The surrogate models and the main model 
    are all instances of the NeuralNet class, each with their own input and output sizes, hidden layer sizes, and 
    activation functions.
    
    Attributes:
        main_net (NeuralNet): The main neural network.
        n_surrog (int): The number of surrogate networks.
        surrog_nets (nn.ModuleList): The list of surrogate networks.

    Args:
        input_size (int): The number of input features for the main neural network.
        output_size (int): The number of output units for the main neural network.
        hidden_size (tuple): The size of the hidden layers for the main neural network. Each element represents 
                             the size of a specific hidden layer.
        n_surrog (int): The number of surrogate networks to be created.
    """
    def __init__(self, input_size, output_size, hidden_size, n_surrog):
        super(NSurrogatesModel, self).__init__()
        self.n_surrog = n_surrog
        self.main_net = NeuralNet(n_surrog*input_size, output_size, hidden_size, activ_f = "tanh", out_activ_f = "sigmoid").double()
        self.surrog_nets = nn.ModuleList()
        for i in range(n_surrog):
            surrog_net = NeuralNet(output_size, input_size, hidden_size = (100,), activ_f = "relu").double()
            self.surrog_nets.append(surrog_net)

    def forward(self, x):
        """
        Perform a forward pass on the main network.
        
        Args:
            x (torch.Tensor): The input data.
            
        Returns:
            torch.Tensor: The output of the network.
        """
        return self.main_net(x)

    def freeze_surrogates(self):
        """
        Freeze the parameters of the surrogate networks, preventing them from being updated during training.
        """
        for surrog in self.surrog_nets:
            for param in surrog.parameters():
                param.requires_grad = False

    def unfreeze_surrogates(self):
        """
        Unfreeze the parameters of the surrogate networks, allowing them to be updated during training.
        """
        for surrog in self.surrog_nets:
            for param in surrog.parameters():
                param.requires_grad = True

    def freeze_main(self):
        """
        Freeze the parameters of the main network, preventing them from being updated during training.
        """
        for param in self.main_net.parameters():
            param.requires_grad = False

    def unfreeze_main(self):
        """
        Unfreeze the parameters of the main network, allowing them to be updated during training.
        """
        for param in self.main_net.parameters():
            param.requires_grad = True

    def add_surrogate(self, input_size, output_size, hidden_size=(100,), activ_f="relu"):
        """
        Adds a new surrogate network to the model.

        Args:
            input_size (int): The number of input features for the new surrogate network.
            output_size (int): The number of output units for the new surrogate network.
            hidden_size (tuple, optional): The size of the hidden layers for the new surrogate network. 
                                            Each element represents the size of a specific hidden layer.
                                            Default is (100,).
            activ_f (str, optional): The activation function to use in the new surrogate network. Default is "relu".
        """
        surrog_net = NeuralNet(input_size, output_size, hidden_size=hidden_size, activ_f=activ_f)
        self.surrog_nets.append(surrog_net)
        self.n_surrog += 1
