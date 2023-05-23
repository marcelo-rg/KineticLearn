import numpy as np
import torch 
import torch.nn as nn
# torch.set_default_tensor_type(torch.DoubleTensor)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------

class NeuralNet(nn.Module):
    """
    A custom neural network module with configurable hidden layers and activation functions.

    Args:
        input_size (int): The number of input features.
        output_size (int): The number of output units.
        hidden_size (int or tuple, optional): The size of the hidden layers. If it's an integer, all hidden layers will have the same size. If it's a tuple, each element represents the size of a specific hidden layer. Default is (10, 10).
        activ_f (str, optional): The activation function for the hidden layers. Available options are 'tanh' and 'relu'. Default is 'tanh'.
        out_activ_f (str or None, optional): The activation function for the output layer. Available options are 'sigmoid' and 'tanh'. If None, no activation function is applied to the output layer. Default is None.

    Attributes:
        model (torch.nn.Sequential): The sequential model representing the neural network architecture.

    """
    def __init__(self, input_size, output_size, hidden_size=(10, 10), activ_f='tanh', out_activ_f=None):
        super(NeuralNet, self).__init__()

        # Check if the hidden_size is a single value or a tuple
        if isinstance(hidden_size, int):
            hidden_size = (hidden_size,)

        # Create a list to hold the layers
        layers = []

        # Add the input layer
        layers.append(nn.Linear(input_size, hidden_size[0]))
        if activ_f == 'tanh':
            layers.append(nn.Tanh())
        elif activ_f == 'relu':
            layers.append(nn.ReLU())
        else:
            raise ValueError("Invalid activation function: {}".format(activ_f))

        # Add the hidden layers
        for i in range(len(hidden_size) - 1):
            layers.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))
            if activ_f == 'tanh':
                layers.append(nn.Tanh())
            elif activ_f == 'relu':
                layers.append(nn.ReLU())
            else:
                raise ValueError("Invalid activation function: {}".format(activ_f))

        # Add the output layer
        layers.append(nn.Linear(hidden_size[-1], output_size))

        if out_activ_f is not None:
            if out_activ_f == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif out_activ_f == 'tanh':
                layers.append(nn.Tanh())
            else:
                raise ValueError("Invalid activation function: {}".format(out_activ_f))

        # Create the sequential model using the layers
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass of the neural network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor produced by the neural network.

        """
        return self.model(x)

if __name__ == "__main__":
    print("NeuralNetworkModels.py")
    net = NeuralNet(4, 3, hidden_size=(100, 100), activ_f='tanh', out_activ_f=None)
    print(net)