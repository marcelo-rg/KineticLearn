from src.NeuralNetworkModels import NeuralNet

class NSurrogatesModel():
    """
    A class that represents a model composed of multiple surrogate neural networks and a main neural network.
    
    This class is used to manage multiple surrogate models and a main model. The surrogate models and the main model 
    are all instances of the NeuralNet class, each with their own input and output sizes, hidden layer sizes, and 
    activation functions.
    
    Attributes:
        main_net (NeuralNet): The main neural network.
        n_surrog (int): The number of surrogate networks.
        surrog_nets (list): The list of surrogate networks.

    Args:
        input_size (int): The number of input features for the main neural network.
        output_size (int): The number of output units for the main neural network.
        hidden_size (tuple): The size of the hidden layers for the main neural network. Each element represents 
                             the size of a specific hidden layer.
        n_surrog (int): The number of surrogate networks to be created.
    """
    def __init__(self, input_size, output_size, hidden_size, n_surrog):
        self.main_net = NeuralNet(input_size, output_size, hidden_size, activ_f = "tanh", out_activ_f = "sigmoid")
        self.n_surrog = n_surrog
        self.surrog_nets = []
        for i in range(n_surrog):
            surrog_net = NeuralNet(output_size, input_size, hidden_size = (100,), activ_f = "relu")
            self.surrog_nets.append(surrog_net)
        
    def add_surrogate(self, input_size, output_size, hidden_size= (100,), activ_f = "relu"):
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
        surrog_net = NeuralNet(input_size, output_size, hidden_size =hidden_size, activ_f=activ_f)
        self.surrog_nets.append(surrog_net)
        self.n_surrog += 1
