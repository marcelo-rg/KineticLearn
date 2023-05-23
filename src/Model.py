import NeuralNetworkModels

class NSurrogatesModel():
    def __init__(self, input_size, output_size, hidden_size, n_surrog):
        self.main_net = NeuralNetworkModels.NeuralNet(input_size, output_size, hidden_size, activ_f = "tanh", out_activ_f = "sigmoid")
        self.n_surrog = n_surrog
        self.surrog_nets = []
        for i in range(n_surrog):
            surrog_net = NeuralNetworkModels.NeuralNet(output_size, input_size, hidden_size = (100,), activ_f = "relu")
            self.surrog_nets.append(surrog_net)