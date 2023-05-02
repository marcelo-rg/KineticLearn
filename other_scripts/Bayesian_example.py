import numpy as np
from sklearn import datasets

import torch
import torch.nn as nn
import torch.optim as optim

import torchbnn as bnn
import matplotlib.pyplot as plt
device = torch.device("cpu")

torch.manual_seed(10)

x = torch.linspace(-2, 2, 500)
y = x.pow(3) - x.pow(2) + 3*torch.rand(x.size())

#print("before: ", x)

x = torch.unsqueeze(x, dim=1) # what is this
y = torch.unsqueeze(y, dim=1)

#print("after: ", x)

#plt.plot(x, y, "o",label= 'data')
#plt.show()

# Define neural network structure
class BayesianNet(nn.Module):
    def __init__(self):
        super(BayesianNet, self).__init__()
        # The Linear() class defines a fully connected network layer
        self.hid1 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
          in_features=1, out_features=10)  # hidden 1
        self.oupt = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
          in_features=10, out_features=1)  # output

    # Missing initialization of weights
    #T.nn.init.uniform_(self.hid1.weight, -0.05, +0.05)

    def forward(self, x):
        z = torch.tanh(self.hid1(x)) # try also relu activ. f.
        z = self.oupt(z)  # no activation
        return z


net = BayesianNet().to(device)
#net.to(torch.double) # set model to float64

# Define training model
mse_loss = nn.MSELoss()
kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
kl_weight = 0.01

max_epochs = 3000
ep_log_interval = 10
lrn_rate = 0.01

optimizer = optim.Adam(net.parameters(), lr=lrn_rate)

net.train() # set mode

# Training
epoch_list=[]
epoch_loss_list=[]
for step in range(max_epochs):
    pre = net(x)
    mse = mse_loss(pre, y)
    kl = kl_loss(net)
    cost = mse + kl_weight*kl
    epoch_list.append(step)
    epoch_loss_list.append(cost.item())
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if step % ep_log_interval == 0:
        print("epoch = %4d   loss = %0.4f" % (step, cost))

net.eval() # set mode
    
print('- MSE : %2.2f, KL : %2.2f' % (mse.item(), kl.item()))


# Plot loss curves
plt.clf()
plt.plot(epoch_list, epoch_loss_list, '-o', label = 'train')
#plt.plot(epoch_list, test_loss_list, '-o', label = 'validation')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.savefig('C:\\Users\\clock\\Desktop\\Python\\Images\\Bayesian\\loss_curve_example.png')


# Test the model

x_test = torch.linspace(-2, 2, 500)
y_test = x_test.pow(3) - x_test.pow(2) + 3*torch.rand(x_test.size())

x_test = torch.unsqueeze(x_test, dim=1)
y_test = torch.unsqueeze(y_test, dim=1)


plt.clf()
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')

plt.scatter(x_test.data.numpy(), y_test.data.numpy(), color='k', s=2) 

y_predict = net(x_test)
plt.plot(x_test.data.numpy(), y_predict.data.numpy(), 'r-', linewidth=5, label='First Prediction')

y_predict = net(x_test)
plt.plot(x_test.data.numpy(), y_predict.data.numpy(), 'b-', linewidth=5, label='Second Prediction')

y_predict = net(x_test)
plt.plot(x_test.data.numpy(), y_predict.data.numpy(), 'g-', linewidth=5, label='Third Prediction')

plt.legend()

plt.show()


