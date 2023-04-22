import numpy as np
import matplotlib.pyplot as plt
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
import torch as T
import torch.nn as nn
device = T.device("cpu")


# Define the transform to normalize data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),])
  # normalizes all the values in the tensor so that they lie between 0.5 and 1.

# Loading train set
mnist_trainset = datasets.MNIST(root='./data', train=True, download=False, transform=transform) #only dowload once
train_loader = T.utils.data.DataLoader(mnist_trainset, batch_size=100, shuffle=True) 

# Loading test set
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform) #only dowload once
test_loader = T.utils.data.DataLoader(mnist_testset, batch_size=100, shuffle=True)


#---------------------------------------------------------------------------------------

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # The Linear() class defines a fully connected network layer
        self.hid1 = nn.Linear(28*28, 50)  # hidden 1
        self.hid2 = nn.Linear(50, 50) # hidden 2
        self.hid3 = nn.Linear(50, 50) # hidden 3
        self.hid4 = nn.Linear(500, 50) # hidden 4
        self.oupt = nn.Linear(50, 10)  # output

    # Missing initialization of weights
    #T.nn.init.uniform_(self.hid1.weight, -0.05, +0.05)

    def forward(self, x):
        x = x.view(-1, 28*28)
        z = T.tanh(self.hid1(x)) # try also relu activ. f.
        z = T.tanh(self.hid2(z))
        z = T.tanh(self.hid3(z))
        z = T.tanh(self.hid4(z))
        z = self.oupt(z)  # no activation
        return z

#------------------------------------------------------------------------------------

# 2. create network
net = Net().to(device)

# 3. train model
max_epochs = 20
ep_log_interval = 1
lrn_rate = 0.001

#loss_func = T.nn.MSELoss()
loss_func = T.nn.CrossEntropyLoss()
#optimizer = T.optim.SGD(net.parameters(), lr=lrn_rate)
optimizer = T.optim.Adam(net.parameters(), lr=lrn_rate)

epoch_list, epoch_loss_list, test_loss_list = [], [], []

print("Started training \n")
for epoch in range(0, max_epochs):
    T.manual_seed(1)  # recovery reproducibility in initialization of weights
    epoch_loss = 0  # for one full epoch
    test_loss = 0

    net.train()  # set mode

    for (batch_idx, batch) in enumerate(train_loader):
      (X_batch, Y_batch) = batch           # (predictors, targets)
      optimizer.zero_grad()                # prepare gradients (set to zero before each loss calculation)
      oupt = net(X_batch.view(-1, 28*28))  # predicted prices (+ reshape of input)
      loss_val = loss_func(oupt, Y_batch)  # avg per item in batch
      epoch_loss += loss_val.item()        # accumulate avgs
      loss_val.backward()                  # compute gradients
      optimizer.step()                     # update wts
      n_batches = batch_idx+1

    #--------------------------------------------------------------
    # Print and save loss and errors
    if epoch % ep_log_interval == 0:
      epoch_list.append(epoch)
      epoch_loss_list.append(epoch_loss/n_batches)

      net.eval()
      with T.no_grad():
        for (idx, data) in enumerate(test_loader):
          x, y = data
          prediction = net(x.view(-1, 28*28))
          loss_val = loss_func(prediction, y)
          test_loss += loss_val.item()
          n_batches = idx+1
        test_loss_list.append(test_loss/n_batches)

      print("epoch = %4d   loss = %0.4f   test_loss= %0.4f" % \
       (epoch, epoch_loss_list[-1], test_loss_list[-1]))
    #--------------------------------------------------------------
 
print("Done \n")

net.eval() # set mode

# Compute accuracy 
correct, total = 0, 0
with T.no_grad():
    for data in test_loader:
        (x, y) = data
        output = net(x.view(-1, 28*28))
        for idx, i in enumerate(output):
            if T.argmax(i) == y[idx]:
                correct +=1
            total +=1
print(f'accuracy: {round(correct/total, 3)}')


# Plot Loss Curves
plt.plot(epoch_list, epoch_loss_list, '-o', label = 'train')
plt.plot(epoch_list, test_loss_list, '-o', label = 'validation')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.savefig('C:\\Users\\clock\\Desktop\\Python\\Images\\MNIST_loss_curve.png')
