import rnvp
import torch
from torch import nn
import numpy as np
from torchvision import datasets, transforms
import itertools
import matplotlib.pyplot as plt
device = torch.device('cpu')

INPUT_DIM = 2 # The dimension of the embeddings
FLOW_N = 9 # Number of affine coupling layers
RNVP_TOPOLOGY = [200] # Size of the hidden layers in each coupling layer
AE_EPOCHS = 0 # Epochs for training the autoencoder
NF_EPOCHS = 20 # Epochs for training the normalizing flow
SEED = 0 # Seed of the random number generator
BATCH_SIZE = 100 # Batch size

# Set the random seeds
torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # The Linear() class defines a fully connected network layer
        self.hid1 = nn.Linear(3,10)  # hidden 1
        self.hid2 = nn.Linear(10, 10) # hidden 2
        # self.hid3 = nn.Linear(10, 10) # hidden 3
        self.oupt = nn.Linear(10, 3)  # output
        torch.nn.init.xavier_uniform_(self.hid1.weight)
        torch.nn.init.xavier_uniform_(self.hid2.weight)
        # torch.nn.init.xavier_uniform(self.hid3.weight)


    def forward(self, x):
        z = torch.relu(self.hid1(x)) # try also relu activ. f.
        z = torch.relu(self.hid2(z))
        # z = torch.tanh(self.hid3(z))
        z = self.oupt(z)  # no activation
        return z

n_trainSet = 100
def linear_ill_posed(x1, x2, x3):
    y1 = x1 + x3
    y2 = x2
    # z = x1 - x3
    return np.array([y1, y2])

x = np.random.uniform(0, 1, size = (3, n_trainSet))
y = linear_ill_posed(*x)
# print(y)
# print(np.shape(array_random))

# 2. Create neural network
net = Net().to(device)
net.to(torch.double) # set model to float64

# 3. Build training Model
max_epochs = 1000
ep_log_interval =20
lrn_rate = 0.001

# 4. Choose loss and optimizer
loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lrn_rate) # , weight_decay=1e-4)

data_ = []
for i in range(len(y.T)):
            data_.append((y.T[i], x.T[i]))
data_loader = torch.utils.data.DataLoader(data_, BATCH_SIZE)

# -----------------------INN------------------------------
# See the file realmvp.py for the full definition
nf_model = rnvp.LinearRNVP(input_dim=INPUT_DIM, coupling_topology=RNVP_TOPOLOGY, flow_n=FLOW_N, batch_norm=True,
                      mask_type='odds', conditioning_size=10, use_permutation=True, single_function=True)
nf_model = nf_model.to(device)
nf_model.to(torch.double)

optimizer1 = torch.optim.Adam(itertools.chain(nf_model.parameters()), lr=1e-4, weight_decay=1e-5)

nf_model.train()
for i in range(max_epochs):
    print('Epoch #{}'.format(i+1))

    losses = []
    for batch_idx, data in enumerate(data_loader):

        emb, y = data
        # print(np.shape(emb), np.shape(y))
        # exit()
        emb = emb.to(device)
        y = y.to(device)
        # y = torch.nn.functional.one_hot(y, 10).to(device).float()
        
        # Get the inverse transformation and the corresponding log determinant of the Jacobian
        u, log_det = nf_model.forward(emb, y=y) 

        # Train via maximum likelihood
        prior_logprob = nf_model.logprob(u)
        log_prob = -torch.mean(prior_logprob.sum(1) + log_det)
        epoch_loss = log_prob.item()
        nf_model.zero_grad()

        log_prob.backward()

        optimizer1.step()
        n_batches = batch_idx+1

    # Print and save loss and errors
    if (i+1 % ep_log_interval) == 0:
        print("epoch = %4d   loss = %0.4f " % \
        (i+1, epoch_loss/n_batches))




# 5. Training algorithm
# print("Start training\n")
# for epoch in range(0, max_epochs):
#     epoch_loss = 0  # for one full epoch

#     net.train()  # set mode

#     for (batch_idx, batch) in enumerate(data_loader):
#         (X_batch, Y_batch) = batch           # (predictors, targets)
#         optimizer.zero_grad()                # prepare gradients
#         oupt = net(X_batch)                  # predicted rate coefficients
#         loss_val_loki = 16*loss_func(oupt, Y_batch)  
#         # loss_val_mse = loss_mse(oupt, Y_batch)
#         loss_val = loss_val_loki #+ loss_val_mse  # avg per item in batch
#         epoch_loss += loss_val.item()        # accumulate avgs
#         loss_val.backward()                  # compute gradients
#         optimizer.step()                     # update wts
#         n_batches = batch_idx+1              # save number of batches

#     #-------------------------------------------------------------
#     # Print and save loss and errors
#     if (epoch % ep_log_interval) == 0:
#         # myplot.epoch_list.append(epoch)
#         # myplot.epoch_loss_list.append(loss_val_mse.item()/n_batches)
#         # myplot.epoch_loss_list_loki.append(loss_val_loki.item()/n_batches)

#         # net.eval() # (?)
#         # prediction = net(x_val)
#         # # loss_val = loss_func(prediction, y_val)
#         # loss_val = loss_func(prediction, x_val) #+ loss_func(prediction,y_val)
#         # myplot.val_loss_list.append(loss_val.item())

#         print("epoch = %4d   loss = %0.4f   validation_loss= %0.4f" % \
#         (epoch, epoch_loss/n_batches, loss_val.item()))
#     #--------------------------------------------------------------

print("Training complete \n")