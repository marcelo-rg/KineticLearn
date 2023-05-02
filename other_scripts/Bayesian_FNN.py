import metrics_normalization as mn
import numpy as np
import math
import matplotlib.pyplot as plt
import torch as T
import torch.nn as nn
import torchbnn as bnn
device = T.device("cpu")
from mlxtend.preprocessing import standardize


# -----------------------------------------------------------
class LoadDataset(T.utils.data.Dataset):
  # last 3 columns: densities (input)
  # first 9 columns: k's  (output)

  def __init__(self, src_file, m_rows=None):
    all_xy = np.loadtxt(src_file, max_rows=m_rows,
      usecols=[0,1,2,3,4,5,6,7,8,9,10,11], delimiter='  ',
      # usecols=range(0,9), delimiter="\t",
      comments="#", skiprows=0, dtype=np.float64)

    # list =[]
    # with open(src_file, 'r') as file:
    #   for line in file:
    #     line_values = line.split()
    #     list.append(line_values)
    
    # array = np.array(map(np.float64, list))
    # print(array)
  
    #print("after loadtxt tensor:\n ", all_xy[:,0])  

    tmp_x = all_xy[:,[9,10,11]] 
    tmp_y = all_xy[:,[0,1,2]] 
    #[0,1,2,3,4,5,6,7,8]

    # Normalize data
    tmp_x = mn.densitie_fraction(tmp_x)
    #tmp_x = standardize(tmp_x)
    #tmp_y = np.log(tmp_y)
    tmp_y = standardize(tmp_y)

    #print("before tensor:\n ", tmp_y[:,0])

    self.x_data = T.tensor(tmp_x, \
      dtype=T.float64).to(device)
    self.y_data = T.tensor(tmp_y, \
      dtype=T.float64).to(device)


  def __len__(self):
    return len(self.x_data)

  def __getitem__(self, idx):
    densities = self.x_data[idx,:]  # or just [idx]
    coef = self.y_data[idx,:] 
    return (densities, coef)       # tuple of two matrices 

# ------------------------------------------------------------------------------

class BayesianNet(nn.Module):
    def __init__(self):
        super(BayesianNet, self).__init__()
        self.hid1 = bnn.BayesLinear(prior_mu=0, prior_sigma=1.5,
          in_features=3, out_features=10)  # hidden 1
        self.hid2 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
             in_features=10, out_features=10)  # hidden 2
        self.oupt = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
          in_features=10, out_features=3)  # output

    # Missing initialization of weights
    #T.nn.init.uniform_(self.hid1.weight, -0.05, +0.05)

    def forward(self, x):
        z = T.tanh(self.hid1(x)) # try also relu activ. f.
        z = T.tanh(self.hid2(z))
        #z = T.tanh(self.hid3(z))
        #z = T.tanh(self.hid4(z))
        z = self.oupt(z)  # no activation
        return z


#------------------------------------------------------------------------------------

src_file = 'C:\\Users\\clock\\Desktop\\Python\\datapointsk1k2k3_500.txt' #datapointsk1k2k3_500
full_dataset = LoadDataset(src_file) #,m_rows=500) 


# x_data = full_dataset.x_data
# y_data = full_dataset.y_data

# to_plot = y_data[:,0].numpy()

# plt.hist(to_plot, edgecolor='black', bins=50)
# plt.clf()

# plt.plot(np.arange(0,len(to_plot),1), np.sort(to_plot))
# #print(np.sort(to_plot))
# plt.show()
# exit()


T.manual_seed(10) # set seed
np.random.seed(1) # just in case

# 2. create network
net = BayesianNet().to(device)
net.to(T.double) # set model to float64

# 3. train model
max_epochs = 3500
ep_log_interval = 10
lrn_rate = 0.01
kl_weight= 0
vl_weight = 0 #0.3 funcou
mse_weight = 1.0

mse_loss = T.nn.MSELoss()
kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False) #normalization
optimizer = T.optim.Adam(net.parameters(), lr=lrn_rate ) #, weight_decay=1e-4)


# Split into training and test sets
train_size = int(0.90 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = T.utils.data.random_split(full_dataset, [train_size, test_size])

# Create minibatch on training set
bat_size= 1000
train_loader = T.utils.data.DataLoader(train_dataset,
    batch_size=bat_size, shuffle=True) # set to True


# CHANGE THIS 
x_test = test_dataset[:][0]
y_test = test_dataset[:][1]

#--------------------------------------------------------------------------------------------

def my_loss(output, target):
    variance = T.var(output)
    #print(variance)
    loss = T.mean((output - target)**2)/variance + T.log(1+variance)
    return loss

#--------------------------------------------------------------------------------------------

epoch_list = []
epoch_loss_list = []
test_loss_list = []
print("Start training\n")
for epoch in range(0, max_epochs):

    epoch_loss = 0  # for one full epoch

    net.train()  # set mode

    for (batch_idx, batch) in enumerate(train_loader):
      (X_batch, Y_batch) = batch           # (predictors, targets)
      optimizer.zero_grad()                # prepare gradients
      oupt = net(X_batch)                  # predicted prices
      msel = mse_loss(oupt, Y_batch)       # avg per item in batch
      #print(msel)
      #print(my_loss(oupt, Y_batch))
      oupt_val = net(x_test)
      var_loss = my_loss(oupt_val, y_test)
      kll = kl_loss(net)
      loss_val = (mse_weight *msel) + (kl_weight * kll) + (vl_weight *var_loss)      # total loss
      epoch_loss += loss_val.item()        # accumulate avgs
      loss_val.backward()                  # compute gradients
      optimizer.step()                     # update wts
      n_batches = batch_idx+1

    #-------------------------------------------------------------
    # Print and save loss and errors
    if epoch % ep_log_interval == 0:
      epoch_list.append(epoch)
      epoch_loss_list.append(epoch_loss/n_batches)

      net.eval()
      prediction = net(x_test)
      loss_val = mse_loss(prediction, y_test) + (kl_weight * kll) # confirm that this is already computed previously
      test_loss_list.append(loss_val.item())

      print("epoch = %4d   loss = %0.4f   validation_loss= %0.4f" % \
       (epoch, epoch_loss/n_batches, loss_val.item()))
    #--------------------------------------------------------------
 
print("Done \n")

net.eval() # set mode


# Evaluation
train_predictions = net(train_dataset[:][0]).detach().numpy()
y_train = train_dataset[:][1].numpy()

test_predictions = net(x_test).detach().numpy()
y_test = y_test.numpy()

# Second evaluation of training data #-------------------------------------------------------------------------
train_predictions2 = net(train_dataset[:][0]).detach().numpy()
train_predictions3 = net(train_dataset[:][0]).detach().numpy()
#y_train = train_dataset[:][1].numpy()

#test_predictions2 = net(x_test).detach().numpy()
#y_test = y_test.numpy()

#-------------------------------------------------------------------------------------------------


# Plot ks of training set
for idx in range(len(train_predictions[0])):
    filename = 'C:\\Users\\clock\\Desktop\\Python\\Images\\Bayesian\\ks\\k' + str(idx+1)+'.png'
    plt.clf()
    plt.plot(np.arange(0,len(train_predictions),1), np.sort(train_predictions[:,idx]), 'ro',markersize=4. , label='predicted 1')
    plt.plot(np.arange(0,len(train_predictions),1), np.sort(train_predictions2[:,idx]), 'go', markersize=4. , label='predicted 2')
    plt.plot(np.arange(0,len(train_predictions),1), np.sort(train_predictions3[:,idx]), 'yo', markersize=4. , label='predicted 3')
    plt.plot(np.arange(0,len(train_predictions),1), np.sort(y_train[:,idx]), 'bo', markersize=4. , label= 'target')
    plt.legend()
    plt.title('k'+str(idx+1))
    plt.savefig(filename)




# plot k9 validation
plt.clf()
plt.plot(np.arange(0,len(test_predictions),1), np.sort(test_predictions[:,0]), 'ro', label='predicted')
plt.plot(np.arange(0,len(test_predictions),1),np.sort(y_test[:,0]), 'bo', label= 'target')
plt.legend()
plt.savefig('C:\\Users\\clock\\Desktop\\Python\\Images\\Bayesian\\k1_validation.png')


# Plot loss curves
plt.clf()
plt.plot(epoch_list, epoch_loss_list, '-o', label = 'train')
plt.plot(epoch_list, test_loss_list, '-o', label = 'validation')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.savefig('C:\\Users\\clock\\Desktop\\Python\\Images\\Bayesian\\loss_curve.png')


# Print mean rel. error
print("\n\nPrinting mean rel. err in validation:")
for idx, rel_err in enumerate(mn.rel_error(train_predictions, y_train)):
  print("k%d:   rel.err = %0.2f   " % \
      (idx+1, rel_err*100))

print("\n\nPrinting mean rel. err in training:")
for idx, rel_err in enumerate(mn.rel_error(test_predictions, y_test)):
  print("k%d:   rel.err = %0.2f   " % \
      (idx+1, rel_err*100))