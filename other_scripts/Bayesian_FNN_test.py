import metrics_normalization as mn
import numpy as np
import math
import matplotlib.pyplot as plt
import torch as T
import torch.nn as nn
import torchbnn as bnn
device = T.device("cpu")
from mlxtend.preprocessing import standardize


# # create tensors
# T1 = T.Tensor(([1,2,3,4],[1,2,3,4]))
# T2 = T.Tensor(([0,3,4,1],[0,3,4,1]))
# T3 = T.Tensor([4,3,2,5])

# tensor = T.cat((T1,T2),-2)
# print(np.shape(tensor))
# print(tensor)
# exit()

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
        self.hid1 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
          in_features=3, out_features=20)  # hidden 1
        self.hid2 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
              in_features=20, out_features=20)  # hidden 2
        self.oupt = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
          in_features=20, out_features=6)  # output

    # Missing initialization of weights
    #T.nn.init.uniform_(self.hid1.weight, -0.05, +0.05)

    def forward(self, x):
        z = T.tanh(self.hid1(x)) # try also relu activ. f.
        z = T.tanh(self.hid2(z))
        #z = T.tanh(self.hid3(z)) 
        #z = T.tanh(self.hid4(z))
        z = self.oupt(z)  # no activation
        z = T.cat(( z[:,(0,1,2)], T.abs(z[:,(3,4,5)]) ),-1) #T.nn.Softplus() 
        # tensor = T.cat((densities, variances),-1)
        return z 


#------------------------------------------------------------------------------------

src_file = 'C:\\Users\\clock\\Desktop\\Python\\datapointsk1k2k3_3k.txt' #datapointsk1k2k3_500
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
max_epochs = 4000
ep_log_interval = 10
lrn_rate = 0.01
kl_weight= 0.
vl_weight = 1.0 #0.3 funcou
mse_weight = 0
load_model = False

mse_loss = T.nn.MSELoss()
kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False) # regularization
optimizer = T.optim.Adam(net.parameters(), lr=lrn_rate  , weight_decay=1e-4)


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

    densities = output[:,(0,1,2)]
    variances = output[:,(3,4,5)]

    s = T.log(variances+1) # +1 looks necessary to prevent negative log; also abs
    
    #print(densities)
    #exit()
    mse_loss = (densities - target)**2 
    # print(T.mean(mse_loss))
    # print(s)
    loss = T.exp(-s)* mse_loss
    # print("\nmse_loss: ", T.mean(mse_loss))
    # print("\nlog(var): ", T.exp(-s))
    #print("\nloss: ", T.mean(s))

    return T.mean(loss) + T.mean(s)

#--------------------------------------------------------------------------------------------


def save_checkpoint(state, filename= "checkpoint.pth.tar"):
  print("=> Saving checkpoint")
  T.save(state, filename)

def load_checkpoint(checkpoint):
  print("=> Loading checkpoint")
  net.load_state_dict(checkpoint['state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer'])

#--------------------------------------------------------------------------------------------

epoch_list = []
epoch_loss_list = []
test_loss_list = []

if(load_model):
  load_checkpoint(T.load("checkpoint.pth.tar"))

print("Start training\n")
for epoch in range(0, max_epochs):

    epoch_loss = 0  # for one full epoch

    if (epoch == max_epochs-1):
      checkpoint = {'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()}
      save_checkpoint(checkpoint, ) # note: it is saving in the documents

    net.train()  # set mode

    for (batch_idx, batch) in enumerate(train_loader):
      (X_batch, Y_batch) = batch           # (predictors, targets)
      optimizer.zero_grad()                # prepare gradients
      oupt = net(X_batch)                  # predicted prices
      #msel = mse_loss(oupt, Y_batch)       # avg per item in batch
      #print(msel)
      #print(my_loss(oupt, Y_batch))
      kll = kl_loss(net)
      loss_val = my_loss(oupt, Y_batch) + kl_weight * kll
      epoch_loss += loss_val.item()        # accumulate avgs
      loss_val.backward()                  # compute gradients
      # T.nn.utils.clip_grad_value_(net.parameters(), 1e-5) # gradient cliping by value
      optimizer.step()                     # update wts
      n_batches = batch_idx+1

    #-------------------------------------------------------------
    # Print and save loss and errors
    if epoch % ep_log_interval == 0:
      epoch_list.append(epoch)
      epoch_loss_list.append(epoch_loss/n_batches)

      net.eval()
      prediction = net(x_test)
      loss_val = my_loss(prediction, y_test) +kl_weight * kll # confirm that this is already computed previously
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
for idx in range(len(train_predictions[0])-3):
    filename = 'C:\\Users\\clock\\Desktop\\Python\\Images\\Bayesian\\ks\\k' + str(idx+1)+'.png'
    plt.clf()
    plt.plot(np.arange(0,len(train_predictions),1), np.sort(train_predictions[:,idx]), 'ro',markersize=4. , label='predicted 1')
    plt.plot(np.arange(0,len(train_predictions),1), np.sort(train_predictions2[:,idx]), 'go', markersize=4. , label='predicted 2')
    plt.plot(np.arange(0,len(train_predictions),1), np.sort(train_predictions3[:,idx]), 'yo', markersize=4. , label='predicted 3')
    plt.plot(np.arange(0,len(train_predictions),1), np.sort(y_train[:,idx]), 'bo', markersize=4. , label= 'target')
    plt.legend()
    plt.title('k'+str(idx+1))
    plt.savefig(filename)




# plot k1 validation
plt.clf()
# -----------Sort the 2d array to plot (y, y_err)------------
y = np.stack((test_predictions[:,0], np.abs(test_predictions[:,3])), axis=1) # (predictions k1, prediction_variances for k1)
sorted_y = y[np.argsort(y[:, 0])]
#--------------
plt.errorbar(np.arange(0,len(test_predictions),1), sorted_y[:,0], yerr= sorted_y[:,1], fmt="ro", label='predicted')
plt.plot(np.arange(0,len(test_predictions),1),np.sort(y_test[:,0]), 'bo', label= 'target')
plt.legend()
plt.savefig('C:\\Users\\clock\\Desktop\\Python\\Images\\Bayesian\\k1_validation.png')


# plot k2 validation
plt.clf()
# -----------Sort the 2d array to plot (y, y_err)------------
y = np.stack((test_predictions[:,1], np.abs(test_predictions[:,4])), axis=1) # (predictions k1, prediction_variances for k1)
sorted_y = y[np.argsort(y[:, 0])]
#--------------
plt.errorbar(np.arange(0,len(test_predictions),1), sorted_y[:,0], yerr= sorted_y[:,1], fmt="ro", label='predicted')
plt.plot(np.arange(0,len(test_predictions),1),np.sort(y_test[:,1]), 'bo', label= 'target')
plt.legend()
plt.savefig('C:\\Users\\clock\\Desktop\\Python\\Images\\Bayesian\\k2_validation.png')


# plot k3 validation
plt.clf()
# -----------Sort the 2d array to plot (y, y_err)------------
y = np.stack((test_predictions[:,2], np.abs(test_predictions[:,5])), axis=1) # (predictions k1, prediction_variances for k1)
sorted_y = y[np.argsort(y[:, 0])]
#--------------
plt.errorbar(np.arange(0,len(test_predictions),1), sorted_y[:,0], yerr= sorted_y[:,1], fmt="ro", label='predicted')
plt.plot(np.arange(0,len(test_predictions),1),np.sort(y_test[:,2]), 'bo', label= 'target')
plt.legend()
plt.savefig('C:\\Users\\clock\\Desktop\\Python\\Images\\Bayesian\\k3_validation.png')




# Plot loss curves
plt.clf()
plt.plot(epoch_list, epoch_loss_list, '-o', label = 'train')
plt.plot(epoch_list, test_loss_list, '-o', label = 'validation')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.savefig('C:\\Users\\clock\\Desktop\\Python\\Images\\Bayesian\\loss_curve.png')


# # Print mean rel. error
# print("\n\nPrinting mean rel. err in validation:")
# for idx, rel_err in enumerate(mn.rel_error(train_predictions, y_train)):
#   print("k%d:   rel.err = %0.2f   " % \
#       (idx+1, rel_err*100))

# print("\n\nPrinting mean rel. err in training:")
# for idx, rel_err in enumerate(mn.rel_error(test_predictions, y_test)):
#   print("k%d:   rel.err = %0.2f   " % \
#       (idx+1, rel_err*100))