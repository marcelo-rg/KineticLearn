import numpy as np
import matplotlib.pyplot as plt
import torch as T
import torch.nn as nn
from sklearn.model_selection import KFold
device = T.device("cpu")
from mlxtend.preprocessing import standardize

#-----------------------------------------------------------------
def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()


#------------------------------------------------------------------

def rel_error(predictions_tensor, targets_array):
 
  predictions = np.array(predictions_tensor).T
  targets = targets_array.T
  n = len(predictions[0]) # number of datapoints to avg
  error = []
  
  for i in range(len(predictions)):
    soma = 0
    for j in range(n):
      if(targets[i][j]!=0):
        soma += np.abs(predictions[i][j]-targets[i][j])/targets[i][j]
    error.append(soma/n)
  return error


#-------------------------------------------------------------------

def densitie_fraction(Y_array):
  list=[]
  for _ in Y_array.T:
    list.append(_/2.56e22)
  return np.array(list).T

def standarization(Y_array):
  def std_transform(array):
    return (array-np.mean(array))/np.std(array)

  list=[]
  for _ in Y_array.T:
    list.append(std_transform(_))
  return np.array(list).T


# -----------------------------------------------------------
class LoadDataset(T.utils.data.Dataset):
  # last 3 columns: densities (input)
  # first 9 columns: k's  (output)

  def __init__(self, src_file, m_rows=None):
    all_xy = np.loadtxt(src_file, max_rows=m_rows,
      usecols=[0,1,2,3,4,5,6,7,8,9,10,11], delimiter="  ",
      # usecols=range(0,9), delimiter="\t",
      comments="#", skiprows=0, dtype=np.float32)

    tmp_x = all_xy[:,[9,10,11]] 
    tmp_y = all_xy[:,[0,1,2,3,4,5,6,7,8]]

    # Normalize data
    tmp_x = densitie_fraction(tmp_x)
    #tmp_y= np.log10(tmp_y)
    tmp_y = standardize(tmp_y, columns=[0,1,2,3,4,5,6,7,8])


    self.x_data = T.tensor(tmp_x, \
      dtype=T.float32).to(device)
    self.y_data = T.tensor(tmp_y, \
      dtype=T.float32).to(device)

  def __len__(self):
    return len(self.x_data)

  def __getitem__(self, idx):
    densities = self.x_data[idx,:]  # or just [idx]
    coef = self.y_data[idx,:] 
    return (densities, coef)       # tuple of two matrices 

# ------------------------------------------------------------------------------

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # The Linear() class defines a fully connected network layer
        self.hid1 = nn.Linear(3, 50)  # hidden 1
        self.hid2 = nn.Linear(50, 50) # hidden 2
        self.hid3 = nn.Linear(50, 50) # hidden 3
        self.hid4 = nn.Linear(50, 50) # hidden 4
        self.oupt = nn.Linear(50, 9)  # output

    # Missing initialization of weights
    #T.nn.init.uniform_(self.hid1.weight, -0.05, +0.05)

    def forward(self, x):
        z = T.tanh(self.hid1(x)) # try also tanh activ. f.
        z = T.tanh(self.hid2(z))
        z = T.tanh(self.hid3(z))
        z = T.tanh(self.hid4(z))
        z = self.oupt(z)  # no activation
        return z


#------------------------------------------------------------------------------------

src_file = 'C:\\Users\\clock\\Desktop\\Python\\datapoints_log_1k.txt'
full_dataset = LoadDataset(src_file)

T.manual_seed(1) # initialization of weights

# 2. create network
net = Net().to(device)

# 3. train model
max_epochs = 2000
ep_log_interval = 20
lrn_rate = 0.001

loss_func = T.nn.MSELoss()
#optimizer = T.optim.SGD(net.parameters(), lr=lrn_rate)
optimizer = T.optim.Adam(net.parameters(), lr=lrn_rate)


"""# Split into training and test sets
train_size = int(0.9 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = T.utils.data.random_split(full_dataset, [train_size, test_size])"""

# Define the K-fold Cross Validator
k_folds= 100
kfold = KFold(n_splits=k_folds, shuffle=True)

"""
# Create minibatch on training set
bat_size= 1000
train_loader = T.utils.data.DataLoader(train_dataset,
    batch_size=bat_size, shuffle=True) # set to True


# CHANGE THIS 
x_test = test_dataset[:][0]
y_test = test_dataset[:][1]
"""

#--------------------------------------------------------------------------------------------

# K-fold Cross Validation model evaluation
for fold, (train_ids, test_ids) in enumerate(kfold.split(full_dataset)):
  
  # Print
  net.apply(reset_weights)
  print(f'FOLD {fold}')
  print('--------------------------------')
  
  bat_size = 1000

  # Sample elements randomly from a given list of ids, no replacement.
  train_subsampler = T.utils.data.SubsetRandomSampler(train_ids)
  test_subsampler = T.utils.data.SubsetRandomSampler(test_ids)

  train_loader = T.utils.data.DataLoader(full_dataset,
    batch_size=bat_size, sampler=train_subsampler) 

  test_loader = T.utils.data.DataLoader(full_dataset,
    batch_size=bat_size, sampler=test_subsampler) 


  epoch_list = []
  epoch_loss_list = []
  test_loss_list = []
  print("Start training\n")
  for epoch in range(0, max_epochs):
    T.manual_seed(1)  # recovery reproducibility
    epoch_loss = 0  # for one full epoch

    net.train()  # set mode

    for (batch_idx, batch) in enumerate(train_loader):
      (X_batch, Y_batch) = batch           # (predictors, targets)
      optimizer.zero_grad()                # prepare gradients
      oupt = net(X_batch)                  # predicted prices
      loss_val = loss_func(oupt, Y_batch)  # avg per item in batch
      epoch_loss += loss_val.item()        # accumulate avgs
      loss_val.backward()                  # compute gradients
      optimizer.step()                     # update wts
      n_batches = batch_idx+1

    #-------------------------------------------------------------
    # Print and save loss and errors
    if epoch % ep_log_interval == 0:
      epoch_list.append(epoch)
      epoch_loss_list.append(epoch_loss/n_batches)
     
      # Evaluation for this epoch
      net.eval() # set mode
      with T.no_grad():
        # Iterate over the test data and generate predictions (test_loader may have batches)
        for (i, test_data) in enumerate(test_loader):
          (inputs, targets) = test_data
          predictions =  net(inputs)
          #print(len(predictions))
          #exit()
          loss_val = loss_func(predictions, targets)
          test_loss_list.append(loss_val.item())
      
      print("epoch = %4d   loss = %0.4f   validation_loss= %0.4f" % \
      (epoch, epoch_loss/n_batches, loss_val.item()))
    #--------------------------------------------------------------
  

    

  #Plot loss curves
  plt.clf()
  plt.plot(epoch_list, epoch_loss_list, '-o', label = 'train')

  plt.plot(epoch_list, test_loss_list, '-o', label = 'validation')
  plt.xlabel('epoch')
  plt.ylabel('loss')
  plt.legend()
  plt.savefig('C:\\Users\\clock\\Desktop\\Python\\Images\\cross_val\\loss_curve_fold'+ str(fold)+ '.png')
 
print("Done \n")

net.eval() # set mode

"""

# Evaluation
train_predictions = net(train_dataset[:][0]).detach().numpy()
y_train = train_dataset[:][1].numpy()

test_predictions = net(x_test).detach().numpy()
y_test = y_test.numpy()


# Plot ks of training set
for idx in range(len(train_predictions[0])):
  filename = 'C:\\Users\\clock\\Desktop\\Python\\Images\\ks\\k' + str(idx+1)+'.png'
  plt.clf()
  plt.plot(np.arange(0,len(train_predictions),1), train_predictions[:,idx], 'ro', label='predicted')
  plt.plot(np.arange(0,len(train_predictions),1),y_train[:,idx], 'bo', label= 'target')
  plt.legend()
  plt.title('k'+str(idx+1))
  plt.savefig(filename)


# Plot loss curves
plt.clf()
plt.plot(epoch_list, epoch_loss_list,  label = 'train')
plt.plot(epoch_list, test_loss_list,  label = 'validation')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.savefig('C:\\Users\\clock\\Desktop\\Python\\Images\\loss_curve.png')


# Print mean rel. error
print("\n\nPrinting mean rel. err in validation:")
for idx, rel_err in enumerate(rel_error(train_predictions, y_train)):
  print("k%d:   rel.err = %0.2f   " % \
      (idx+1, rel_err*100))

print("\n\nPrinting mean rel. err in training:")
for idx, rel_err in enumerate(rel_error(test_predictions, y_test)):
  print("k%d:   rel.err = %0.2f   " % \
      (idx+1, rel_err*100))

"""