import matplotlib.pyplot as plt
import numpy as np
import torch as T
import torch.nn as nn
device = T.device("cpu")


with open('C:\\Users\\clock\\Desktop\\Python\\datapoints_test.txt', 'r') as file :
    filedata = file.readlines()


dataset=[]
for line in filedata:
    dataset.append([float(item) for item in line.split()])


X = [item[-3:] for item in dataset] # densities
Y = [item[:9] for item in dataset]  # coeficients k's



#--------------------------------------------------------------------
# Normalize inputs;  MinMax
X = np.array(X)
max0 = np.max(X[:,0])
max1 = np.max(X[:,1])
max2 = np.max(X[:,2])

min0 = np.min(X[:,0])
min1 = np.min(X[:,1])
min2 = np.min(X[:,2])

def MinMax_transform(x, min, max):
  return (x-min)/(max-min)
  #return x/max

def inverse_transform(x_scaled, min, max):
  return x_scaled*(max-min)+ min

v0 = MinMax_transform(X[:,0],min0, max0)
v1 = MinMax_transform(X[:,1],min1, max1)
v2 = MinMax_transform(X[:,2],min2, max2)

X = np.vstack((v0, v1, v2)).T


#Nomalize outputs k's
Y = np.array(Y)

def normalize_outputs(Y_array):
  def Max_transform(array):
    return array/np.max(array)

  list=[]
  for _ in Y_array.T:
    list.append(Max_transform(_))
  return np.array(list).T

Y = normalize_outputs(Y)

#---------------------------------------------------------------------



class HouseDataset(T.utils.data.Dataset):

  def __init__(self, m_rows=None):
    all_xy = dataset
    tmp_x = X
    tmp_y = Y

    self.x_data = T.tensor(tmp_x, \
      dtype=T.float32).to(device)
    self.y_data = T.tensor(tmp_y, \
      dtype=T.float32).to(device)

    #Normalize data
    #self.x_data = nn.functional.normalize(self.x_data, p= 2)
    #self.y_data = nn.functional.normalize(self.y_data, p= 1)


  def __len__(self):
    return len(self.x_data)

  def __getitem__(self, idx):
    coeficients = self.x_data[idx,:]  # or just [idx]
    densities = self.y_data[idx,:] 
    return (coeficients, densities)   


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # The Linear() class defines a fully connected network layer
        self.hid1 = nn.Linear(3, 50)  # hidden 1
        self.hid2 = nn.Linear(50, 50) # hidden 2
        self.hid3 = nn.Linear(50, 50) # hidden 3
        self.hid4 = nn.Linear(50, 50) # hidden 4
        #self.hid5 = nn.Linear(50, 50) # hidden 5
        #self.hid6 = nn.Linear(50, 50) # hidden 6
        #self.hid7 = nn.Linear(50, 50) # hidden 7
        self.oupt = nn.Linear(50, 9)  # output

    # Missing initialization of weights
    #T.nn.init.uniform_(self.hid1.weight, -0.05, +0.05)

    def forward(self, x):
        z = T.tanh(self.hid1(x)) # try also tanh activ. f.
        z = T.tanh(self.hid2(z))
        z = T.tanh(self.hid3(z))
        z = T.tanh(self.hid4(z))
        #z = T.tanh(self.hid5(z))
        #z = T.tanh(self.hid6(z))
        #z = T.sigmoid(self.hid7(z))
        z = self.oupt(z)  # no activation
        return z
    

#--------------------------------------------------------------------


T.manual_seed(1) # initialization of weights

# 2. create network
net = Net().to(device)

# 3. train model
max_epochs = 5000
ep_log_interval = 20
lrn_rate = 0.0001

loss_func = T.nn.MSELoss()
#optimizer = T.optim.SGD(net.parameters(), lr=lrn_rate)
optimizer = T.optim.Adam(net.parameters(), lr=lrn_rate)


# Split into training and test sets
full_dataset = HouseDataset()
train_size = int(0.9 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = T.utils.data.random_split(full_dataset, [train_size, test_size])
#train_dataset, test_dataset = T.utils.data.split(full_dataset, [train_size, test_size])

# Create minibatch on training set
bat_size= 200
train_ldr = T.utils.data.DataLoader(train_dataset,
    batch_size=bat_size, shuffle=True)


#-------------------------------------------------------------

train_dataset = np.array(train_dataset)
test_dataset = np.array(test_dataset)

x_train = [item.numpy() for item in train_dataset[:,0]]
x_train = np.array(x_train)
y_train = [item.numpy() for item in train_dataset[:,1]]
y_train = np.array(y_train)

x_test = [item.numpy() for item in test_dataset[:,0]]
x_test = np.array(x_test)
y_test = [item.numpy() for item in test_dataset[:,1]]
y_test = np.array(y_test)

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
print("\nStarting training with saved checkpoints")


epoch_list = []
epoch_loss_list = []
test_loss_list = []
# What is this for ? <---------------------
for epoch in range(0, max_epochs):
    T.manual_seed(1)  # recovery reproducibility
    epoch_loss = 0  # for one full epoch

    net.train()  # set mode

    for (batch_idx, batch) in enumerate(train_ldr):
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

      net.eval()
      prediction = net(T.tensor(x_test))
      loss_val = loss_func(prediction, T.tensor(y_test))
      test_loss_list.append(loss_val.item())

      print("epoch = %4d   loss = %0.4f   test_loss= %0.4f" % \
       (epoch, epoch_loss/n_batches, loss_val.item()))
    #--------------------------------------------------------------
 
print("Done \n")

net.eval() # set mode




# Plot
predicted = net(T.tensor(x_train)).detach()


plt.plot(np.arange(0,len(x_train),1), predicted[:,0], 'ro', label='predicted')
plt.plot(np.arange(0,len(x_train),1),y_train[:,0], 'bo', label= 'target')
plt.savefig('C:\\Users\\clock\\Desktop\\Python\\Images\\k0.png')

plt.clf()
plt.plot(epoch_list, epoch_loss_list, '-o', label = 'train')
plt.plot(epoch_list, test_loss_list, '-o', label = 'validation')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.savefig('C:\\Users\\clock\\Desktop\\Python\\Images\\loss_curve.png')


for idx, rel_err in enumerate(rel_error(predicted, y_train)):
  print("k%d:   rel. error = %0.2f   " % \
      (idx+1, rel_err*100))


