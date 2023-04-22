import metrics_normalization as mn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch as T
import torch.nn as nn
device = T.device("cpu")
from sklearn import preprocessing
# from mlxtend.preprocessing import standardize
# import math

scaler = preprocessing.MinMaxScaler()

# -----------------------------------------------------------
class LoadDataset(T.utils.data.Dataset):
  # last 3 columns: densities (input)
  # first 9 columns: k's  (output)

  def __init__(self, src_file, m_rows=None):
    all_xy = np.loadtxt(src_file, max_rows=m_rows,
      usecols=[0,1,2,3,4,5,6,7,8,9,10,11], delimiter="  ",
      # usecols=range(0,9), delimiter="\t",
      comments="#", skiprows=0, dtype=np.float64)

    tmp_x = all_xy[:,[0,1,2]] # [0,1,2]
    tmp_y = all_xy[:,[9,10,11]] 
    #[0,1,2,3,4,5,6,7,8]

    # self.my_standardize = mn.Standardize()
    # tmp_y = self.my_standardize.standardization(tmp_y)



    # Normalize data
    #scale k's
    scaler.fit(tmp_x) # standard scaler
    tmp_x = scaler.transform(tmp_x)

    tmp_y = mn.densitie_fraction(tmp_y)



    #tmp_x = standardize(tmp_x)
    #tmp_y = np.log(tmp_y)
    #tmp_y = standardize(tmp_y)

    self.x_data = T.tensor(tmp_x, \
      dtype=T.float64).to(device)
    self.y_data = T.tensor(tmp_y, \
      dtype=T.float64).to(device)
    self.all_data = T.tensor(all_xy, \
      dtype=T.float64).to(device)


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
        self.hid1 = nn.Linear(3,10)  # hidden 1
        self.hid2 = nn.Linear(10, 10) # hidden 2
        # self.hid3 = nn.Linear(10, 10) # hidden 3
        self.oupt = nn.Linear(10, 3)  # output
        T.nn.init.xavier_uniform_(self.hid1.weight)
        T.nn.init.xavier_uniform_(self.hid2.weight)
        # T.nn.init.xavier_uniform(self.hid3.weight)

    # Missing initialization of weights
    #T.nn.init.uniform_(self.hid1.weight, -0.05, +0.05)
    # T.nn.init.xavier_uniform(Net.hid1.weight)


    def forward(self, x):
        z = T.relu(self.hid1(x)) # try also relu activ. f.
        z = T.relu(self.hid2(z))
        # z = T.tanh(self.hid3(z))
        z = self.oupt(z)  # no activation
        return z

# ------------------------------------------------------------------------------
class MyPlots():
    def __init__(self):
        self.epoch_list = []
        self.epoch_loss_list = []
        self.val_loss_list = []
    
    def configure(self):
        A = 5  # We want figures to be A6
        plt.figure(figsize=(46.82 * .5**(.5 * A), 33.11 * .5**(.5 * A)))

        matplotlib.rc('xtick', labelsize=15) 
        matplotlib.rc('ytick', labelsize=15)
        matplotlib.rcParams.update({'font.size': 20}) 

#----------------------------------------------------------------------------------------

def save_checkpoint(state, filename= "checkpoint_forward_k1k2k3_minmax.pth.tar"):
  print("=> Saving checkpoint")
  T.save(state, filename)


#------------------------------------------------------------------------------------
src_file = 'C:\\Users\\clock\\Desktop\\Python\\datapointsk1k2k3_3k.txt' #datapoints500_k1k2k3
full_dataset = LoadDataset(src_file) #,m_rows=500) 


T.manual_seed(10)  # recover reproducibility

# 2. create network
net = Net().to(device)
net.to(T.double) # set model to float64


# 3. train model
max_epochs = 100
ep_log_interval =10
lrn_rate = 0.005

loss_func = T.nn.MSELoss()
optimizer = T.optim.Adam(net.parameters(), lr=lrn_rate ) #, weight_decay=1e-4)


# Split into training and validation sets
train_size = int(0.90 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, val_dataset = T.utils.data.random_split(full_dataset, [train_size, test_size])

# Create minibatch on training set
bat_size= 20
train_loader = T.utils.data.DataLoader(train_dataset,
    batch_size=bat_size, shuffle=True) # set to True


# CHANGE THIS 
x_val = val_dataset[:][0]
y_val = val_dataset[:][1]

#--------------------------------------------------------------------------------------------

# Compute norm of network weights
def param_norm():
    total_norm = 0
    parameters = [p for p in net.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

#-------------------------------------------------------------------------------------

# Initialize data structures to store info
myplot = MyPlots() 

print("Start training\n")
for epoch in range(0, max_epochs):
    epoch_loss = 0  # for one full epoch
    k1_loss = 0
    k3_loss = 0

    if (epoch == max_epochs-1):
      checkpoint = {'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()}
      save_checkpoint(checkpoint, ) # note: it is saving in the documents

    net.train()  # set mode

    for (batch_idx, batch) in enumerate(train_loader):
      (X_batch, Y_batch) = batch           # (predictors, targets)
      optimizer.zero_grad()                # prepare gradients
      oupt = net(X_batch)                  # predicted rate coefficients
      loss_val = loss_func(oupt, Y_batch)  # avg per item in batch
      epoch_loss += loss_val.item()        # accumulate avgs
      loss_val.backward()                  # compute gradients
      optimizer.step()                     # update wts
      n_batches = batch_idx+1              # save number of batches

    #-------------------------------------------------------------
    # Print and save loss and errors
    if epoch % ep_log_interval == 0:
      myplot.epoch_list.append(epoch)
      myplot.epoch_loss_list.append(epoch_loss/n_batches)

      net.eval() # (?)
      prediction = net(x_val)
      loss_val = loss_func(prediction, y_val)
      myplot.val_loss_list.append(loss_val.item())

      print("epoch = %4d   loss = %0.4f   validation_loss= %0.4f,  param_norm = %0.4f" % \
       (epoch, epoch_loss/n_batches, loss_val.item(), param_norm()))
    #--------------------------------------------------------------
 
print("Training complete \n")


print("\ntotal parameters' norm: ", param_norm())

net.eval() # set mode


# Evaluation
train_predictions = net(train_dataset[:][0]).detach().numpy()
y_train = train_dataset[:][1].numpy()

val_predictions = net(x_val).detach().numpy()
y_val = y_val.numpy()

# Set matplotlib fig. size, etc...
myplot.configure()

species = ['O2(X)','O2(a)', 'O(3P)']
# Plot densities of training set
for idx in range(len(train_predictions[0])):
    filename = 'C:\\Users\\clock\\Desktop\\Python\\Images\\species\\' + species[idx]+'.png'
    plt.clf()
    a = y_train[:,idx] # target
    b = train_predictions[:,idx] # predicted
    ab = np.stack((a,b), axis=-1)
    sorted_ab = ab[ab[:,0].argsort()]
    # print(sorted_ab)
    plt.plot(np.arange(0,len(train_predictions),1), sorted_ab[:,1], 'ro', label='predicted')
    plt.plot(np.arange(0,len(train_predictions),1),sorted_ab[:,0], 'bo', label= 'target')
    plt.legend()
    plt.title(species[idx])
    plt.savefig(filename)


# # plot k0 validation
# plt.clf()
# plt.plot(np.arange(0,len(val_predictions),1), val_predictions[:,0], 'ro', label='predicted')
# plt.plot(np.arange(0,len(val_predictions),1),y_val[:,0], 'bo', label= 'target')
# plt.legend()
# plt.savefig('C:\\Users\\clock\\Desktop\\Python\\Images\\k0_validation.png')



# Plot loss curves
plt.clf()
plt.plot(myplot.epoch_list, myplot.epoch_loss_list, '-o', label = 'train')
plt.plot(myplot.epoch_list, myplot.val_loss_list, '-o', label = 'validation')

# plt.plot(epoch_list, k1_loss_list, '-o', label = 'k1 train loss')
# plt.plot(epoch_list, k3_loss_list, '-o', label = 'k3 train loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.savefig('C:\\Users\\clock\\Desktop\\Python\\Images\\loss_curve.png')


# # Print mean rel. error
# print("\n\nPrinting mean rel. err in validation:")
# for idx, rel_err in enumerate(mn.rel_error(train_predictions, y_train)):
#   print("k%d:   rel.err = %0.2f   " % \
#       (idx+1, rel_err*100))

# print("\n\nPrinting mean rel. err in training:")
# for idx, rel_err in enumerate(mn.rel_error(val_predictions, y_val)):
#   print("k%d:   rel.err = %0.2f   " % \
#       (idx+1, rel_err*100))


# #---------------------------------------------EVALUATION OF TEST SET------------------------------------------------------
#Evaluation on a 10 point test set:
test_file = 'C:\\Users\\clock\\Desktop\\Python\\datapointsk1k2k3.txt'
all_xy =  np.loadtxt(test_file,
      usecols=[0,1,2,3,4,5,6,7,8,9,10,11], delimiter="  ",
      # usecols=range(0,9), delimiter="\t",
      comments="#", skiprows=0, dtype=np.float64)

tmp_x = all_xy[:,[0,1,2]] # [0,1,2]
tmp_y = all_xy[:,[9,10,11]] 
#[0,1,2,3,4,5,6,7,8]

# Normalize data
tmp_y = mn.densitie_fraction(tmp_y)
#tmp_y = standardize(tmp_y)
tmp_x = scaler.transform(tmp_x)

x_data = T.tensor(tmp_x, \
      dtype=T.float64).to(device)
y_data = T.tensor(tmp_y, \
      dtype=T.float64).to(device)

#predict =  net(x_data).detach().numpy()*full_dataset.my_standardize.std + full_dataset.my_standardize.mean
predict =  net(x_data).detach().numpy()
# predict = scaler.inverse_transform(predict)
target = y_data.numpy()
densities = x_data.numpy()
#target = y_data.numpy()*full_dataset.my_standardize.std + full_dataset.my_standardize.mean

# # Sort the arrays
# # pred_k1_sorted = np.sort(predict[:,0])
# # target_k1_sorted = np.sort(target[:,0])
# # pred_k2_sorted = np.sort(predict[:,1])
# # target_k2_sorted = np.sort(target[:,1])
# # pred_k3_sorted = np.sort(predict[:,2])
# # target_k3_sorted = np.sort(target[:,2])

# print("k1: prediction , target || k2: prediction , target")
# for i in range(len(predict)):
#   # print("k1: (%.4e, %.4e) || k2: (%.4e, %.4e)" % (pred_k1_sorted[i], target_k1_sorted[i], pred_k2_sorted[i], target_k2_sorted[i]))
#   print("%.4e  %.4e    %.4e  %.4e    %.4e  %.4e" % (predict[i,0], target[i,0], predict[i,1], target[i,1], predict[i,2], target[i,2]))

# for i in range(len(predict)):
#   print("%.5f  %.5f  %.5f    %.4f" % (densities[i,0], densities[i,1], densities[i,2], densities[i,0]+ densities[i,1]+ densities[i,2]))




# Plot densities of test set
for idx in range(len(predict[0])):
    filename = 'C:\\Users\\clock\\Desktop\\Python\\Images\\species_test_' + species[idx]+'.png'
    plt.clf()
    # plt.xticks(np.arange(1, len(target[:,0]), 1))
    # plt.figure(figsize=(46.82 * .5**(.5 * A), 33.11 * .5**(.5 * A)))
    a = target[:,idx] # target
    b = predict[:,idx] # predicted
    ab = np.stack((a,b), axis=-1)
    sorted_ab = ab[ab[:,0].argsort()]
    # print(sorted_ab)
    plt.plot(np.arange(1,len(predict)+1,1), sorted_ab[:,1], 'ro', label='predicted')
    plt.plot(np.arange(1,len(predict)+1,1),sorted_ab[:,0], 'bo', label= 'target')
    plt.legend()
    plt.ylabel(species[idx])
    plt.savefig(filename)


# print(full_dataset.my_standardize.std , full_dataset.my_standardize.mean)
# print(predict)
# print(target)


#---------------------------SAVE THE PREDICTED K'S TO BE INSERTED IN THE SIMULATION AGAIN--------------------------------
# predict, densities (target)
# I want to test if with the predicted k's we obtain these same densities

# array = np.hstack((predict,densities*2.56e22))
# np.savetxt("C:\\Users\\clock\\Desktop\\Python\\predictions_test.txt", array, fmt= "%.4e")
# # print(predict)