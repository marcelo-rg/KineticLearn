import metrics_normalization as mn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch as T
import torch.nn as nn
import torchbnn as bnn
device = T.device("cpu")
from sklearn import preprocessing


# -----------------------------------------------------------
class LoadDataset(T.utils.data.Dataset):

    def __init__(self, src_file, nspecies, react_idx = None, m_rows=None, columns= None):
        all_xy = np.loadtxt(src_file, max_rows=m_rows,
            usecols=columns, delimiter="  ",
            # usecols=range(0,9), delimiter="\t", delimter= any whitespace by default
            comments="#", skiprows=0, dtype=np.float64)

        self.scaler = preprocessing.MinMaxScaler()
        self.scaler_max_abs = preprocessing.MaxAbsScaler()  

        ncolumns = len(all_xy[0])
        x_columns = np.arange(ncolumns-nspecies,ncolumns,1)
        y_columns = react_idx
        if react_idx == None:
            y_columns = np.arange(0,ncolumns-nspecies,1)

        tmp_x = all_xy[:,x_columns] 
        tmp_y = all_xy[:,y_columns] # [0,1,2]

        # Normalize data
        self.scaler.fit(tmp_y) 
        tmp_y = self.scaler.transform(tmp_y)

        self.scaler_max_abs.fit(tmp_x) 
        tmp_x = self.scaler_max_abs.transform(tmp_x)

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
        self.hid2 = nn.Linear(10, 10)# hidden 2
        # self.hid3 = nn.Linear(10, 10) # hidden 3
        self.oupt = nn.Linear(10, 3)  # output
        T.nn.init.xavier_uniform_(self.hid1.weight)
        T.nn.init.xavier_uniform_(self.hid2.weight)
        # T.nn.init.xavier_uniform(self.hid3.weight)
        # T.nn.init.xavier_uniform_(self.oupt.weight)


    def weights_init(self,m):
      if isinstance(m, nn.Linear):
        T.nn.init.xavier_uniform(m.weight.data)


    def forward(self, x):
        z = T.tanh(self.hid1(x)) # try also relu activ. f.
        z = T.tanh(self.hid2(z))
        # z = T.tanh(self.hid3(z))
        # z = self.oupt(z)  # no activation
        z =T.abs(self.oupt(z))  # with activation
        return z
    
# ------------------------------------------------------------------------------

class BayesianNet(nn.Module):
    def __init__(self):
        super(BayesianNet, self).__init__()
        self.hid1 = bnn.BayesLinear(prior_mu=0., prior_sigma=0.1,
          in_features=3, out_features=10)  # hidden 1
        self.hid2 = bnn.BayesLinear(prior_mu=0., prior_sigma=0.1,
             in_features=10, out_features=10)  # hidden 2
        self.oupt = bnn.BayesLinear(prior_mu=0., prior_sigma=0.1,
          in_features=10, out_features=3)  # output

    # Missing initialization of weights
    #T.nn.init.uniform_(self.hid1.weight, -0.05, +0.05)

    def forward(self, x):
        z = T.tanh(self.hid1(x)) # try also relu activ. f.
        z = T.tanh(self.hid2(z))
        #z = T.tanh(self.hid3(z))
        #z = T.tanh(self.hid4(z))
        # z = self.oupt(z)  # no activation
        z = T.abs(self.oupt(z))  # with activation
        return z

#------------------------------------------------------------------------------------

class Net_forward(nn.Module):
    def __init__(self):
        super(Net_forward, self).__init__()
        self.hid1 = nn.Linear(3,10)  # hidden 1
        self.hid2 = nn.Linear(10, 10) # hidden 2
        self.oupt = nn.Linear(10, 3)  # output
        T.nn.init.xavier_uniform_(self.hid1.weight)
        T.nn.init.xavier_uniform_(self.hid2.weight)

    
    def weights_init(self,m):
      if isinstance(m, nn.Linear):
        T.nn.init.xavier_uniform(m.weight.data)

    def forward(self, x):
        z = T.relu(self.hid1(x)) 
        z = T.relu(self.hid2(z))
        z = self.oupt(z)  # no activation
        return z


def load_checkpoint(checkpoint):
    # checkpoint = T.Load(file)
    print("=> Loading checkpoint")
    net_forward.load_state_dict(checkpoint['state_dict'])


def surrogate_loss(output, input_target): 
    loki = net_forward(output)
    loss = T.mean((loki - input_target)**2)

    return loss
# ------------------------------------------------------------------------------

class MyPlots():
    def __init__(self):
        self.epoch_list = []
        self.epoch_loss_list = []
        self.val_loss_list = []
        self.epoch_loss_list_loki = []
    
    def configure(self):
        A = 5  # We want figures to be A5
        plt.figure(figsize=(46.82 * .5**(.5 * A), 33.11 * .5**(.5 * A)))

        matplotlib.rc('xtick', labelsize=15) 
        matplotlib.rc('ytick', labelsize=15)
        matplotlib.rcParams.update({'font.size': 20}) 

    def plot_loss_curves(self):
        plt.plot(self.epoch_list, self.epoch_loss_list, '-o', label = 'MSE')
        plt.plot(self.epoch_list, self.epoch_loss_list_loki, '-o', label = 'LoKI')
        plt.plot(self.epoch_list, self.val_loss_list, '-o', label = 'validation')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig('C:\\Users\\clock\\Desktop\\Python\\Images\\Bayesian\\loss_curve.png')
    
    def plot_predict_target(self, predict, target, sort_by_target= False, y_err = None):
        npoints = len(predict)
        x_ = np.arange(0,npoints,1)
        a = target # target
        b = predict # predicted
        ab = np.stack((a,b), axis=-1)
        sorted_ab = ab[ab[:,0].argsort()]
        if (not sort_by_target):
            sorted_ab = ab
        plt.plot(x_, sorted_ab[:,1], 'ro', label='predicted')
        plt.plot(x_, sorted_ab[:,0], 'bo', label= 'target')

        if (y_err is not None):
            plt.clf()
            plt.errorbar(x_, sorted_ab[:,1],yerr= y_err, fmt='ro',ecolor="yellow", label='predicted')
            plt.plot(x_, sorted_ab[:,0], 'bo', label= 'target')
        plt.legend()



#----------------------------------------------------------------------------------------

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

if __name__=='__main__':

    # x_1 = T.randn(10, 3)
    # y_1 = T.randn(10, 3)
    # z_1 = T.stack((x_1, y_1))
    # mean = T.mean(z_1, dim = 0)
    # var = T.var(z_1,dim =0)
    # print(np.shape(var))
    # print(np.shape(mean))
    # exit()
    T.manual_seed(9) # recover reproducibility

    # 1. Load training dataset 
    src_file = 'C:\\Users\\clock\\Desktop\\Python\\datapointsk1k2k3_3k.txt' 
    species = ['O2(X)','O2(a)', 'O(3P)']
    k_columns = [0,1,2] # Set to None to read all reactions/columns in the file
    full_dataset = LoadDataset(src_file, nspecies= len(species), react_idx= k_columns) #(data already scaled)

    # 2. Create neural network
    net = BayesianNet().to(device)
    net.to(T.double) # set model to float64

    # 3. Build training Model
    max_epochs = 1000
    ep_log_interval =20
    lrn_rate = 0.01

    # Create and load surrogate NN
    net_forward = Net_forward().to(device)
    net_forward.to(T.double) # set model to float64
    load_checkpoint(T.load("checkpoint_forward_k1k2k3_minmax.pth.tar"))
    net_forward.eval() # set mode

    # 4. Choose loss and optimizer
    loss_func = surrogate_loss
    kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False) #normalization
    loss_mse = T.nn.MSELoss()
    optimizer = T.optim.Adam(net.parameters(), lr=lrn_rate) # , weight_decay=1e-4)

    # Split into training and validation sets
    train_size = int(0.90 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, val_dataset = T.utils.data.random_split(full_dataset, [train_size, test_size])

    # Create minibatch on training set
    bat_size= 1000
    train_loader = T.utils.data.DataLoader(train_dataset,
        batch_size=bat_size, shuffle=True) # set to True
    
    # Extract x and y of validation set
    x_val = val_dataset[:][0]
    y_val = val_dataset[:][1]
    
    # Initialize data structures to store info
    myplot = MyPlots() 

    # 5. Training algorithm
    print("Start training\n")
    for epoch in range(0, max_epochs):
        epoch_loss = 0  # for one full epoch

        net.train()  # set mode

        for (batch_idx, batch) in enumerate(train_loader):
            (X_batch, Y_batch) = batch           # (predictors, targets)
            optimizer.zero_grad()                # prepare gradients
            oupt = net(X_batch)                  # predicted rate coefficients
            loss_val_loki = 16*loss_func(oupt, X_batch)
            kll = kl_loss(net)  
            loss_val_mse = loss_mse(oupt, Y_batch)
            loss_val = loss_val_loki + 0.01*kll #+ loss_val_mse  # avg per item in batch
            epoch_loss += loss_val.item()        # accumulate avgs
            loss_val.backward()                  # compute gradients
            optimizer.step()                     # update wts
            n_batches = batch_idx+1              # save number of batches

        #-------------------------------------------------------------
        # Print and save loss and errors
        if (epoch % ep_log_interval) == 0:
            myplot.epoch_list.append(epoch)
            myplot.epoch_loss_list.append(loss_val_mse.item()/n_batches)
            myplot.epoch_loss_list_loki.append(loss_val_loki.item()/n_batches)

            net.eval() # (?)
            prediction = net(x_val)
            # loss_val = loss_func(prediction, y_val)
            loss_val = loss_func(prediction, x_val) #+ loss_func(prediction,y_val)
            myplot.val_loss_list.append(loss_val.item())

            print("epoch = %4d   loss = %0.4f   validation_loss= %0.4f,  param_norm = %0.4f" % \
            (epoch, epoch_loss/n_batches, loss_val.item(), param_norm()))
        #--------------------------------------------------------------

    print("Training complete \n")

    print("\ntotal parameters' norm: ", param_norm())


    # --------------------------------------EVALUATION OF TRAINING SET--------------------------------------------
    train_predictions = net(train_dataset[:][0]).detach().numpy()
    y_train = train_dataset[:][1].numpy()

    val_predictions = net(x_val).detach().numpy()
    y_val = y_val.numpy()

    # Set matplotlib fig. size, etc...
    myplot.configure()

    # Plot loss curves
    myplot.plot_loss_curves()

    # Plot k's of training set
    for idx in range(len(train_predictions[0])):
        filename = 'C:\\Users\\clock\\Desktop\\Python\\Images\\Bayesian\\ks\\k' + str(idx+1)+'.png'
        plt.clf()
        a = y_train[:,idx] # target
        b = train_predictions[:,idx] # predicted
        myplot.plot_predict_target(b, a, sort_by_target=True)
        plt.title('k'+str(k_columns[idx]+1))
        plt.savefig(filename)

    # Plot densities using LoKI surrogate (forward) --------------------------------------------------------
    predict =  net_forward(net(train_dataset[:][0]).detach()).detach().numpy()
    target = train_dataset[:][0] # input densities
    for idx in range(len(predict[0])):
        filename = 'C:\\Users\\clock\\Desktop\\Python\\Images\\Bayesian\\ks\\training_forward_' + species[idx]+'.png'
        plt.clf()
        a = target[:,idx] # target
        b = predict[:,idx] # predicted
        myplot.plot_predict_target(b, a, sort_by_target=True)
        plt.ylabel(species[idx])
        plt.savefig(filename)

    #---------------------------------------------EVALUATION OF TEST SET------------------------------------------------------

    test_file = 'C:\\Users\\clock\\Desktop\\Python\\datapointsk1k2k3.txt'
    all_xy =  np.loadtxt(test_file,
        usecols=[0,1,2,3,4,5,6,7,8,9,10,11], delimiter="  ",
        # usecols=range(0,9), delimiter="\t",
        comments="#", skiprows=0, dtype=np.float64)

    tmp_x = all_xy[:,[9,10,11]]  # Change this manually
    tmp_y = all_xy[:,k_columns] 

    # Normalize data
    tmp_x = full_dataset.scaler_max_abs.transform(tmp_x)
    tmp_y = full_dataset.scaler.transform(tmp_y)

    x_data = T.tensor(tmp_x, \
        dtype=T.float64).to(device)
    y_data = T.tensor(tmp_y, \
        dtype=T.float64).to(device)

    # Loop an ensemble of models and predictions
    nsamples = 1000
    for i in range(nsamples):
        predict = net(x_data)
        if i == 0:
            stack =  net(x_data)
            stack = T.stack((stack,predict))
        else:
            stack = T.cat((stack,T.reshape(predict,(1,100,3))), dim=0)

    # print(T.mul(soma,1/nsamples))
    predict = T.mean(stack, dim=0).detach().numpy()
    var = T.var(stack, dim=0).detach().numpy()
    print(var[0])
    target = y_data.numpy()
    densities = x_data.numpy()
    # np.savetxt("seed_1.txt", predict)
    # Plot k's of test set
    for idx in range(len(predict[0])):
        filename = 'C:\\Users\\clock\\Desktop\\Python\\Images\\Bayesian\\k_test_' + str(idx+1)+'.png'
        plt.clf()
        a = target[:,idx] # target
        b = predict[:,idx] # predicted
        err = np.sqrt(var[:,idx])
        myplot.plot_predict_target(b, a, sort_by_target=True, y_err = err)
        plt.ylabel('k'+str(k_columns[idx]+1))
        plt.savefig(filename)

    # Plot densities using LoKI surrogate (forward) in test set --------------------------------------------------------
    my_x = T.mean(stack, dim=0)
    predict_fwd =  net_forward(my_x).detach().numpy()
    target = x_data.numpy()
    for idx in range(len(predict_fwd[0])):
        filename = 'C:\\Users\\clock\\Desktop\\Python\\Images\\Bayesian\\ks\\forward_' + species[idx]+'.png'
        plt.clf()
        a = target[:,idx] # target
        b = predict_fwd[:,idx] # predicted
        myplot.plot_predict_target(b, a, sort_by_target=True)
        plt.ylabel(species[idx])
        plt.savefig(filename)


    #---------------------------SAVE THE PREDICTED k'S TO BE INSERTED IN THE SIMULATION AGAIN--------------------------------
    # I want to test if with the predicted k's we obtain these same densities
    array = np.hstack((full_dataset.scaler.inverse_transform(predict),full_dataset.scaler_max_abs.inverse_transform(densities)))
    np.savetxt("C:\\Users\\clock\\Desktop\\Python\\predictions_test.txt", array, fmt= "%.4e")