import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch as T
import torch.nn as nn
device = T.device("cpu")
from sklearn import preprocessing
import os


# -----------------------------------------------------------
class LoadDataset(T.utils.data.Dataset):
    # last 3 columns: densities (input)
    # first 9 columns: k's  (output)
    # 10th column is the pressure

    def __init__(self, src_file, nspecies, react_idx = None, m_rows=None, columns= None):
        all_xy = np.loadtxt(src_file, max_rows=m_rows,
            usecols=columns, delimiter="  ",
            # usecols=range(0,9), delimiter="\t", delimter= any whitespace by default
            comments="#", skiprows=0, dtype=np.float64)

        self.scaler = preprocessing.MinMaxScaler()
        self.scaler_max_abs = preprocessing.MaxAbsScaler()  

        ncolumns = len(all_xy[0])
        x_columns = np.arange(ncolumns-nspecies-1,ncolumns,1)
        y_columns = react_idx
        if react_idx == None:
            y_columns = np.arange(0,ncolumns-nspecies,1)

        tmp_x = all_xy[:,x_columns] # pressure and densities
        # multiply the first column (pressure) by 10
        tmp_x[:,0] = tmp_x[:,0]*10
        tmp_y = all_xy[:,y_columns]*10 # k's 


        # Normalize data
        self.scaler_max_abs.fit(tmp_x) 
        tmp_x = self.scaler_max_abs.transform(tmp_x)

        self.scaler.fit(tmp_y) 
        tmp_y = self.scaler.transform(tmp_y)


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
        self.hid1 = nn.Linear(4,10)  # hidden 1
        self.hid2 = nn.Linear(10, 10)# hidden 2
        # self.hid3 = nn.Linear(100, 100) # hidden 3
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
        z = self.oupt(z)  # no activation
        # z =T.sigmoid(self.oupt(z))  # with activation
        z =T.sigmoid(z)  # with activation
        return z
    
# ------------------------------------------------------------------------------

class Net_forward(nn.Module):
    def __init__(self):
        super(Net_forward, self).__init__()
        self.hid1 = nn.Linear(4,300)  # hidden 1
        # self.hid2 = nn.Linear(10, 10) # hidden 2
        self.oupt = nn.Linear(300, 3)  # output
        T.nn.init.xavier_uniform_(self.hid1.weight)
        # T.nn.init.xavier_uniform_(self.hid2.weight)

    
    def weights_init(self,m):
      if isinstance(m, nn.Linear):
        T.nn.init.xavier_uniform(m.weight.data)

    def forward(self, x):
        z = T.relu(self.hid1(x)) 
        # z = T.relu(self.hid2(z))
        z = self.oupt(z)  # no activation
        return z


def load_checkpoint(checkpoint):
    # checkpoint = T.Load(file)
    print("=> Loading checkpoint")
    net_forward.load_state_dict(checkpoint['state_dict'])

def load_checkpoint2(checkpoint):
    # checkpoint = T.Load(file)
    print("=> Loading checkpoint")
    net.load_state_dict(checkpoint['state_dict'])

# ------------------------------------------------------------------------------

def save_checkpoint(state, filename= "checkpoint_forward_pressure.pth.tar"):
  print("=> Saving checkpoint")
  T.save(state, filename)

# ------------------------------------------------------------------------------

def surrogate_loss(output, input_target): 
    # concatenate the first column of the input_target (pressure) to the output
    output = T.cat((output, input_target[:,:1]), 1)
    loki = net_forward(output)
    loss = T.mean((loki - input_target[:,1:])**2)

    

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
        plt.savefig('Images\\changing_pressure\\loss_curve.png')
    
    def plot_predict_target(self, predict, target, sort_by_target= False):
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
        plt.legend()

#-------------------------------------------------------------------------------------

if __name__=='__main__':

    # T.manual_seed(8) # recover reproducibility

    # 1. Load training dataset 
    src_file = 'data\\datapoints_pressure_0.1to10.txt' 
    species = ['O2(X)','O2(a)', 'O(3P)']
    k_columns = [0,1,2] # Set to None to read all reactions/columns in the file
    training_dataset  = LoadDataset(src_file, nspecies= len(species), react_idx= k_columns) # load the dataset again to access the scaler

    # Load test dataset of fixed k's
    full_dataset = np.loadtxt('data\\datapoints_fixed_test.txt', max_rows=None,
        usecols=[0,1,2,3], delimiter="  ",
        # usecols=range(0,9), delimiter="\t", delimter= any whitespace by default
        comments="#", skiprows=0, dtype=np.float64)
    
    # multiply the first column (pressure) by 10
    full_dataset[:,0] = full_dataset[:,0]*10
    full_dataset = T.tensor(full_dataset, dtype=T.float64).to(device)

    # 2. Create and load neural network model
    net = Net().to(device)
    net.to(T.double) # set model to float64
    load_checkpoint2(T.load("checkpoint_pressure_model.pth.tar"))
    net.eval()

    # 3. Create and load surrogate NN
    net_forward = Net_forward().to(device)
    net_forward.to(T.double) # set model to float64
    load_checkpoint(T.load("checkpoint_forward_pressure.pth.tar"))
    net_forward.eval() # set mode

    # 4. Choose loss for metrics
    loss_func = surrogate_loss
    loss_mse = T.nn.MSELoss()
    
    # Initialize data structures to store info
    myplot = MyPlots() 
    
    # --------------------------------------EVALUATION OF Test SET--------------------------------------------
    # Read fixed k values from chem file
    chem_file = 'O2_simple_1.chem'
    cwd = os.getcwd() # get current working directory 

    with open(cwd + '\\simulFiles\\' + chem_file, 'r') as file :
        values = []
        for line in file:
            values.append(line.split()[-2])
    # create a numpy array with the k values of type float
    k = np.array(values)
    k = k.astype(float)

    # ---------------------------------------------------------------
    # scale the full_dataset
    scaled_full_dataset = T.tensor(training_dataset.scaler_max_abs.transform(full_dataset.numpy())).to(device)
    # print("full_dataset: ", scaled_full_dataset)
    test_predictions = net(scaled_full_dataset) # k's predicted by the NN
    # print("test_predictions: ", test_predictions)
    y_test = k[:3] # fixed k's from the chem file

    # inverse transform the predicted k's to be compared with the fixed values from the chem file
    # test_predictions = training_dataset.scaler.inverse_transform(test_predictions.detach().numpy())
    # transform the fixed k's to be compared with the predicted k's
    y_test = y_test*10
    y_test = training_dataset.scaler.transform(y_test.reshape(1,-1)).squeeze()
    
    # exit()
    # print(test_predictions)
  

    # Set matplotlib fig. size, etc...
    myplot.configure()

    # Plot loss curves
    # myplot.plot_loss_curves()

    # verify the max and min values of the densities
    densities_training = training_dataset.scaler_max_abs.inverse_transform( training_dataset.x_data.numpy())
    print("max densities training: ", np.max(densities_training, axis=0))
    print("min densities training: ", np.min(densities_training, axis=0))

    densities_test = full_dataset.numpy()
    print("max densities test: ", np.max(densities_test, axis=0))
    print("min densities test: ", np.min(densities_test, axis=0))
    # exit()

    # Plot k's of training set
    test_predictions_numpy = test_predictions.detach().numpy()
    for idx in range(len(test_predictions[0])):
        filename = 'Images\\changing_pressure\\fixedK\\k' + str(idx+1)+'.png'
        plt.clf()
        a = y_test[idx] # target
        b = test_predictions_numpy[:,idx] # predicted

        #plot histogram of k values
        # fix the histogram acale
        # Calculate mean and standard deviation
        mean = np.mean(b)
        std = np.std(b)

        # Set the range for the histogram
        range_min = mean - 3 * std
        range_max = mean + 3 * std

        plt.xlim(range_min, range_max)
        # plt.xlim(0, 1)
        plt.hist(b, bins= 20, density=True, edgecolor='black',label='predicted')
        # plot a line at the fixed k value
        plt.axvline(x=a, color='r', linestyle='dashed', linewidth=2, label= 'fixed k')
        plt.title('k'+str(k_columns[idx]+1))
        plt.savefig(filename)

    # concatenate input pressure, train_dataset[:][0] first column, with net(train_dataset[:][0]) 
    surrog_input = T.cat((test_predictions, scaled_full_dataset[:,0].unsqueeze(1)), 1)
    predict_densities = net_forward(surrog_input).detach().numpy()
    target_densities = scaled_full_dataset.detach().numpy() # input densities
    # remove the first column (pressure) from the input densities
    target_densities = np.delete(target_densities, 0, axis=1)


    # Create a scatter plot of the two arrays against each other
    for idx in range(len(predict_densities[0])):
        filename = 'Images\\changing_pressure\\fixedK\\densities_correlations_' + str(idx) +'.png'
        plt.clf()
        a = target_densities[:,idx]
        b = predict_densities[:,idx]
        plt.scatter(a, b)

        rel_err = np.abs(np.subtract(a,b)/a)
        # print(rel_err)
        # print("stats: ",stats.chisquare(f_obs= b, f_exp= a))

        textstr = '\n'.join((
        r'$Mean\ \epsilon_{rel}=%.2f$%%' % (rel_err.mean()*100, ),
        r'$Max\ \epsilon_{rel}=%.2f$%%' % (max(rel_err)*100, )))

        # colour point o max error
        max_index = np.argmax(rel_err)
        plt.scatter(a[max_index],b[max_index] , color="gold", zorder= 2)
        
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', alpha=0.5) #, facecolor='none', edgecolor='none')

        # place a text box in upper left in axes coords
        plt.text(0.70, 0.25, textstr, fontsize=14,  transform=plt.gca().transAxes,
            verticalalignment='top', bbox=props)

        # Add labels and a title
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title(species[idx])
        # Add a diagonal line representing perfect agreement
        plt.plot([0, 1], [0, 1], linestyle='--', color='k')
        plt.savefig(filename)

    exit()

    # Plot densities using LoKI surrogate (forward) --------------------------------------------------------
    for idx in range(len(predict[0])):
        filename = 'Images\\changing_pressure\\ks\\training_forward_' + species[idx]+'.png'
        plt.clf()
        a = target[:,idx+1] # target
        b = predict[:,idx] # predicted
        myplot.plot_predict_target(b, a, sort_by_target=True)
        plt.ylabel(species[idx])
        plt.savefig(filename)



    #---------------------------------------------EVALUATION OF TEST SET------------------------------------------------------

    test_file = 'data\\datapoints_pressure_test.txt'
    all_xy =  np.loadtxt(test_file,
        usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12], delimiter="  ",
        # usecols=range(0,9), delimiter="\t",
        comments="#", skiprows=0, dtype=np.float64)

    tmp_x = all_xy[:,[9,10,11,12]]  # Change this manually
    tmp_y = all_xy[:,k_columns] 

    # Normalize data
    tmp_x = full_dataset.scaler_max_abs.transform(tmp_x)
    tmp_y = full_dataset.scaler.transform(tmp_y)

    x_data = T.tensor(tmp_x, \
        dtype=T.float64).to(device)
    y_data = T.tensor(tmp_y, \
        dtype=T.float64).to(device)

    predict =  net(x_data).detach().numpy()
    target = y_data.numpy()
    densities = x_data.numpy()

    np.savetxt("seed_1.txt", predict)
    # Plot k's of test set
    for idx in range(len(predict[0])):
        filename = 'D:\\Marcelo\\github\\Thesis\\Images\\k_test_' + str(idx+1)+'.png'
        plt.clf()
        a = target[:,idx] # target
        b = predict[:,idx] # predicted

        myplot.plot_predict_target(b, a, sort_by_target=True)
        plt.ylabel('k'+str(k_columns[idx]+1))
        plt.savefig(filename)

    # Plot densities using LoKI surrogate (forward) in test set --------------------------------------------------------
    # my_x = net(x_data)
    # predict_fwd =  net_forward(my_x).detach().numpy()
    # target = x_data.numpy()
    # for idx in range(len(predict_fwd[0])):
    #     filename = 'D:\\Marcelo\\github\\Thesis\\Images\\ks\\forward_' + species[idx]+'.png'
    #     plt.clf()
    #     a = target[:,idx] # target
    #     b = predict_fwd[:,idx] # predicted
    #     myplot.plot_predict_target(b, a, sort_by_target=True)
    #     plt.ylabel(species[idx])
    #     plt.savefig(filename)

    # Create a scatter plot of the two densitie arrays against each other
    predict_fwd = net_forward(T.cat((net(x_data), x_data[:,0].unsqueeze(1)), 1)).detach().numpy() # using forward model on net output
    for idx in range(1,len(densities[0]),1):
        filename = 'Images\\changing_pressure\\correlations_test' + str(idx+1)+'.png'
        plt.clf()
        a = densities[:,idx]
        b = predict_fwd[:,idx-1]
        plt.scatter(a, b)

        rel_err = np.abs(np.subtract(a,b)/a)
        # print(rel_err)
        # print("stats: ",stats.chisquare(f_obs= b, f_exp= a))

        textstr = '\n'.join((
        r'$Mean\ \epsilon_{rel}=%.2f$%%' % (rel_err.mean()*100, ),
        r'$Max\ \epsilon_{rel}=%.2f$%%' % (max(rel_err)*100, )))

        # colour point o max error
        max_index = np.argmax(rel_err)
        plt.scatter(a[max_index],b[max_index] , color="gold", zorder= 2)

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', alpha=0.5) #, facecolor='none', edgecolor='none')

        # place a text box in upper left in axes coords
        plt.text(0.70, 0.25, textstr, fontsize=14,  transform=plt.gca().transAxes,
            verticalalignment='top', bbox=props)


        # Add labels and a title
        plt.xlabel('True densities')
        plt.ylabel('Predicted densities')
        plt.title(species[idx-1])
        # Add a diagonal line representing perfect agreement
        plt.plot([0, 1], [0, 1], linestyle='--', color='k')
        plt.savefig(filename)


    # #---------------------------SAVE THE PREDICTED k'S TO BE INSERTED IN THE SIMULATION AGAIN--------------------------------
    # # I want to test if with the predicted k's we obtain these same densities
    # array = np.hstack((full_dataset.scaler.inverse_transform(predict),full_dataset.scaler_max_abs.inverse_transform(densities)))
    # np.savetxt("C:\\Users\\clock\\Desktop\\Python\\predictions_test.txt", array, fmt= "%.4e")