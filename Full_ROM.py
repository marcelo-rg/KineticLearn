import metrics_normalization as mn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch as T
# import torch.nn as nn
device = T.device("cpu")
from sklearn import preprocessing


def divide_and_extract(arr):
    arr = np.array(arr)
    first_col = arr[:, 0]
    second_col = arr[:, 1]
    third_col = arr[:, 2]
    return np.column_stack((second_col / first_col, third_col))

def multipy(arr, col1, col2):
    arr = np.array(arr)
    first_col = arr[:, col1]
    second_col = arr[:, col2]
    return first_col*second_col

def divide(arr, col1, col2):
    arr = np.array(arr)
    first_col = arr[:, col1]
    second_col = arr[:, col2]
    return first_col/second_col

def add(arr, col1, col2):
    arr = np.array(arr)
    first_col = arr[:, col1]
    second_col = arr[:, col2]
    return first_col+second_col

# -----------------------------------------------------------

class Samples_Dataset(T.utils.data.Dataset):
    def __init__(self, features_file, targets_file):
        self.features = T.load(features_file).to(device) # rate coefficients
        self.targets = T.tensor(np.load(targets_file), dtype=T.float64
        ).to(device) # densities

        # choose number of wanted samples (max 500)
        nsamples = 15
        # get size of last dimension
        size = np.shape(self.features)[-1]
        self.features =  self.features[:nsamples,:,:].reshape(-1, size)

        tmp_coeff = self.features.detach().numpy()

        func1 = multipy(tmp_coeff, col1 = 0, col2 = 1)
        func2 = divide(tmp_coeff, col1 = 0, col2 = 1)
        func3 = add(tmp_coeff, col1 = 0, col2 = 1) 

        tmp_coeff = np.column_stack((func1,func2, func3, tmp_coeff[:,-1]))
        # Transform back to tensor 
        self.features = T.tensor(tmp_coeff, dtype=T.float64).to(device)

        # repeat the targets array nsamples times
        self.targets = T.tile(self.targets, (nsamples, 1)).to(device)

    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        x = self.features[index] # rate coeff
        y = self.targets[index]
        return x, y

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

        tmp_x = all_xy[:,x_columns] # species
        tmp_y = all_xy[:,y_columns] # rate coefficients

        # tmp_y = divide_and_extract(tmp_y) # k2/k1

        func1 = multipy(tmp_y, col1 = 0, col2 = 1)
        func2 = divide(tmp_y, col1 = 0, col2 = 1)
        func3 = add(tmp_y, col1 = 0, col2 = 1)

        stack = np.column_stack((func1,func2, func3, tmp_y[:,-1]))
        tmp_y = stack

        # Normalize data
        self.scaler.fit(tmp_y) 
        tmp_y = self.scaler.transform(tmp_y)

        self.scaler_max_abs.fit(tmp_x) 
        tmp_x = self.scaler_max_abs.transform(tmp_x)
        # print(tmp_y, np.shape(tmp_y))

        # Change this back again 
        self.x_data = T.tensor(tmp_y, \
            dtype=T.float64).to(device)
        self.y_data = T.tensor(tmp_x, \
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

class Full_ROM(T.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Full_ROM, self).__init__()

        # self.LinearRegression = T.nn.Sequential(
        #     T.nn.Linear(input_size, 1),
        #     # T.nn.Tanh()
        # )
        self.LinearRegression = T.nn.Sequential(T.nn.Linear(input_size, 1),T.nn.Tanh())
            
        self.decoder = T.nn.Sequential(
            T.nn.Linear(1, hidden_size),
            T.nn.Tanh(),
            T.nn.Linear(hidden_size, hidden_size),
            T.nn.Tanh(),
            T.nn.Linear(hidden_size, 3),
        )

    def forward(self, x):
        encoded = self.LinearRegression(x)
        decoded = self.decoder(encoded)
        return decoded

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
        plt.plot(self.epoch_list, self.epoch_loss_list, '-o', label = 'Loss MSE + L1')
        # plt.plot(self.epoch_list, self.epoch_loss_list_loki, '-o', label = 'LoKI')
        plt.plot(self.epoch_list, self.val_loss_list, '-o', label = 'validation')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig('C:\\Users\\clock\\Desktop\\Python\\Images\\Full_ROM_model\\loss_curve.png')
    
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



#----------------------------------------------------------------------------------------

# Compute norm of network weights
def param_norm():
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

#-------------------------------------------------------------------------------------

if __name__ == '__main__':
    T.manual_seed(8) # recover reproducibility

    # 1. Load training dataset 
    src_file = 'C:\\Users\\clock\\Desktop\\Python\\datapointsk1k2k3_3k.txt' 
    species = ['O2(X)', 'O2(a)', 'O(3P)']
    k_columns = [0,1,2] # Set to None to read all reactions/columns in the file
    full_dataset = LoadDataset(src_file, nspecies= len(species), react_idx= k_columns) #(data already scaled)
    nfunctions = 4

    dir_path = "C:\\Users\\clock\\Desktop\\Python\\Images\\statistics\\stacks\\500_samples\\"
    samples_dataset = Samples_Dataset(features_file=dir_path+"stack_train.pt",
                                       targets_file=dir_path+"densities_targets_train.npy")

    # 2. Create neural network
    model = Full_ROM(input_size= nfunctions, hidden_size= 10).to(device)
    model.to(T.double) # set model to float64

    # 3. Build training Model
    max_epochs = 200
    ep_log_interval =10
    lrn_rate = 0.01
    l1_coeff = 0.01

    # 4. Choose loss and optimizer
    loss_func = T.nn.MSELoss()
    # loss_mse = T.nn.MSELoss()
    optimizer = T.optim.Adam(model.parameters(), lr=lrn_rate) # , weight_decay=1e-4)

    # Split into training and validation sets | samples_dataset -> full_dataset
    train_size = int(0.90 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, val_dataset = T.utils.data.random_split(full_dataset, [train_size, test_size])

    # Create minibatch on training set
    bat_size= 20
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

        model.train()  # set mode

        for (batch_idx, batch) in enumerate(train_loader):
            (X_batch, Y_batch) = batch           # (predictors, targets)
            optimizer.zero_grad()                # prepare gradients
            oupt = model(X_batch)                # predicted rate coefficient 
            # print(np.shape(X_batch))
            # exit()
            loss_val_mse = loss_func(oupt, Y_batch)
            # Add L1 regularization to the first input layer 
            loss_val = loss_val_mse #+ l1_loss
            l1_loss = T.tensor(0., requires_grad=True)
            for param in model.LinearRegression.parameters():
                l1_loss = l1_loss + T.norm(param, p=1) # p=1 is norm 1
            loss_val = loss_val_mse + l1_coeff * l1_loss
            epoch_loss += loss_val.item()        # accumulate avgs
            loss_val.backward()                  # compute gradients
            optimizer.step()                     # update wts
            n_batches = batch_idx+1              # save number of batches

        #-------------------------------------------------------------
        # Print and save loss and errors
        if (epoch % ep_log_interval) == 0:
            myplot.epoch_list.append(epoch)
            myplot.epoch_loss_list.append(epoch_loss/n_batches)
            # myplot.epoch_loss_list_loki.append(loss_val_loki.item()/n_batches)

            model.eval() # (?)
            prediction = model(x_val)
            # print(np.shape(x_val))
            # exit()
            # loss_val = loss_func(prediction, y_val)
            loss_val = loss_func(prediction, y_val) #+ l1 loss
            myplot.val_loss_list.append(loss_val.item())

            print("epoch = %4d   loss = %0.4f  l1_loss= %0.4f  validation_loss= %0.4f,  param_norm = %0.4f" % \
            (epoch, epoch_loss/n_batches, l1_coeff*l1_loss, loss_val.item() ,param_norm()))
        #--------------------------------------------------------------

    print("Training complete \n")

    print("\ntotal parameters' norm: ", param_norm())


    # Iterate through the model parameters and print out their values
    for name, param in model.named_parameters():
        if 'LinearRegression' in name:  # Only print encoder parameters
            print(name, param.data)

    # --------------------------------------EVALUATION OF TRAINING SET--------------------------------------------
    model.eval()
    train_predictions = model(train_dataset[:][0]).detach().numpy()
    y_train = train_dataset[:][1].numpy()

    # Set matplotlib fig. size, etc...
    myplot.configure()

    # Plot loss curves
    myplot.plot_loss_curves()

    # Plot densities of training set
    for idx in range(len(train_predictions[0])):
        filename = 'C:\\Users\\clock\\Desktop\\Python\\Images\\Full_ROM_model\\training' + species[idx]+'.png'
        plt.clf()
        a = y_train[:,idx] # target
        b = train_predictions[:,idx] # predicted
        myplot.plot_predict_target(b, a, sort_by_target=True)
        plt.title(species[idx])
        plt.savefig(filename)


#---------------------------------------------EVALUATION OF TEST SET------------------------------------------------------

    test_file = 'C:\\Users\\clock\\Desktop\\Python\\datapointsk1k2k3.txt'
    all_xy =  np.loadtxt(test_file,
        usecols=[0,1,2,3,4,5,6,7,8,9,10,11], delimiter="  ",
        # usecols=range(0,9), delimiter="\t",
        comments="#", skiprows=0, dtype=np.float64)

    tmp_x = all_xy[:,[9,10,11]]  # Change this manually
    tmp_y = all_xy[:,k_columns] 

    func1 = multipy(tmp_y, col1 = 0, col2 = 1)
    func2 = divide(tmp_y, col1 = 0, col2 = 1)
    func3 = add(tmp_y, col1 = 0, col2 = 1)

    stack = np.column_stack((func1,func2, func3, tmp_y[:,-1]))
    tmp_y = stack

    # Normalize data
    tmp_x = full_dataset.scaler_max_abs.transform(tmp_x)
    tmp_y = full_dataset.scaler.transform(tmp_y)

    x_data = T.tensor(tmp_y, \
        dtype=T.float64).to(device)
    y_data = T.tensor(tmp_x, \
        dtype=T.float64).to(device)

    predict =  model(x_data).detach().numpy()
    target = y_data.numpy()
    densities = x_data.numpy()

    # Create a scatter plot of the two densitie arrays against each other
    for idx in range(len(predict[0])):
        filename = 'C:\\Users\\clock\\Desktop\\Python\\Images\\Full_ROM_model\\correlations_test' + str(idx+1)+'.png'
        plt.clf()
        a = target[:,idx]
        b = predict[:,idx]
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
        plt.title(species[idx])
        # Add a diagonal line representing perfect agreement
        plt.plot([0, 1], [0, 1], linestyle='--', color='k')
        plt.savefig(filename)


