import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch as T
device = T.device("cpu")
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import lassonet


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

        tmp_coeff = np.column_stack((func1,func2, func3, tmp_coeff))
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

        stack = np.column_stack((func1,func2, func3, tmp_y))
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

class SparseLayer(T.nn.Module):
    def __init__(self, input_size, output_size, connectivity) -> None:
        super(SparseLayer, self).__init__()
        self.in_size = input_size
        self.out_size = output_size
        self.connectivity = connectivity
        self.weight = T.nn.Parameter(T.randn(input_size, output_size))
    
    def forward(self, x):
        # apply sparse connectivity pattern to weight matrix
        x = T.matmul(x, self.weight*self.connectivity) #include .t()?
        return x

# ------------------------------------------------------------------------------

class Full_ROM(T.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Full_ROM, self).__init__()

        self.connectivity = T.eye(input_size)

        self.LinearRegression = SparseLayer(input_size, input_size, self.connectivity)
        
        # set weights outside the diagonal to zero (apply mask)
        self.LinearRegression.weight.data *= self.connectivity

            
        self.decoder = T.nn.Sequential(
            T.nn.Linear(input_size, hidden_size),
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
        plt.savefig('Images\\Full_ROM_model\\loss_curve.png')
    
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
    src_file = 'data\\datapointsk1k2k3_3k.txt' 
    species = ['O2(X)', 'O2(a)', 'O(3P)']
    k_columns = [0,1,2] # Set to None to read all reactions/columns in the file
    full_dataset = LoadDataset(src_file, nspecies= len(species), react_idx= k_columns) #(data already scaled)
    nfunctions = 6
    feature_names = ['k1*k2','k1/k2','k1+k2','k1', 'k2', 'k3']

    dir_path = "Images\\statistics\\stacks\\500_samples\\"
    samples_dataset = Samples_Dataset(features_file=dir_path+"stack_train.pt",
                                       targets_file=dir_path+"densities_targets_train.npy")


    # Choose optimizer
    # optimizer = T.optim.Adam(lr=1e-3)

    # 1 Split into training and validation sets | samples_dataset -> full_dataset
    train_size = int(0.90 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = T.utils.data.random_split(full_dataset, [train_size, test_size])

    # 2. Create LassoNet model
    # patience: Number of epochs to wait without improvement during early stopping.
    model = lassonet.LassoNetRegressor(
        n_iters = (1000,100), 
        patience = 10, 
        hidden_dims=(100,), 
        torch_seed= 8, 
        lambda_start= 'auto', 
        M=10, 
        batch_size=20, 
        path_multiplier=1.1,
        # gamma= 1e-4,
        # gamma_skip= 1e-4,
    ) #, optimizer=optimizer)
    
    X_train = train_dataset[:][0].detach().numpy()
    Y_train = train_dataset[:][1].detach().numpy()
    x_test = test_dataset[:][0].detach().numpy()
    y_test = test_dataset[:][1].detach().numpy()

    # 3. train model
    path = model.path(X_train, Y_train)
    # path = model.fit(X_train, Y_train)


    # 4. Plot results ----------------------------------------------------------
    fig = plt.figure(figsize=(12, 12))
    # def plot_results_path():
    n_selected = []
    mse = []
    lambda_ = []

    for save in path:
        model.load(save.state_dict)
        y_pred = model.predict(x_test)
        n_selected.append(save.selected.sum().cpu().numpy())
        mse.append(mean_squared_error(y_test, y_pred))
        lambda_.append(save.lambda_)

    mse_min_idx = np.argmin(mse)
    lambda_min_mse = lambda_[mse_min_idx]
    n_selected_min_mse = n_selected[mse_min_idx]
    
    # y_predic = model.predict(x_test)
    plt.subplot(311)
    plt.grid(True)
    plt.plot(n_selected, mse, ".-")
    plt.xlabel("number of selected features")
    plt.ylabel("MSE")

    plt.subplot(312)
    plt.grid(True)
    plt.plot(lambda_, mse, ".-")
    plt.xlabel("lambda")
    plt.xscale("log")
    plt.ylabel("MSE")

    plt.subplot(313)
    plt.grid(True)
    plt.plot(lambda_, n_selected, ".-")
    plt.xlabel("lambda")
    plt.xscale("log")
    plt.ylabel("number of selected features")

    plt.savefig("Images\\Full_ROM_model\\lasso_net.png")

    plt.clf()

    n_features = X_train.shape[1]
    importances = model.feature_importances_.numpy() #When does each feature disappear on the path?
    order = np.argsort(importances)[::-1]
    importances = importances[order]
    ordered_feature_names = [feature_names[i] for i in order]
    # color = np.array(["g"] * true_features + ["r"] * (n_features - true_features))[order]


    plt.subplot(211)
    plt.bar(
        np.arange(n_features),
        importances,
        # color=color,
    )
    plt.xticks(np.arange(n_features), ordered_feature_names, rotation=90)
    colors = {"real features": "g", "fake features": "r"}
    labels = list(colors.keys())
    # handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
    # plt.legend(handles, labels)
    plt.ylabel("Feature importance")

    _, order = np.unique(importances, return_inverse=True)

    plt.subplot(212)
    plt.bar(
        np.arange(n_features),
        order + 1,
        # color=color,
    )
    plt.xticks(np.arange(n_features), ordered_feature_names, rotation=90)
    # plt.legend(handles, labels)
    plt.ylabel("Feature order")

    plt.savefig("Images\\Full_ROM_model\\lasso_net-bar.png")



    # --------------------------------------EVALUATION OF TRAINING SET--------------------------------------------
    model.load(path[mse_min_idx].state_dict)
    train_predictions = model.predict(X_train)

    # Set matplotlib fig. size, etc...
    myplot = MyPlots()
    myplot.configure()

    # Plot loss curves
    # myplot.plot_loss_curves()

    # Plot densities of training set
    for idx in range(len(train_predictions[0])):
        filename = 'Images\\Full_ROM_model\\training' + species[idx]+'.png'
        plt.clf()
        a = Y_train[:,idx] # target
        b = train_predictions[:,idx] # predicted
        myplot.plot_predict_target(b, a, sort_by_target=True)
        plt.title(species[idx])
        plt.savefig(filename)


#---------------------------------------------EVALUATION OF TEST SET------------------------------------------------------

    test_file = 'data\\datapointsk1k2k3.txt'
    all_xy =  np.loadtxt(test_file,
        usecols=[0,1,2,3,4,5,6,7,8,9,10,11], delimiter="  ",
        # usecols=range(0,9), delimiter="\t",
        comments="#", skiprows=0, dtype=np.float64)

    tmp_x = all_xy[:,[9,10,11]]  # Change this manually
    tmp_y = all_xy[:,k_columns] 

    func1 = multipy(tmp_y, col1 = 0, col2 = 1)
    func2 = divide(tmp_y, col1 = 0, col2 = 1)
    func3 = add(tmp_y, col1 = 0, col2 = 1)

    stack = np.column_stack((func1,func2, func3, tmp_y))
    tmp_y = stack

    # Normalize data
    tmp_x = full_dataset.scaler_max_abs.transform(tmp_x)
    tmp_y = full_dataset.scaler.transform(tmp_y)

    x_data = T.tensor(tmp_y, \
        dtype=T.float64).to(device)
    y_data = T.tensor(tmp_x, \
        dtype=T.float64).to(device)

    predict =  model.predict(x_data.detach().numpy())
    target = y_data.numpy()
    densities = x_data.numpy()


    plt.clf()
    lassonet.plot.plot_path(model,path, x_data.detach().numpy(), target)
    plt.savefig("Images\\Full_ROM_model\\lasso_net-plot-path.png")
    myplot.configure()

    # Create a scatter plot of the two densitie arrays against each other
    for idx in range(len(predict[0])):
        filename = 'Images\\Full_ROM_model\\correlations_test' + str(idx+1)+'.png'
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

        # Final print
        print("num selected features: ", n_selected_min_mse)
        print("lambda: ",lambda_min_mse)
        print("MSE: ", mse[mse_min_idx])