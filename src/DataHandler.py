import numpy as np
import torch 
from sklearn import preprocessing
# torch.set_default_tensor_type(torch.DoubleTensor)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------
class LoadDataset(torch.utils.data.Dataset):
    """A custom dataset class for loading and preprocessing data.

    Args:
        src_file (str): The path to the source file containing the data.
        nspecies (int): The number of species.
        react_idx (list or None, optional): The indices of the columns representing the output (k's). 
                      If None, the first 9 columns are considered as output. Default is None.
        m_rows (int or None, optional): The maximum number of rows to load from the source file. 
                      Default is None, which loads all rows.
        columns (list or None, optional): The indices of the columns to load from the source file. 
                      Default is None, which loads all columns.
        scaler_input (sklearn.preprocessing.MaxAbsScaler, optional): The MaxAbsScaler used for scaling the input (densities and pressure). 
                      If None, a new scaler is created. Default is None.
        scaler_output (sklearn.preprocessing.MaxAbsScaler, optional): The MaxAbsScaler used for scaling the output (k's). 
                      If None, a new scaler is created. Default is None.

    Attributes:
        scaler_input (sklearn.preprocessing.MaxAbsScaler): The MaxAbsScaler used for scaling the input (densities and pressure).
        scaler_output (sklearn.preprocessing.MaxAbsScaler): The MaxAbsScaler used for scaling the output (k's).
        x_data (torch.Tensor): The preprocessed input data (densities and pressure).
        y_data (torch.Tensor): The preprocessed output data (k's).
        all_data (torch.Tensor): The original unprocessed data.

    """

    def __init__(self, src_file, nspecies, react_idx = None, m_rows=None, columns= None, scaler_input = None, scaler_output = None):
        super(LoadDataset, self).__init__()
        self.scaler_input = scaler_input
        self.scaler_output = scaler_output

        all_xy = np.loadtxt(src_file, max_rows=m_rows,
            usecols=columns, delimiter="  ",
            # usecols=range(0,9), delimiter="\t", delimter= any whitespace by default
            comments="#", skiprows=0, dtype=np.float64)

        ncolumns = len(all_xy[0])
        y_columns = np.arange(ncolumns-nspecies,ncolumns,1)
        x_columns = react_idx
        if react_idx == None:
            x_columns = np.arange(0,ncolumns-nspecies,1)

        tmp_x = all_xy[:,x_columns]*10 # k's  #*10 to avoid being at float32 precision limit 1e-17  
        tmp_y = all_xy[:,y_columns] # densities

        # Create scalers
        if scaler_input == None and scaler_output == None:
            self.scaler_input = preprocessing.MaxAbsScaler()  
            self.scaler_output = preprocessing.MaxAbsScaler()
            self.scaler_input.fit(tmp_x) 
            self.scaler_output.fit(tmp_y)   
        
        # Scale data
        tmp_x = self.scaler_input.transform(tmp_x)
        tmp_y = self.scaler_output.transform(tmp_y)


        # Convert to tensors
        self.x_data = torch.tensor(tmp_x, \
            dtype=torch.float64).to(device)
        self.y_data = torch.tensor(tmp_y, \
            dtype=torch.float64).to(device)
        self.all_data = torch.tensor(all_xy, \
            dtype=torch.float64).to(device)

    def __len__(self):
        """Return the length of the dataset.

        Returns:
            int: The number of samples in the dataset.

        """
        return len(self.x_data)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset at the given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple of two matrices representing the densities and coefficients.

        """
        densities = self.x_data[idx,:]  # or just [idx]
        coef = self.y_data[idx,:] 
        return (densities, coef) # tuple of two matrices 
    



if __name__ == "__main__":
    k_idx = [0,1,2]
    dataset = LoadDataset(src_file="data/datapoints_pressure_3k.txt", nspecies=3, react_idx=k_idx, m_rows=None, columns=None)
    print(dataset.x_data.shape)
    print(dataset.y_data.shape)