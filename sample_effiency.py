import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
# import time
import os

folder_path = "D:\\Marcelo\\github\\Dissertation\\Images\\"

class LoadMultiPressureDatasetNumpy:

    def __init__(self, src_file, nspecies, num_pressure_conditions, react_idx=None, m_rows=None, columns=None,
                 scaler_input=None, scaler_output=None):
        self.num_pressure_conditions = num_pressure_conditions

        all_data = np.loadtxt(src_file, max_rows=m_rows,
                              usecols=columns, delimiter="  ",
                              comments="#", skiprows=0, dtype=np.float64)

        ncolumns = len(all_data[0])
        x_columns = np.arange(ncolumns - nspecies, ncolumns, 1)
        y_columns = react_idx
        if react_idx is None:
            y_columns = np.arange(0, ncolumns - nspecies, 1)

        x_data = all_data[:, x_columns]  # densities
        y_data = all_data[:, y_columns] * 1e30  # k's  # *10 to avoid being at float32 precision limit 1e-17

        # Reshape data for multiple pressure conditions
        x_data = x_data.reshape(num_pressure_conditions, -1, x_data.shape[1])
        y_data = y_data.reshape(num_pressure_conditions, -1, y_data.shape[1])

        # Create scalers
        self.scaler_input = scaler_input or [preprocessing.MaxAbsScaler() for _ in range(num_pressure_conditions)]
        self.scaler_output = scaler_output or [preprocessing.MaxAbsScaler() for _ in range(num_pressure_conditions)]
        
        for i in range(num_pressure_conditions):
            if scaler_input is None:
                self.scaler_input[i].fit(x_data[i])
            if scaler_output is None:
                self.scaler_output[i].fit(y_data[i])
            x_data[i] = self.scaler_input[i].transform(x_data[i])
            y_data[i] = self.scaler_output[i].transform(y_data[i])

        # Transpose x_data to move the pressure condition axis to the end, then flatten
        x_data = np.transpose(x_data, (1, 0, 2)).reshape(-1, self.num_pressure_conditions * x_data.shape[-1])
        
        # Flatten the output data to be of shape (2000,3)
        y_data = y_data[0]


        # Assign the preprocessed data
        self.x_data = x_data
        self.y_data = y_data
        self.all_data = all_data

    def get_data(self):
        """
        Return the preprocessed input and output data.
        """
        return self.x_data, self.y_data


def generate_subsets(x_data, y_data, subset_sizes):
    subsets = []
    for size in subset_sizes:
        x_subset = x_data[:size]
        y_subset = y_data[:size]
        subsets.append((x_subset, y_subset))
    return subsets


def calculate_mse_for_dataset(dataset_train, dataset_test, best_params, subset_sizes, seed=40):
    x_train, y_train = dataset_train.get_data()
    x_test, y_test = dataset_test.get_data()

    x_train, y_train = shuffle(x_train, y_train, random_state= seed)  # set a random_state for reproducibility

    data_subsets = generate_subsets(x_train, y_train, subset_sizes)

    mse_list = []

    for i in range(y_train.shape[1]):
        mse_output = []
        for (x_subset, y_subset) in data_subsets:
            model = SVR(C=best_params[i]['C'], epsilon=best_params[i]['epsilon'], 
                        gamma=best_params[i]['gamma'], kernel=best_params[i]['kernel'])

            model.fit(x_subset, y_subset[:,i])

            y_pred = model.predict(x_test)

            mse = mean_squared_error(y_test[:, i], y_pred)

            mse_output.append(mse)
        
        mse_list.append(mse_output)

    total_mse_list = np.sum(np.array(mse_list), axis=0)

    return mse_list, total_mse_list


if __name__ == "__main__":
    nspecies = 3
    num_pressure_conditions = 2
    subset_sizes = [i for i in range(200, 2100, 200)]
    num_seeds = 20  # Number of seeds to use
    print("Subset sizes:", subset_sizes)
    best_params = [
        {'C': 10, 'epsilon': 0.005, 'gamma': 2, 'kernel': 'rbf'},
        {'C': 20, 'epsilon': 0.005, 'gamma': 5, 'kernel': 'rbf'},
        {'C': 5, 'epsilon': 0.005, 'gamma': 2, 'kernel': 'rbf'}
    ]

    datasets = [
        'O2_simple_log.txt',
        'O2_simple_morris.txt',
        'O2_simple_uniform.txt',
        'O2_simple_latin_log_uniform.txt',
        'O2_simple_latin.txt',
    ]
    labels = [
        'Log-Uniform',
        'Morris Method',
        'Uniform',
        'Log-Uniform Latin Hypercube',
        'Uniform Latin Hypercube',
    ]

    # set figure size
    plt.figure(figsize=(9, 6))

    for idx, dataset in enumerate(datasets):
        src_file_train = 'data/SampleEfficiency/' + dataset
        src_file_test = 'data/SampleEfficiency/O2_simple_test.txt'

        dataset_train = LoadMultiPressureDatasetNumpy(src_file_train, nspecies, num_pressure_conditions, react_idx=[0, 1, 2])
        dataset_test = LoadMultiPressureDatasetNumpy(src_file_test, nspecies, num_pressure_conditions, react_idx=[0, 1, 2], 
                                                    scaler_input=dataset_train.scaler_input, scaler_output=dataset_train.scaler_output)

        total_mse_for_seeds = []
        
        for seed in range(num_seeds):
            _, total_mse_list = calculate_mse_for_dataset(dataset_train, dataset_test, best_params, subset_sizes, seed=seed)
            total_mse_for_seeds.append(total_mse_list)
        
        mean_total_mse = np.mean(total_mse_for_seeds, axis=0)
        std_total_mse = np.std(total_mse_for_seeds, axis=0) / np.sqrt(num_seeds)

        plt.errorbar(subset_sizes, mean_total_mse, yerr=std_total_mse, label=labels[idx], marker='o')


    plt.rcParams.update({'font.size': 16})
    plt.xlabel('Dataset size', fontsize=14)
    plt.ylabel('MSE on test set', fontsize=14)
    # plt.yscale('log')
    plt.legend(loc='upper right', fontsize=14)
    # plt.title('Sample efficiency analysis')
    plt.grid(True)
    plt.tight_layout() 
    # plt.savefig(os.path.join('images', 'sample_efficiency','SVR_sample_efficiency_seeds.png'))
    # plt.show()
    plt.savefig(folder_path + 'sample_efficiency.pdf')