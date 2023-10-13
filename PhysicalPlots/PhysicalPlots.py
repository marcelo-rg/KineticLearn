import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import time
import os
import pandas as pd
import joblib

from src.config import dict as dictionary

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



if __name__ == "__main__":

    # Select scheme
    scheme = 'O2_novib'

    # Parameters for data loading
    src_file_train = dictionary[scheme]['main_dataset']
    src_file_test = dictionary[scheme]['main_dataset_test']
    nspecies = dictionary[scheme]['n_densities']
    num_pressure_conditions = dictionary[scheme]['n_conditions']

    loaded_models = joblib.load(os.path.join('PhysicalPlots','svr_models.pkl'))

    # Load the training data
    dataset_train = LoadMultiPressureDatasetNumpy(src_file_train, nspecies, num_pressure_conditions, react_idx=dictionary[scheme]['k_columns'])
    x_train, y_train = dataset_train.get_data()

    true_point = LoadMultiPressureDatasetNumpy(os.path.join('data','true_point.txt'), 
                                               nspecies=11, 
                                               num_pressure_conditions=2, 
                                               react_idx=[0, 1, 2], 
                                               scaler_input=dataset_train.scaler_input, 
                                               scaler_output=dataset_train.scaler_output)
    
    
    x_true, y_true = true_point.get_data()

    predictions = []

    for model in loaded_models:
        prediction = model.predict(x_true)
        predictions.append(prediction)


    # Convert the list of predictions to a 2D array
    y_pred = np.array(predictions).T
    scaler_output = dataset_train.scaler_output[0]

    # for prediction in y_pred:
    point_prediction = (scaler_output.inverse_transform(y_pred.reshape(1,-1))*1e-30).flatten()
    print(point_prediction)

    k_true_values = [7.6e-22, 3E-44, 4e-20]

    # compute relative error
    relative_error = (point_prediction - k_true_values)/k_true_values*100
    print(relative_error)   