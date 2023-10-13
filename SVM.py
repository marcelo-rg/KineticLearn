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

    # Load the training data
    dataset_train = LoadMultiPressureDatasetNumpy(src_file_train, nspecies, num_pressure_conditions, react_idx=dictionary[scheme]['k_columns'])
    x_train, y_train = dataset_train.get_data()

    # Load the test data
    dataset_test = LoadMultiPressureDatasetNumpy(src_file_test, nspecies, num_pressure_conditions, react_idx=dictionary[scheme]['k_columns'], 
                                                 scaler_input=dataset_train.scaler_input, scaler_output=dataset_train.scaler_output)
    x_test, y_test = dataset_test.get_data()


    # Check the shape of the data
    print(f"Shape of x_data: {x_train.shape}") # (2000, 9)
    print(f"Shape of y_data: {y_train.shape}") # (2000, 3)


    # Define an initial SVR model
    model = SVR()

    # Define the grid of hyperparameters to search
    param_grid = {
        'C': [0.1, 1, 10], # 3
        'epsilon': [0.01, 0.1, 1 ,10, 20], # 5
        'kernel': ['rbf'],
        'gamma': [0.01, 0.1, 1] # 3
    }

    refined_param_grid = {
        'C': [0.5, 1, 5, 10, 15, 20],
        'epsilon': [0.005, 0.01, 0.02, 0.03, 0.04, 0.05],
        'kernel': ['rbf'],
        'gamma': [0.05, 0.1, 0.5, 1, 2, 5]
    }

    # this is the old one that was used for the real search
    # refined_param_grid_v2 = {
    #     'C': [4, 5, 10, 15, 20, 25, 30],
    #     'epsilon': [0.001, 0.0025, 0.005, 0.0075, 0.01],
    #     'kernel': ['rbf'],
    #     'gamma': [1, 2, 3, 4, 5, 6, 7, 8]
    # }

    # this is the new one to be used in the heatmap
    refined_param_grid_v2 = {
        'C': [10, 15, 20, 25, 30], # 5
        'epsilon': [0.001, 0.0025, 0.005], # 3
        'kernel': ['rbf'],
        'gamma': [1, 2, 4, 5] # 4
    }

    refined_param_grid_v3 = {
        'C': [40, 50, 60, 70],  # Expanded to cover above 50, given that 50 is an optimal value
        'epsilon': [1e-05, 3e-05, 5e-05, 1e-04],  # Included even lower values because 5e-05 is at the edge
        'kernel': ['rbf'],  # No change as 'rbf' still appears to be the best option
        'gamma': [4, 6, 8, 10]  # Expanded to cover above 8, as 8 and 6 are optimal but close to the edge
    }

    

    # Create an empty list to store models for each output
    models = []
    grid_results = []

    ################## START GRID SEARCH ##################
    # start = time.time()
    # # For each output, perform hyperparameter tuning with GridSearchCV
    # for i in range(y_train.shape[1]):
    #     grid_search = GridSearchCV(model, refined_param_grid_v3, cv=5, scoring='neg_mean_squared_error')
    #     grid_search.fit(x_train, y_train[:, i])
    #     best_model = grid_search.best_estimator_
    #     models.append(best_model)
    #     grid_results.append(grid_search.cv_results_)

    #     # Print the best hyperparameters for this output
    #     print(f"Best hyperparameters for output {i}: {grid_search.best_params_}")
    # end = time.time()
    # print(f"Training time: {end-start}")

    # # Convert the grid results to a pandas DataFrame
    # df_grid_results = pd.concat([pd.DataFrame(g) for g in grid_results], keys=[f'Output_{i}' for i in range(y_train.shape[1])])
    # df_grid_results.to_csv('SVM_grid_search_results_revised.csv')  # Save to a CSV file for later use
    ################## END GRID SEARCH ##################


    ################## START PLOT OF BEST MODEL ##################
    # Hyperparameters of best models
    hyperparams_0 = {'C': 30, 'epsilon': 0.001, 'gamma': 1, 'kernel': 'rbf'}
    hyperparams_1 = {'C': 15, 'epsilon': 0.001, 'gamma': 1, 'kernel': 'rbf'}
    hyperparams_2 = {'C': 10, 'epsilon': 0.001, 'gamma': 4, 'kernel': 'rbf'}

    
    # Create the models with the specified hyperparameters
    svr_0 = SVR(C=hyperparams_0['C'], epsilon=hyperparams_0['epsilon'], gamma=hyperparams_0['gamma'], kernel=hyperparams_0['kernel'])
    svr_1 = SVR(C=hyperparams_1['C'], epsilon=hyperparams_1['epsilon'], gamma=hyperparams_1['gamma'], kernel=hyperparams_1['kernel'])
    svr_2 = SVR(C=hyperparams_2['C'], epsilon=hyperparams_2['epsilon'], gamma=hyperparams_2['gamma'], kernel=hyperparams_2['kernel'])

    # Combine the models into a list
    models = [svr_0, svr_1, svr_2]

    # Train each model on the training data
    for i in range(y_train.shape[1]):
        models[i].fit(x_train, y_train[:, i])

    # Save the list of trained SVR models
    joblib.dump(models, os.path.join('PhysicalPlots','svr_models.pkl'))
    ################## END PLOT OF BEST MODEL ##################


    # Create an empty list to store predictions for each output
    y_pred = []

    # For each model, predict the corresponding output
    for i in range(y_test.shape[1]):
        y_pred_i = models[i].predict(x_test)
        y_pred.append(y_pred_i)

    # Convert the list of predictions to a 2D array
    y_pred = np.array(y_pred).T

    # Calculate the Mean Squared Error (MSE) of the models
    mse = mean_squared_error(y_test, y_pred)

    # Print the MSE
    print(f"Mean Squared Error (MSE) of the models on the test data: {mse}")

    # Plot true vs predicted values for each output
    fig, axs = plt.subplots(1, y_test.shape[1], figsize=(15, 5), sharey=True)  # Share the same y-axis
    plt.rcParams.update({'font.size': 16, 'text.usetex': True})

    for i in range(len(axs)):
        axs[i].scatter(y_test[:, i], y_pred[:, i], alpha=0.8, color=(0.9, 0, 0))  # red
        # draw the y=x line
        axs[i].plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), '--', color='black')
        axs[i].set_xlabel('True Values', fontsize=14)
        
        # Only set the y-label for the first subplot since they share the same y-axis
        if i == 0:
            axs[i].set_ylabel('Predicted Values', fontsize=14)
        
        # LaTeX format for the title
        axs[i].set_title(f"$k_{{{i+1}}}$")

        # Calculate relative error
        denominator = y_test[:,i]
        denominator[np.abs(denominator) < 1e-9] = 1e-9  # Set small values to a small constant
        rel_err = np.abs(np.subtract(y_test[:,i], y_pred[:, i]) / denominator)

        textstr = '\n'.join((
            r'$Mean\ \delta_{rel}=%.2f\%%$' % (rel_err.mean() * 100,),
            r'$Max\ \delta_{rel}=%.2f\%%$' % (max(rel_err) * 100,)))


        # Colour point with max error
        max_index = np.argmax(rel_err)
        axs[i].scatter(y_test[max_index, i], y_pred[max_index, i], color="gold", zorder=2)

        # Define the text box properties
        props = dict(boxstyle='round', alpha=0.5)

        # Place a text box in upper left in axes coords
        axs[i].text(0.63, 0.25, textstr, fontsize=12, transform=axs[i].transAxes,
                verticalalignment='top', bbox=props)
        
        # Remove tick bars from non-first plots
        if i > 0:
            axs[i].tick_params(left=False)

    plt.tight_layout()
    # plt.savefig(os.path.join('images','SVR_'+ scheme +'.pdf'))
    plt.show()

