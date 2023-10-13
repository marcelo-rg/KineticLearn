import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.gridspec as gridspec

from src.config import dict as dictionary

folder_path = "D:\\Marcelo\\github\\Dissertation\\Images\\"

# Define the LoadMultiPressureDatasetNumpy class as provided
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
            # Comment out the next two lines to disable normalization
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
    

def create_split_histograms(features_1Torr, features_10Torr, species):
    
    # Find the global min and max values across all features
    global_min = min(features_1Torr.min(), features_10Torr.min())
    global_max = max(features_1Torr.max(), features_10Torr.max())

    for i, (feat_1Torr, feat_10Torr, spec) in enumerate(zip(features_1Torr.T, features_10Torr.T, species)):
        fig, ax = plt.subplots(1, 2, figsize=(15, 4))

        # Histogram for 1 Torr
        ax[0].hist(feat_1Torr, bins=60, color='blue', alpha=0.7, label='1 Torr', range=(global_min, global_max))
        ax[0].set_title(f'Distribution of {spec} at 1 Torr')
        ax[0].set_xlabel('Feature Value')
        ax[0].set_ylabel('Frequency')
        
        # Histogram for 10 Torr
        ax[1].hist(feat_10Torr, bins=60, color='green', alpha=0.7, label='10 Torr', range=(global_min, global_max))
        ax[1].set_title(f'Distribution of {spec} at 10 Torr')
        ax[1].set_xlabel('Feature Value')
        ax[1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(folder_path + f'histogram_{i}.pdf')
        plt.close(fig)  # Close the figure to free up memory






# Function to create and save box plots as separate figures, but retaining 2 columns for each pressure condition
def create_split_boxplots(features_1Torr, features_10Torr, species, figure_num):
    plt.clf()
    # Divide the features and species into two sets
    features_1Torr_set1 = features_1Torr[:, :7]
    features_1Torr_set2 = features_1Torr[:, 7:]
    features_10Torr_set1 = features_10Torr[:, :7]
    features_10Torr_set2 = features_10Torr[:, 7:]
    species_set1 = species[:7]
    species_set2 = species[7:]
    
    # Create and save the first set of box plots
    fig1, axes1 = plt.subplots(7, 2, figsize=(15, 25))
    # fig1.suptitle(f'Box Plots of Features for Different Pressure Conditions (Figure {figure_num}a)', fontsize=16)
    for i, (feat_1Torr, feat_10Torr, spec) in enumerate(zip(features_1Torr_set1.T, features_10Torr_set1.T, species_set1)):
        # Box plot for 1 Torr
        sns.boxplot(x=feat_1Torr, ax=axes1[i, 0], color='blue')
        axes1[i, 0].set_title(f'Distribution of {spec} at 1 Torr')
        axes1[i, 0].set_xlabel('Feature Value')
        # Box plot for 10 Torr
        sns.boxplot(x=feat_10Torr, ax=axes1[i, 1], color='green')
        axes1[i, 1].set_title(f'Distribution of {spec} at 10 Torr')
        axes1[i, 1].set_xlabel('Feature Value')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.savefig(folder_path + 'boxplot1.pdf')
    plt.show()
    
    # Create and save the second set of box plots
    fig2, axes2 = plt.subplots(4, 2, figsize=(15, 15))
    # fig2.suptitle(f'Box Plots of Features for Different Pressure Conditions (Figure {figure_num}b)', fontsize=16)
    for i, (feat_1Torr, feat_10Torr, spec) in enumerate(zip(features_1Torr_set2.T, features_10Torr_set2.T, species_set2)):
        # Box plot for 1 Torr
        sns.boxplot(x=feat_1Torr, ax=axes2[i, 0], color='blue')
        axes2[i, 0].set_title(f'Distribution of {spec} at 1 Torr')
        axes2[i, 0].set_xlabel('Feature Value')
        # Box plot for 10 Torr
        sns.boxplot(x=feat_10Torr, ax=axes2[i, 1], color='green')
        axes2[i, 1].set_title(f'Distribution of {spec} at 10 Torr')
        axes2[i, 1].set_xlabel('Feature Value')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.savefig(folder_path + 'boxplot2.pdf')
    plt.show()



# Function to create and save correlation heatmaps as a single figure
def create_correlation_heatmaps(features_1Torr, features_10Torr, species, figure_num):
    # Convert the numpy arrays to Pandas DataFrames for easier plotting
    df_1Torr = pd.DataFrame(features_1Torr, columns=species)
    df_10Torr = pd.DataFrame(features_10Torr, columns=species)
    
    # Set up the gridspec layout
    fig = plt.figure(figsize=(18, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2])  # 2 main plots and a narrow column for the colorbar
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    # ax_cbar = plt.subplot(gs[2])  # This will be used for the colorbar
    
    # Heatmap for 1 Torr with no colorbar
    sns.heatmap(df_1Torr.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax0, cbar=False)
    ax0.set_title('Correlation Heatmap at 1 Torr')
    
    # Heatmap for 10 Torr with colorbar
    sns.heatmap(df_10Torr.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax1, cbar=True)
    ax1.set_title('Correlation Heatmap at 10 Torr')
    
    plt.tight_layout()
    plt.savefig(folder_path + 'correlations_heatmap.pdf')
    # plt.show()







if __name__ == '__main__':
    # Select scheme
    scheme = 'O2_novib'

    # Parameters for data loading
    src_file_train = dictionary[scheme]['main_dataset']
    src_file_test = dictionary[scheme]['main_dataset_test']
    nspecies = dictionary[scheme]['n_densities']
    num_pressure_conditions = dictionary[scheme]['n_conditions']

    # Configuration dictionary provided
    config = {
        'n_conditions': 2,
        'k_columns': [0, 1, 2],
        'n_densities': 11,
        'species': ['O2(X)', 'O2(a)', 'O2(b)', 'O2(A3Su+_)', 'O2(+,X)', 'O($^3$P)', 'O(1D)', 'O(+,gnd)', 'O(-,gnd)', 'O3(X)', 'O3(exc)'],
    }

    # Load the training data
    dataset_loader = LoadMultiPressureDatasetNumpy(src_file_train, nspecies, num_pressure_conditions, react_idx=dictionary[scheme]['k_columns'])

    # Retrieve the processed data
    x_data, y_data = dataset_loader.get_data()

    # Show the shape of the loaded data
    print(x_data.shape, y_data.shape)

    # Extract features corresponding to each pressure condition
    features_1Torr = x_data[:, :11]
    features_10Torr = x_data[:, 11:]

    plt.rcParams.update({'font.size': 16, 'text.usetex': True})

    # Create histograms
    create_split_histograms(features_1Torr, features_10Torr, config['species'])

    # Create box plots
    # create_split_boxplots(features_1Torr, features_10Torr, config['species'], 1)

    # Create the correlation heatmaps
    # create_correlation_heatmaps(features_1Torr, features_10Torr, config['species'], 3)
    