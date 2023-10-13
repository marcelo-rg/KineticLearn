import os
import torch
from torch.nn import MSELoss
from torch.optim import Adam

from src.Model import NSurrogatesModel
from src.Trainer import NSurrogatesModelTrainer
from src.DataHandler import LoadDataset, LoadMultiPressureDataset
from src.PlottingTools import PlottingTools
from src.config import dict


# recover reproducibility
torch.manual_seed(8)

# Specify device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#############################################################
##                                                         ##
##   ############# Specify scheme and path #############   ## 
##                                                         ##
#############################################################
scheme = 'O2_novib'
hidden_size = (10, 10) 
seed_idx = 8
n_surrog = 2 # number of surrogate models
folder = '200epochs'
path = os.path.join(
    scheme,
    folder,
    f"{hidden_size}_checkpoints",
    f"main_model_seed{seed_idx}.pth",
)
#############################################################


n_param = dict[scheme]['n_param'] # number of input densities
k_columns = dict[scheme]['k_columns']

# Define the model parameters
input_size = dict[scheme]['n_densities'] # number of input densities
output_size = len(k_columns)  # number of coefficients
max_epoch = 200

# Initialize your model
plotter = PlottingTools(species= dict[scheme]['species'])

# Initialize your model
model = NSurrogatesModel(input_size, output_size, hidden_size, n_surrog)


# Load surrogate datasets
datasets = [LoadDataset(src_file=dict[scheme]['surrogate_dataset']+f"{i}.txt", nspecies=n_param, react_idx=k_columns) for i in range(n_surrog)]

# Load main net datasets
main_dataset = LoadMultiPressureDataset(src_file= dict[scheme]['main_dataset'], nspecies=n_param, num_pressure_conditions=n_surrog, react_idx=k_columns,
                                         scaler_input=[datasets[i].scaler_input for i in range(n_surrog)], scaler_output=[datasets[i].scaler_output for i in range(n_surrog)])

main_dataset_test = LoadMultiPressureDataset(src_file= dict[scheme]['main_dataset_test'], nspecies=n_param, num_pressure_conditions=n_surrog, react_idx=k_columns,\
                            scaler_input=main_dataset.scaler_input, scaler_output=main_dataset.scaler_output)

# Specify loss function
criterion = MSELoss()

# Specify optimizer with parameters for main net (surrogate models are frozen)
optimizer = Adam(model.main_net.parameters(), lr=0.1)

# --------------------   Loading   -------------------- #

# Create trainer
trainer = NSurrogatesModelTrainer(model, datasets, device, criterion, optimizer)

# Load surrogate models
trainer.load_surrogate_models()

# Load main net
trainer.model.main_net.load_model(path)

# -------------------   Evaluation   ------------------- #

# Move model to cpu
if device.type == 'cuda':  
    model = model.to('cpu')

# Get main net
main_net = trainer.model.main_net

# Plot validation of main net
plotter.plot_predictions_main(model, main_dataset_test, \
                              filename="predictions_vs_true_values_main.png")
