import torch
from torch.nn import MSELoss
from torch.optim import Adam
import time

from src.Model import NSurrogatesModel
from src.Trainer import NSurrogatesModelTrainer
from src.DataHandler import LoadDataset, LoadMultiPressureDataset
from src.PlottingTools import PlottingTools

# recover reproducibility
torch.manual_seed(0)

# Specify device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Specify number of surrogate models and densities
n_surrog = 2 # number of surrogate models 
n_param = 3 # number of input densities
k_columns = [0,1,2]

# Define the model parameters
input_size = 3 # number of input densities
output_size = 3  # number of coefficients
hidden_size = (10,10)  # architecture of the main model
max_epoch = 200

# Initialize your model
model = NSurrogatesModel(input_size, output_size, hidden_size, n_surrog)

# Load surrogate datasets
datasets = [LoadDataset(src_file=f"data/datapoints_pressure_{i}.txt", nspecies=3, react_idx=k_columns) for i in range(n_surrog)]

# Load main net dataset
main_dataset = LoadMultiPressureDataset(src_file="data/datapoints_mainNet.txt", nspecies=3, num_pressure_conditions=n_surrog, react_idx=k_columns)

# Specify loss function
criterion = MSELoss()

# Specify optimizer with parameters of all models
optimizer = Adam(model.parameters(), lr=0.0001)

# Create trainer
trainer = NSurrogatesModelTrainer(model, datasets, device, criterion, optimizer)

start = time.time()
# Train surrogate models
training_losses, validation_losses = trainer.train_surrg_models(max_epoch)

# Train main net
training_losses_main, validation_losses_main = trainer.train_main_model(main_dataset, max_epoch)

end = time.time()
print("Training time: ", end - start)

# for evaluation, move model to cpu
if device.type == 'cuda':  
    model = model.to('cpu')

# # Plot training and validation loss histories
# plotter = PlottingTools()
# plotter.plot_loss_history(training_losses, validation_losses)

# # Get surrogate models
# surrogate_model_0 = model.surrog_nets[0]
# surrogate_model_1 = model.surrog_nets[1]

# # Plot validation using test dataset
# test_dataset = LoadDataset(src_file="data/datapoints_pressure_0_test.txt", nspecies=3, react_idx=k_columns,\
#                             scaler_input=datasets[0].scaler_input, scaler_output=datasets[0].scaler_output)
# plotter.plot_predictions(surrogate_model_0, test_dataset, filename="predictions_vs_true_values_0.png")

# # Plot validation using test dataset
# test_dataset = LoadDataset(src_file="data/datapoints_pressure_1_test.txt", nspecies=3, react_idx=k_columns,\
#                             scaler_input=datasets[1].scaler_input, scaler_output=datasets[1].scaler_output)
# plotter.plot_predictions(surrogate_model_1, test_dataset, filename="predictions_vs_true_values_1.png")


