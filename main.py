import torch
from torch.nn import MSELoss
from torch.optim import Adam
import time

from src.Model import NSurrogatesModel
from src.Trainer import NSurrogatesModelTrainer
from src.DataHandler import LoadDataset, LoadMultiPressureDataset
from src.PlottingTools import PlottingTools

# recover reproducibility
torch.manual_seed(101)

# Specify device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Specify number of surrogate models and densities
n_surrog = 2 # number of surrogate models 
n_param = 3 # number of input densities
k_columns = [0,1,2]

# Define the model parameters
input_size = 3 # number of input densities
output_size = 3  # number of coefficients
hidden_size = (20,20)  # architecture of the main model
max_epoch = 100

# Initialize your model
model = NSurrogatesModel(input_size, output_size, hidden_size, n_surrog)

# Load surrogate datasets
datasets = [LoadDataset(src_file=f"data/datapoints_pressure_{i}.txt", nspecies=3, react_idx=k_columns) for i in range(n_surrog)]

# Load main net dataset
main_dataset = LoadMultiPressureDataset(src_file="data/datapoints_mainNet01.txt", nspecies=3, num_pressure_conditions=n_surrog, react_idx=k_columns)

# Specify loss function
criterion = MSELoss()

# Specify optimizer with parameters of all models
optimizer = Adam(model.parameters(), lr=0.0001)

# --------------------   Training   -------------------- #

# Create trainer
trainer = NSurrogatesModelTrainer(model, datasets, device, criterion, optimizer)

start = time.time()
# Train surrogate models
training_losses, validation_losses = trainer.train_surrg_models(max_epoch)

# trainer.freeze_surrogate_models()

# set new learning rate for main net
trainer.optimizer = Adam(model.main_net.parameters(), lr=0.1)

# Train main net
training_losses_main, validation_losses_main = trainer.train_main_model(main_dataset, epochs = 100, pretrain=True)

end = time.time()
print("Training time: ", end - start)

# -------------------   Evaluation   ------------------- #

# Move model to cpu
if device.type == 'cuda':  
    model = model.to('cpu')

# # Plot training and validation loss histories
plotter = PlottingTools()
plotter.plot_loss_history(training_losses, validation_losses)
plotter.plot_loss_history(training_losses_main, validation_losses_main)

# Get surrogate models and main net
surrogate_model_0 = model.surrog_nets[0]
# surrogate_model_1 = model.surrog_nets[1]
main_net = model.main_net

# Plot validation using test dataset 0
test_dataset = LoadDataset(src_file="data/datapoints_pressure_0_test.txt", nspecies=3, react_idx=k_columns,\
                            scaler_input=datasets[0].scaler_input, scaler_output=datasets[0].scaler_output)
plotter.plot_predictions_surrog(surrogate_model_0, test_dataset, filename="predictions_vs_true_values_0.png")

# Plot validation using test dataset 1}
# test_dataset = LoadDataset(src_file="data/datapoints_pressure_1_test.txt", nspecies=3, react_idx=k_columns,\
#                             scaler_input=datasets[1].scaler_input, scaler_output=datasets[1].scaler_output)
# plotter.plot_predictions_surrog(surrogate_model_1, test_dataset, filename="predictions_vs_true_values_1.png")


# Plot validation of main net
main_dataset_test = LoadMultiPressureDataset(src_file="data/datapoints_mainNet_test_0.txt", nspecies=3, num_pressure_conditions=n_surrog, react_idx=k_columns,\
                            scaler_input=main_dataset.scaler_input, scaler_output=main_dataset.scaler_output)

plotter.plot_predictions_main(model, main_dataset_test, filename="predictions_vs_true_values_main.png")