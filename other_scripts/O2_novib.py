import torch
from torch.nn import MSELoss
from torch.optim import Adam
import time

from src.Model import NSurrogatesModel
from src.Trainer import NSurrogatesModelTrainer
from src.DataHandler import LoadDataset, LoadMultiPressureDataset
from src.PlottingTools import PlottingTools
from src.config import dict

# recover reproducibility
torch.manual_seed(4)

# Specify device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

# Specify number of surrogate models and densities
n_surrog = 2 # number of surrogate models 
n_param = 11 # number of input densities
k_columns = [0,1,2]

# Define the model parameters
input_size = 11 # number of input densities
output_size = 3  # number of coefficients
hidden_size = (50,)  # architecture of the main model
max_epoch_surrg = 200    # maximum number of epochs for training surrogate models

# Initialize your model
model = NSurrogatesModel(input_size, output_size, hidden_size, n_surrog)


# Load surrogate datasets
datasets = [LoadDataset(src_file=f"data/datapoints_O2_novib_pressure_{i}.txt", nspecies=n_param, react_idx=k_columns) for i in range(n_surrog)]


# Load main net dataset
main_dataset = LoadMultiPressureDataset(src_file="data/datapoints_O2_novib_mainNet_2surrog.txt", nspecies=n_param, num_pressure_conditions=n_surrog, react_idx=k_columns,
                                         scaler_input=[datasets[i].scaler_input for i in range(n_surrog)], scaler_output=[datasets[i].scaler_output for i in range(n_surrog)])

# Specify loss function
criterion = MSELoss()

# Specify optimizer with parameters of all models
optimizer = Adam(model.parameters(), lr=0.0001)

# --------------------   Training   -------------------- #
# Create trainer
trainer = NSurrogatesModelTrainer(model, datasets, device, criterion, optimizer, batch_size=20)

start = time.time()
# Train surrogate models
# training_losses, validation_losses = trainer.train_surrg_models(max_epoch_surrg)
# trainer.save_surrogate_models()

# Load surrogate models
trainer.load_surrogate_models()

# trainer.freeze_surrogate_models()

# set new learning rate for main net
trainer.optimizer = Adam(model.main_net.parameters(), lr=0.001)

# Pretrain main net
training_losses_pretrain, validation_losses_pretrain = trainer.pretrain_model(main_dataset, pretrain_epochs=100, lr_rate=0.001)

# Train main net
# training_losses_main, validation_losses_main = trainer.train_main_model(main_dataset, epochs = 250, lr_rate=0.1)

end = time.time()
print("Training time: ", end - start)


# -------------------   Evaluation   ------------------- #
# Move model to cpu
if device.type == 'cuda':  
    model = model.to('cpu')

# Plot training and validation loss histories
plotter = PlottingTools(dict['O2_novib']['species'])
# plotter.plot_loss_history(training_losses, validation_losses) # surrogate models

# pretrain loss history
plotter.plot_loss_history(training_losses_pretrain, validation_losses_pretrain, filename="images/pretrain_loss_history.png") 

# main train loss history
# plotter.plot_loss_history(training_losses_main, validation_losses_main, filename="images/main_model_loss_history.png") 

# Get main net
main_net = model.main_net

# Loop through surrogate models
for i in range(n_surrog):
    surrogate_model = model.surrog_nets[i]
    
    # Load test dataset
    test_dataset = LoadDataset(src_file=f"data/datapoints_O2_novib_pressure_{i}_test.txt", nspecies=n_param, react_idx=k_columns,\
                                scaler_input=datasets[i].scaler_input, scaler_output=datasets[i].scaler_output) 
    
    # Plot validation
    plotter.plot_predictions_surrog(surrogate_model, test_dataset, filename=f"predictions_vs_true_values_{i}.png")

# Plot validation of main net
main_dataset_test = LoadMultiPressureDataset(src_file="data/datapoints_O2_novib_mainNet_2surrog_test.txt", nspecies=n_param, num_pressure_conditions=n_surrog, react_idx=k_columns,\
                            scaler_input=main_dataset.scaler_input, scaler_output=main_dataset.scaler_output) 
plotter.plot_predictions_main(model, main_dataset_test, filename="predictions_vs_true_values_main.png") 