import torch
from torch.nn import MSELoss
from torch.optim import Adam
import os

from src.Model import NSurrogatesModel
from src.Trainer import NSurrogatesModelTrainer
from src.DataHandler import LoadDataset, LoadMultiPressureDataset
from src.PlottingTools import PlottingTools
from src.config import dict
from seeds_analysis import create_histograms, plot_mean_val_loss, plot_heatmap

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
n_surrog = 2 # number of surrogate models
folder = '50epochs'
path = os.path.join(
    scheme,
    folder,
)
#############################################################

n_param = dict[scheme]['n_param'] # number of input densities
k_columns = dict[scheme]['k_columns']

# Define the model parameters
input_size = dict[scheme]['n_densities'] # number of input densities
output_size = len(k_columns)  # number of coefficients
max_epoch = 200


# Load surrogate datasets 
datasets = [LoadDataset(src_file= dict[scheme]['surrogate_dataset']+f"{i}.txt", nspecies=n_param, react_idx=k_columns) for i in range(n_surrog)]


# Load main net dataset
main_dataset = LoadMultiPressureDataset(src_file= dict[scheme]['main_dataset'], nspecies=n_param, num_pressure_conditions=n_surrog, react_idx=k_columns,
                                         scaler_input=[datasets[i].scaler_input for i in range(n_surrog)], scaler_output=[datasets[i].scaler_output for i in range(n_surrog)])

main_dataset_test = LoadMultiPressureDataset(src_file= dict[scheme]['main_dataset_test'], nspecies=n_param, num_pressure_conditions=n_surrog, react_idx=k_columns,\
                            scaler_input=main_dataset.scaler_input, scaler_output=main_dataset.scaler_output)

# Specify loss function
criterion = MSELoss()

# gen list of random seeds
seeds = [i for i in range(10)]

# --------------------   Model   -------------------- #
# List of hidden sizes for different architectures
hidden_sizes = [(10,), (20,), (30,), (40,), (50,),
                (10, 10), (20, 20), (30, 30), (40, 40), (50, 50),
                (10, 10, 10), (20, 20, 20), (30, 30, 30), (40, 40, 40), (50, 50, 50)]

# for O2_novib no activation function:
# hidden_sizes = [(10,), (20,), (30,), (40,), (50,), (60,), (80,), (100,)] 
hidden_sizes = [(10, 10)]

# start_position = hidden_sizes.index((10, 10))
# plot_mean_val_loss(hidden_sizes, path)
# plot_heatmap(hidden_sizes, path)
# exit()


# Initialize plotting tools
plotter = PlottingTools(dict[scheme]['species'])

for hidden_size in hidden_sizes:
    # Directory for the current architecture
    current_arch_dir = os.path.join("checkpoints", path, f"{hidden_size}_checkpoints")
    if not os.path.exists(current_arch_dir):
        os.makedirs(current_arch_dir)

    loss_list = []
    rel_error_list = []

    for idx, seed in enumerate(seeds):
        torch.manual_seed(seed)

        # Initialize model
        model = NSurrogatesModel(input_size, output_size, hidden_size, n_surrog)

        # Specify optimizer with parameters for main net (surrogate models are frozen)
        optimizer = Adam(model.main_net.parameters(), lr=0.1)

        # --------------------   Training   -------------------- #

        # Create trainer
        trainer = NSurrogatesModelTrainer(model, datasets, device, criterion, optimizer)

        # Load surrogate models
        trainer.load_surrogate_models()

        main_net = trainer.model.main_net

        # Pretrain main net
        training_losses_pretrain, validation_losses_pretrain = trainer.pretrain_model(main_dataset, pretrain_epochs=50, lr_rate=0.01)

        # Train main net
        training_losses_main, validation_losses_main = trainer.train_main_model(main_dataset, epochs = 250, lr_rate=0.1)

        # Save info
        loss_list.append([training_losses_main['main_model'][-1], validation_losses_main['main_model'][-1]])
        rel_error_list.append(plotter.get_relative_error(main_net,main_dataset_test))

        # Save model parameters
        main_net.save_model(os.path.join(path, f"{hidden_size}_checkpoints", f"main_model_seed{idx}.pth"))

        # Plot training and validation losses pretrain
        plotter.plot_loss_history(training_losses_pretrain, validation_losses_pretrain, os.path.join(current_arch_dir ,f"loss_history_pretrain_seed{idx}.png"))

        # Plot training and validation losses main
        plotter.plot_loss_history(training_losses_main, validation_losses_main, os.path.join(current_arch_dir ,f"loss_history_main_seed{idx}.png"))

    # write info to file
    with open(os.path.join(current_arch_dir ,f"log_table.txt"), 'w') as f:
        for i in range(len(seeds)):
            # Convert elements of rel_error_list[i] into strings and join them with comma
            rel_error_str = ','.join([str(elem) for elem in rel_error_list[i]])
            f.write(f'{loss_list[i][0]},{loss_list[i][1]},{rel_error_str}\n')
        


# Analysis with plots
create_histograms(hidden_sizes, path)
plot_mean_val_loss(hidden_sizes, path)
plot_heatmap(hidden_sizes, path)