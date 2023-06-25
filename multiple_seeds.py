import torch
from torch.nn import MSELoss
from torch.optim import Adam

from src.Model import NSurrogatesModel
from src.Trainer import NSurrogatesModelTrainer
from src.DataHandler import LoadDataset, LoadMultiPressureDataset
from src.PlottingTools import PlottingTools

# make directory to save multiple models
import os
if not os.path.exists('checkpoints/seeds_checkpoints'):
    os.makedirs('checkpoints/seeds_checkpoints')


# recover reproducibility
torch.manual_seed(8)

# Specify device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Specify number of surrogate models and densities
n_surrog = 3 # number of surrogate models 
n_param = 3 # number of input densities
k_columns = [0,1,2]

# Define the model parameters
input_size = 3 # number of input densities
output_size = 3  # number of coefficients
# hidden_size = (10,10)  # architecture of the main model
max_epoch = 200

# Initialize 
plotter = PlottingTools()

# --------------------   Data   -------------------- #

# Load surrogate datasets
datasets = [LoadDataset(src_file=f"data/datapoints_pressure_{i}.txt",nspecies=3, react_idx=k_columns) for i in range(n_surrog)]

# Load main net datasets
main_dataset = LoadMultiPressureDataset(src_file="data/datapoints_mainNet_2k.txt", nspecies=3, num_pressure_conditions=n_surrog, react_idx=k_columns,
                                         scaler_input=[datasets[i].scaler_input for i in range(n_surrog)], scaler_output=[datasets[i].scaler_output for i in range(n_surrog)])
main_dataset_test = LoadMultiPressureDataset(src_file="data/datapoints_mainNet_test.txt", nspecies=3, num_pressure_conditions=n_surrog, react_idx=k_columns,\
                            scaler_input=main_dataset.scaler_input, scaler_output=main_dataset.scaler_output, m_rows=3000)

# Specify loss function
criterion = MSELoss()

# gen list of random seeds
seeds = [i for i in range(25)]

# --------------------   Model   -------------------- #
# List of hidden sizes for different architectures
hidden_sizes = [(10,), (20,), (30,), (40,), (50,),
                (10, 10), (20, 20), (30, 30), (40, 40), (50, 50),
                (10, 10, 10), (20, 20, 20), (30, 30, 30), (40, 40, 40), (50, 50, 50)]

# hidden_sizes = [(10, 10)]
# start_position = hidden_sizes.index((50, 50))

for hidden_size in hidden_sizes:
    # Directory for the current architecture
    current_arch_dir = f'checkpoints/{hidden_size}_checkpoints'
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

        # reset model
        # main_net.reset_parameters()

        # Train main net
        training_losses_main, validation_losses_main = trainer.train_main_model(main_dataset, epochs = 200,lr_rate=0.05, pretrain=False)

        # Save info
        loss_list.append([training_losses_main['main_model'][-1], validation_losses_main['main_model'][-1]])
        rel_error_list.append(plotter.get_relative_error(main_net,main_dataset_test))

        # Save model parameters
        main_net.save_model(f'{hidden_size}_checkpoints/main_model_seed{idx}.pth')

    # write info to file
    with open(f'checkpoints/{hidden_size}_checkpoints/log_table.txt', 'w') as f:
        for i in range(len(seeds)):
            # Convert elements of rel_error_list[i] into strings and join them with comma
            rel_error_str = ','.join([str(elem) for elem in rel_error_list[i]])
            f.write(f'{loss_list[i][0]},{loss_list[i][1]},{rel_error_str}\n')

