import matplotlib.pyplot as plt
import os
import numpy as np
import torch

class PlottingTools:
    def __init__(self, species):
        self.species = species
        # Make images directory if it doesn't exist
        if not os.path.exists('images'):
            os.makedirs('images')

    def set_species(self, species):
        self.species = species

    def plot_loss_history(self, training_losses, validation_losses, filename=None):
        plt.clf()
        # if training_losses['surrogate_0'] exits, plot surrogate loss history
        if 'surrogate_0' in training_losses.keys():
            n_epochs = len(training_losses['surrogate_0'])
            for i in range(len(training_losses)):
                plt.figure(i)
                plt.plot(np.arange(1,n_epochs+1,1),training_losses[f'surrogate_{i}'],"-o", markersize=4, label='Training Loss')
                plt.plot(np.arange(1,n_epochs+1,1),validation_losses[f'surrogate_{i}'], "-o", markersize=4, label='Validation Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title(f'Surrogate {i} Loss History')
                plt.legend()
                if filename is None:
                    plt.savefig(f'images/surrogate_{i}_loss_history.png')
                else:
                    plt.savefig(os.path.join(filename))
        
        # if training_losses['main_model'] exits, plot main model loss history
        if 'main_model' in training_losses.keys():
            n_epochs = len(training_losses['main_model'])
            plt.figure(len(training_losses))
            # plt.plot(np.arange(1,n_epochs+1,1),training_losses['main_model'],"-o", markersize=4, label='Training Loss')
            x_range = np.arange(1, n_epochs+1, 1)
            plt.plot(x_range, training_losses['main_model'], "-o", markersize=4, label='Training Loss', color='blue')
            plt.plot(x_range[:int(n_epochs)], validation_losses['main_model'], "-o", markersize=4, label='Validation Loss' ,color='purple')
            # plt.plot(np.arange(n_epochs,n_epochs,1),validation_losses['main_model'], "-o", markersize=4, label='Validation Loss', color = 'orange')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            
            # plt.yscale('log')# log scale

            plt.title('Main Model Loss History')
            plt.legend()
            if filename is None:
                plt.savefig('images/main_model_loss_history.png')
            else:
                plt.savefig(filename)


    def plot_predictions_main(self, model, test_dataset, filename='predictions_vs_true_values.png'):
        model.eval()  # Switch to evaluation mode

        main_net = model.main_net
        surrogates = model.surrog_nets

        # Make sure the data is on the CPU
        test_data = test_dataset.x_data.to("cpu")
        test_targets = test_dataset.y_data.to("cpu")


        with torch.no_grad():  # Disable gradient calculation
            # original_data is of shape [n_conditions, batch_size, n_features]
            transposed_data = test_targets.transpose(0, 1)  # Now it's [batch_size, n_conditions, n_features]
            main_input = transposed_data.flatten(start_dim =1)  # Now it's [batch_size, n_conditions * n_features]            
            predictions_k = main_net(main_input)
            predictions_densities = []
            for surrogate in surrogates:
                predictions_densities.append(surrogate(predictions_k))

        # Convert tensors to numpy arrays
        predictions_k = predictions_k.numpy()
        predictions_densities = [prediction.numpy() for prediction in predictions_densities]
        true_values = test_targets.numpy()

        # -
        n_species = len(self.species)
        n_surrogates = len(predictions_densities)

        fig, axs = plt.subplots((n_species*n_surrogates)//5+1, 5, figsize=(20,15))
        axs = axs.flatten()
        colors = ['b', 'g', 'c', 'm', 'y', 'k', 'w']
        # Plots for predictions in densities
        for idx, prediction in enumerate(predictions_densities):
            # Plot for each species
            for i in range(n_species):  # changed the loop to iterate over range of species length
                ax = axs[idx * n_species + i]  # get corresponding axis 
                ax.scatter(true_values[idx,:,i], prediction[:, i], color= colors[idx])
                ax.set_xlabel('True Values')
                ax.set_ylabel('Predictions')
                # Add a diagonal line representing perfect agreement
                ax.plot([0, 1], [0, 1], linestyle='--', color='k')
                ax.set_title(f'{self.species[i]}')

                # Calculate relative error
                denominator = true_values[idx,:,i]
                denominator[np.abs(denominator) < 1e-9] = 1e-9  # Set small values to a small constant

                rel_err = np.abs(np.subtract(true_values[idx,:,i], prediction[:, i]) / denominator)


                textstr = '\n'.join((
                    r'$Mean\ \epsilon_{rel}=%.2f$%%' % (rel_err.mean() * 100,),
                    r'$Max\ \epsilon_{rel}=%.2f$%%' % (max(rel_err) * 100,)))

                # Colour point with max error
                max_index = np.argmax(rel_err)
                ax.scatter(true_values[idx,max_index, i], prediction[max_index, i], color="gold", zorder=2)

                # Define the text box properties
                props = dict(boxstyle='round', alpha=0.5)

                # Place a text box in upper left in axes coords
                ax.text(0.63, 0.25, textstr, fontsize=10, transform=ax.transAxes,
                        verticalalignment='top', bbox=props)

        plt.tight_layout()
        plt.savefig('images/' + filename)

        # PLot output of main model vs true values (test data)
        true_values = test_data.numpy()
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        for i, ax in enumerate(axs):
            ax.scatter(true_values[0,:, i], predictions_k[:, i], color = 'r')
            ax.set_xlabel('True Values')
            ax.set_ylabel('Predictions')
            # Add a diagonal line representing perfect agreement
            ax.plot([0, 1], [0, 1], linestyle='--', color='k')
            ax.set_title(f'True Values vs Predictions for k{i+1}')


            # Calculate relative error
            rel_err = np.abs(np.subtract(true_values[0,:, i], predictions_k[:, i]) / true_values[0,:, i])

            textstr = '\n'.join((
                r'$Mean\ \epsilon_{rel}=%.2f$%%' % (rel_err.mean() * 100,),
                r'$Max\ \epsilon_{rel}=%.2f$%%' % (max(rel_err) * 100,)))

            # Colour point with max error
            max_index = np.argmax(rel_err)
            ax.scatter(true_values[0,max_index, i], predictions_k[max_index, i], color="gold", zorder=2)

            # Define the text box properties
            props = dict(boxstyle='round', alpha=0.5)

            # Place a text box in upper left in axes coords
            ax.text(0.63, 0.25, textstr, fontsize=10, transform=ax.transAxes,
                    verticalalignment='top', bbox=props)
        
        # Color blue the first point for each pressure condition (true physical values)
        for i in range(3):
            axs[i].scatter(true_values[0,0, i], predictions_k[0, i], color = 'b')

        plt.tight_layout()
        plt.savefig('images/main_model_predictions_vs_true_values_ks.png')


    def plot_predictions_surrog(self, model, test_dataset, filename='predictions_vs_true_values_mainModel.png'):
        model.eval()  # Switch to evaluation mode

        # Make sure the data is on the CPU
        test_data = test_dataset[:][0].to("cpu")
        test_targets = test_dataset[:][1].to("cpu")

        with torch.no_grad():  # Disable gradient calculation
            predictions = model(test_data)

        # Convert tensors to numpy arrays
        predictions = predictions.numpy()
        true_values = test_targets.numpy()

        # Create 3x4 grid of subplots (it will have 2 empty subplots)
        fig, axs = plt.subplots(4, 3, figsize=(15,20))

        # Flatten the axis array to make iterating over it easier
        axs = axs.flatten()

        # Plot for each species
        for i in range(len(self.species)):
            axs[i].scatter(true_values[:, i], predictions[:, i])
            axs[i].set_xlabel('True Values')
            axs[i].set_ylabel('Predictions')
            # Add a diagonal line representing perfect agreement
            axs[i].plot([0, 1], [0, 1], linestyle='--', color='k')
            axs[i].set_title(f'True Values vs Predictions for {self.species[i]}')


            # Calculate relative error
            denominator = true_values[:,i]
            denominator[np.abs(denominator) < 1e-9] = 1e-9  # Set small values to a small constant
            rel_err = np.abs(np.subtract(true_values[:,i], predictions[:, i]) / denominator)


            textstr = '\n'.join((
            r'$Mean\ \epsilon_{rel}=%.2f$%%' % (rel_err.mean()*100, ),
            r'$Max\ \epsilon_{rel}=%.2f$%%' % (max(rel_err)*100, )))

            # colour point o max error
            max_index = np.argmax(rel_err)
            axs[i].scatter(true_values[max_index,i],predictions[max_index,i] , color="gold", zorder= 2)

            # these are matplotlib.patch.Patch properties
            props = dict(boxstyle='round', alpha=0.5) #, facecolor='none', edgecolor='none')

            # place a text box in upper left in axes coords
            axs[i].text(0.63, 0.25, textstr, fontsize=10,  transform=axs[i].transAxes,
                verticalalignment='top', bbox=props)

        # Remove the extra subplots
        for i in range(len(self.species), len(axs)):
            fig.delaxes(axs[i])

        plt.tight_layout()
        plt.savefig('images/' + filename)



    def get_relative_error(self, neural_net, test_dataset):
        main_net = neural_net
        main_net.eval()  # Switch to evaluation mode

        # Make sure the data is on the CPU
        test_data = test_dataset.x_data.to("cpu")
        test_targets = test_dataset.y_data.to("cpu")

        with torch.no_grad():  # Disable gradient calculation
            # original_data is of shape [n_conditions, batch_size, n_features]
            transposed_data = test_targets.transpose(0, 1)  # Now it's [batch_size, n_conditions, n_features]
            main_input = transposed_data.flatten(start_dim =1)  # Now it's [batch_size, n_conditions * n_features]            
            predictions_k = main_net(main_input)

        # Convert tensors to numpy arrays
        predictions_k = predictions_k.numpy()
        true_values = true_values = test_data.numpy()

        rel_errors = []
        for i in range(len(predictions_k[0])):
            # Calculate relative error
            rel_err = np.abs(np.subtract(true_values[0,:, i], predictions_k[:, i]) / true_values[0,:, i])
            rel_errors.append(rel_err.mean() * 100)

        return rel_errors


if __name__ == "__main__":
    def readFile(file_address):
        with open(file_address, 'r') as file :
            densities=[]
            for line in file:
                if line.startswith(' '):
                    densities.append(line.split()[0])
        return densities
    
    densities = readFile('C:\\Users\\clock\\Desktop\\LoKI_v3.1.0\\Code\\Output\\oxygen_novib_0\\chemFinalDensities.txt')
    print(densities)