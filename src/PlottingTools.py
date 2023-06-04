import matplotlib.pyplot as plt
import os
import numpy as np
import torch

class PlottingTools:
    def __init__(self):
        # Make images directory if it doesn't exist
        if not os.path.exists('images'):
            os.makedirs('images')

    def plot_loss_history(self, training_losses, validation_losses):
        plt.clf()
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
                plt.savefig(f'images/surrogate_{i}_loss_history.png')
        
        # if training_losses['main_model'] exits, plot main model loss history
        if 'main_model' in training_losses.keys():
            n_epochs = len(training_losses['main_model'])
            plt.figure(len(training_losses))
            plt.plot(np.arange(1,n_epochs+1,1),training_losses['main_model'],"-o", markersize=4, label='Training Loss')
            plt.plot(np.arange(1,n_epochs+1,1),validation_losses['main_model'], "-o", markersize=4, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Main Model Loss History')
            plt.legend()
            plt.savefig('images/main_model_loss_history.png')


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

        species = ['O2(X)', 'O2(a)', 'O(3P)']
        fig, axs = plt.subplots(len(predictions_densities), 3, figsize=(15, 7))
        axs = np.atleast_2d(axs) # Make sure axs is 2D
        colors = ['b', 'g']
        # For each surrogate model
        for idx, prediction in enumerate(predictions_densities):
            # Plot for each species
            for i, ax in enumerate(axs[idx]): # if len(predictions_densities) = 1, Error: axs[idx] is not iterable
                ax.scatter(true_values[idx,:,i], prediction[:, i], color= colors[idx])
                ax.set_xlabel('True Values')
                ax.set_ylabel('Predictions')
                # Add a diagonal line representing perfect agreement
                ax.plot([0, 1], [0, 1], linestyle='--', color='k')
                ax.set_title(f'True Values vs Predictions for {species[i]}')

                # Calculate relative error
                rel_err = np.abs(np.subtract(true_values[idx,:,i], prediction[:, i]) / true_values[idx,:,i])

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
        model.eval()  # Switch to evaluation mode-

        # Make sure the data is on the CPU
        test_data = test_dataset[:][0].to("cpu")
        test_targets = test_dataset[:][1].to("cpu")

        with torch.no_grad():  # Disable gradient calculation
            predictions = model(test_data)

        # Convert tensors to numpy arrays
        predictions = predictions.numpy()
        true_values = test_targets.numpy()

        # Plot for each species
        species = ['O2(X)','O2(a)','O(3P)']
        fig, axs = plt.subplots(1, 3, figsize=(15,5))
        for i, ax in enumerate(axs):
            ax.scatter(true_values[:, i], predictions[:, i])
            ax.set_xlabel('True Values')
            ax.set_ylabel('Predictions')
            # Add a diagonal line representing perfect agreement
            ax.plot([0, 1], [0, 1], linestyle='--', color='k')
            ax.set_title(f'True Values vs Predictions for {species[i]}')

            # Calculate relative error
            rel_err = np.abs(np.subtract(true_values[:, i], predictions[:, i])/true_values[:, i])

            textstr = '\n'.join((
            r'$Mean\ \epsilon_{rel}=%.2f$%%' % (rel_err.mean()*100, ),
            r'$Max\ \epsilon_{rel}=%.2f$%%' % (max(rel_err)*100, )))

            # colour point o max error
            max_index = np.argmax(rel_err)
            ax.scatter(true_values[max_index,i],predictions[max_index,i] , color="gold", zorder= 2)

            # these are matplotlib.patch.Patch properties
            props = dict(boxstyle='round', alpha=0.5) #, facecolor='none', edgecolor='none')

            # place a text box in upper left in axes coords
            ax.text(0.63, 0.25, textstr, fontsize=10,  transform=ax.transAxes,
                verticalalignment='top', bbox=props)

        plt.tight_layout()
        plt.savefig('images/' + filename)



if __name__ == "__main__":
    # array of shape (2,5,3)
    array1 = np.array([[[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]], [[16,17,18],[19,20,21],[22,23,24],[25,26,27],[28,29,30]]])
    print(array1.shape)
    array1 = array1.reshape(-1,6)
    print(array1.shape)
    print(array1)

    # array of shape (5,2,3)
    array2 = np.array([[[1,2,3],[4,5,6]], [[7,8,9],[10,11,12]], [[13,14,15],[16,17,18]], [[19,20,21],[22,23,24]], [[25,26,27],[28,29,30]]]) 
    print(array2.shape)
    array2 = torch.Tensor(array2).flatten(start_dim=1)
    array2 = array2.numpy()
    print(array2.shape)
    print(array2)