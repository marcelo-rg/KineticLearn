import matplotlib.pyplot as plt
import os
import numpy as np
import torch

class PlottingTools:
    def plot_loss_history(self, training_losses, validation_losses):
        # Make images directory if it doesn't exist
        if not os.path.exists('images'):
            os.makedirs('images')

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


    def plot_predictions(self, model, test_dataset):
        model.eval()  # Switch to evaluation mode

        # Make sure the data is on the CPU
        test_data = test_dataset[:][0].to("cpu")
        test_targets = test_dataset[:][1].to("cpu")

        with torch.no_grad():  # Disable gradient calculation
            predictions = model(test_data)

        # Convert tensors to numpy arrays
        predictions = predictions.numpy()
        true_values = test_targets.numpy()

        # Plot for each species
        species = ['O2(X)','O2(a)', 'O(3P)']
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
        plt.savefig('images/predictions_vs_true_values.png')
