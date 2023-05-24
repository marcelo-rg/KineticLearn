import matplotlib.pyplot as plt
import os

class PlottingTools:
    def plot_loss_history(self, training_losses, validation_losses):
        # Make images directory if it doesn't exist
        if not os.path.exists('images'):
            os.makedirs('images')

        for i in range(len(training_losses)):
            plt.figure(i)
            plt.plot(training_losses[f'surrogate_{i}'], label='Training Loss')
            plt.plot(validation_losses[f'surrogate_{i}'], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Surrogate {i} Loss History')
            plt.legend()
            plt.savefig(f'images/surrogate_{i}_loss_history.png')
