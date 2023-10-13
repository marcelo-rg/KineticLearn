import numpy as np
import matplotlib.pyplot as plt

# folder_path = "D:\\Marcelo\\github\\Dissertation\\Images\\"

# Define the range for x values
x = np.linspace(1, 20, 1000)
x_limited = np.linspace(1, 5, 500)


c_modified = 10  # For the modified training curve

# Define the modified training curve
training_curve_limited = c_modified / x_limited**3

# Define the validation curve with c=0.1
c_linear = 0.08
validation_curve_linear_updated = training_curve_limited + c_linear * x_limited**2


# Find the minimum value of the validation curve
min_val = np.min(validation_curve_linear_updated)

# Find the x-coordinate (epoch) corresponding to the minimum value of the validation curve
min_val_index = np.argmin(validation_curve_linear_updated)
min_val_epoch = x_limited[min_val_index]

# Plotting code
plt.figure(figsize=(10,5))
plt.plot((1, 5), (min_val, min_val), 'k-',linestyle='--', color='dimgrey')
plt.plot(x_limited, training_curve_limited, linestyle='-.', label='Training Loss', color='blue')
plt.plot(x_limited, validation_curve_linear_updated, label='Validation Loss', color='red')
# plt.title('Modified Training and Updated Validation Loss Curves (c=0.1)')

# Add vertical arrow and text annotation with larger font
plt.annotate('Optimal Model', 
             xy=(min_val_epoch, min_val), 
             xytext=(min_val_epoch, min_val+3.5), 
             arrowprops=dict(facecolor='black', arrowstyle='->', mutation_scale=15),
             fontsize=15,
             ha='center')


# Adjust x-axis ticks to match the desired epoch range
desired_ticks = np.linspace(0, 200, 6)  # 0, 40, 80, ... 200
current_range = [1, 5]
scaled_ticks = np.linspace(current_range[0], current_range[1], len(desired_ticks))

plt.xticks(scaled_ticks, desired_ticks.astype(int), fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend(fontsize=14)
plt.show()
# plt.savefig(folder_path +"early_stopping.pdf")
