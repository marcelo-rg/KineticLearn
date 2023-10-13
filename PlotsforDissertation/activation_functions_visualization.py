import numpy as np
import matplotlib.pyplot as plt

# Define the functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

# Define the derivatives
def derivative_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def derivative_tanh(x):
    return 1.0 - tanh(x)**2

def derivative_relu(x):
    return np.where(x > 0, 1.0, 0.0)

# Create the x values
x = np.linspace(-4.5, 4.5, 1000)

# Create the plots
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# set a grid 
ax[0].grid()
ax[1].grid()

# set the y domain limits
ax[0].set_ylim([-1.2, 1.2])

# set only 3 ticks to the y 
ax[0].set_yticks([-1., -0.5, 0, 0.5, 1.])

# Add a box around the entire figure
fig.patch.set_edgecolor('black')  
fig.patch.set_linewidth(2)


# Plot the functions on the left plot
ax[0].plot(x, sigmoid(x), label='Sigmoid', color='green', linestyle='--')
ax[0].plot(x, tanh(x), label='Tanh', color='blue', linestyle='-')
ax[0].plot(x, relu(x), label='ReLU', color='red', linestyle='-.')
ax[0].set_title("Activation Functions")
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')
ax[0].legend()

# Plot the derivatives on the right plot
ax[1].plot(x, derivative_sigmoid(x), label='Derivative of Sigmoid', color='green', linestyle='--')
ax[1].plot(x, derivative_tanh(x), label='Derivative of Tanh', color='blue', linestyle='-')
ax[1].plot(x, derivative_relu(x), label='Derivative of ReLU', color='red', linestyle='-.')
ax[1].set_title("Derivatives of Activation Functions")
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')
ax[1].legend(loc= 'upper left')

plt.tight_layout()
plt.show()
