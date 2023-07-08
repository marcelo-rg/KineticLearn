import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit, logit

def softplus(x):
    return np.log1p(np.exp(x))

def sigmoid(x):
    return expit(x)

x = np.linspace(-10, 10, 1000)
y1 = sigmoid(softplus(x))
y2 = sigmoid(x)

plt.figure(figsize=(10, 6))

# Plot sigmoid(softplus(x))
plt.plot(x, y1, label='y = sigmoid(softplus(x))')

# Plot sigmoid(x)
plt.plot(x, y2, label='y = sigmoid(x)', linestyle='--')

plt.title('Comparison of y = sigmoid(softplus(x)) and y = sigmoid(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
