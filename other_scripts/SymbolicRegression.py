import torch as T
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn import linear_model

dir_path = "C:\\Users\\clock\\Desktop\\Python\\Images\\statistics\\"
point = int(70)

# Load the values
load_dir = "stacks\\500_samples\\"
stack = T.load(dir_path+load_dir+"stack_train.pt") 
stack_test = T.load(dir_path+load_dir+"stack_test.pt") 
np_stack = stack.detach().numpy()
y_train = np.load(dir_path+load_dir+"training_targets.npy")
y_test = np.load(dir_path+load_dir+"test_targets.npy")
densities_targets = np.load(dir_path+load_dir+"densities_targets.npy")

A = 5  # We want figures to be A5
plt.figure(figsize=(46.82 * .5**(.5 * A), 33.11 * .5**(.5 * A)))

matplotlib.rc('xtick', labelsize=15) 
matplotlib.rc('ytick', labelsize=15)
matplotlib.rcParams.update({'font.size': 20}) 

stack_test = stack_test.detach().numpy()
for idx in range(20):
    plt.scatter(stack_test[:,idx,0], stack_test[:,idx,1])
# plt.scatter(stack_test[:,60,0], stack_test[:,60,1], label= np.array2string(densities_targets[60], formatter={'float_kind':lambda x: "%.2f" % x}))
# plt.scatter(stack_test[:,70,0], stack_test[:,70,1], label= np.array2string(densities_targets[70], formatter={'float_kind':lambda x: "%.2f" % x}))
# plt.scatter(stack_test[:,50,0], stack_test[:,50,1], label= np.array2string(densities_targets[50], formatter={'float_kind':lambda x: "%.2f" % x}))
# plt.scatter(stack_test[:,71,0], stack_test[:,71,1], label= np.array2string(densities_targets[71], formatter={'float_kind':lambda x: "%.2f" % x}))
plt.plot([0, 1], [0, 1], linestyle='--', color='k')
plt.xlabel('k1')
plt.ylabel('k2')
plt.title('k2 VS k1')
# plt.show()

# For some set of densities -> point
k1 = stack_test[:,point,0]
k2 = stack_test[:,point,1]

# regression = linear_model.LinearRegression()
# regression.fit(np.array(k1).reshape(-1,1),np.array(k2).reshape(-1,1))

# # The coefficients
# text_str = ("k2 = %.4f*k1 + %.4f" %(regression.coef_, regression.intercept_))
# predictions = regression.predict(np.array(k1).reshape(-1,1))
# # Plot outputs
# plt.clf()
# plt.scatter(k1, k2, color="black")
# plt.plot(k1, predictions, color="red", linewidth=3, label= text_str)
# plt.xlabel('k1')
# plt.ylabel('k2')
# plt.legend()
# # plt.show()
# plt.savefig("C:\\Users\\clock\\Desktop\\Python\\Images\\Symbolic_Regression\\linear_regression.png")

# Try linear regression using Pytorch ----------------------------------------------------------------------------
# Convert data to tensors
x_tensor = T.from_numpy(k1).float().reshape(-1,1)
y_tensor = T.from_numpy(k2).float().reshape(-1,1)



# convert the 3D array to a 2D array with shape (n*m, 3)
arr_2d = np_stack[:100,:,:].reshape(-1, 3)

# print the shapes of both arrays
print("Original array shape:", np_stack[100:,:,:].shape)
print("Reshaped array shape:", arr_2d.shape)

exit()

# Define model
class LinearRegression(T.nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = T.nn.Linear(1, 1)

    def forward(self, x):
        out = self.linear(x)
        return out

model = LinearRegression()

log_epoch = 10
lrn_rate = 0.01

# Define loss function and optimizer
criterion = T.nn.MSELoss()
optimizer = T.optim.Adam(model.parameters(), lr=lrn_rate)

# Train model
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(x_tensor)
    loss = criterion(outputs, y_tensor)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % log_epoch == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# Plot results
predicted = model(x_tensor).detach().numpy()
plt.clf()
plt.plot(k1, k2, 'ro', label='Original data')
plt.plot(k1, predicted, label='Fitted line')
plt.legend()
plt.xlabel('k1')
plt.ylabel('k2')
plt.show()

