import torch as T
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import gplearn


dir_path = "C:\\Users\\clock\\Desktop\\Python\\Images\\statistics\\"
point = int(70)
T.manual_seed(10)

# Load the values
load_dir = "stacks\\500_samples\\"
stack = T.load(dir_path+load_dir+"stack_train.pt") 
stack_test = T.load(dir_path+load_dir+"stack_test.pt") 
np_stack = stack_test.detach().numpy()
y_train = np.load(dir_path+load_dir+"training_targets.npy")
y_test = np.load(dir_path+load_dir+"test_targets.npy")
densities_targets = np.load(dir_path+load_dir+"densities_targets.npy")

stack_test = stack_test.detach().numpy()
A = 5  # We want figures to be A5
plt.figure(figsize=(46.82 * .5**(.5 * A), 33.11 * .5**(.5 * A)))

matplotlib.rc('xtick', labelsize=15) 
matplotlib.rc('ytick', labelsize=15)
matplotlib.rcParams.update({'font.size': 20}) 

# For some set of densities -> point
k1 = stack_test[:,point,0]
k2 = stack_test[:,point,1]

# Try linear regression using Pytorch ----------------------------------------------------------------------------
# Convert data to tensors
x_tensor = T.from_numpy(k1).float().reshape(-1,1)
y_tensor = T.from_numpy(k2).float().reshape(-1,1)
print(np.shape(x_tensor))
# exit()

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
lrn_rate = 0.1

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
# plt.show()

from gplearn.genetic import SymbolicRegressor

# Prepare the data
# X = np.linspace(-1, 1, 100).reshape(-1, 1)
# y = X**2 + np.random.normal(0, 0.1, X.shape)

X = k1.reshape(-1,1)
constant = np.ones((500, 1))
X_with_constant = np.hstack((X, constant))
# X = X_with_constant
y = k2.reshape(-1,1)


# Create the estimator
est_gp = SymbolicRegressor(population_size=1000, const_range= (-5,5), parsimony_coefficient= 0.02,
                           generations=11, stopping_criteria=0.007,
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           max_samples=0.9, verbose=1,
                           random_state=0,
                           function_set=('add','mul'))

# Fit the estimator to the data
est_gp.fit(X_with_constant, y)

# Retrieve the best symbolic expression
best_program = est_gp._program

# Print the expression as a string
print(str(best_program))

# Evaluate the estimator
y_pred = est_gp.predict(X_with_constant)

# Plot the results
plt.clf()
plt.scatter(X, y, s=20)
# plt.plot(X, y_pred, color='r', linewidth =0.5)
plt.scatter(X, y_pred, color='r', s= 10)

plt.show()


