from gplearn.genetic import SymbolicRegressor
import numpy as np

# Prepare the data
X = np.linspace(-1, 1, 100).reshape(-1, 1)
constant = np.ones((100, 1))
X_with_constant = np.hstack((X, constant))
y = X**2 + np.random.normal(0, 0.1, X.shape)

# Create the estimator
est_gp = SymbolicRegressor(population_size=5000,
                           generations=20, stopping_criteria=0.01,
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           max_samples=0.9, verbose=1,
                           parsimony_coefficient=0.01, random_state=0)

# Fit the estimator to the data
est_gp.fit(X, y)

# Retrieve the best symbolic expression

best_program = est_gp._program

# Print the expression as a string
print(str(best_program))
