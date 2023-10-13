import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR

# Seed for reproducibility
np.random.seed(42)

# Create an artificial noisy dataset based on a non-linear function
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.15, X.shape[0])

# Fit regression models
svr_lin = SVR(kernel='linear', C=100, epsilon=0.1)
svr_rbf = SVR(kernel='rbf', C=1e2, gamma=0.1, epsilon=0.1)
svr_rbf_lowC = SVR(kernel='rbf', C=0.1, gamma=0.1, epsilon=0.1)

y_lin = svr_lin.fit(X, y).predict(X)
y_rbf = svr_rbf.fit(X, y).predict(X)
y_rbf_lowC = svr_rbf_lowC.fit(X, y).predict(X)

# Define the epsilon-tube margins
y_lin_upper = y_lin + svr_lin.epsilon
y_lin_lower = y_lin - svr_lin.epsilon

y_rbf_upper = y_rbf + svr_rbf.epsilon
y_rbf_lower = y_rbf - svr_rbf.epsilon

y_rbf_lowC_upper = y_rbf_lowC + svr_rbf_lowC.epsilon
y_rbf_lowC_lower = y_rbf_lowC - svr_rbf_lowC.epsilon

# Plotting
plt.figure(figsize=(12, 6))

# Linear Kernel
plt.subplot(1, 2, 1)
plt.scatter(X, y, color='darkorange', label='data')
plt.plot(X, y_lin, color='navy', lw=2, label=r'Linear model (C=100, $\epsilon = 0.1$)')
plt.plot(X, y_lin_upper, 'k--',  label='margin')
plt.plot(X, y_lin_lower, 'k--')
plt.scatter(X[svr_lin.support_], y[svr_lin.support_], facecolor="none", edgecolor="k", s=100)
plt.xlabel('X', fontdict={'fontsize': 12})
plt.ylabel('Y', fontdict={'fontsize': 12})
plt.title('Support Vector Regression with Linear Kernel')
plt.legend(fontsize = 11)

# RBF Kernel
plt.subplot(1, 2, 2)
plt.scatter(X, y, color='darkorange', label='data')
plt.plot(X, y_rbf, color='navy', lw=2, label= r'RBF model (C=100, $\epsilon$ = 0.1)')
plt.plot(X, y_rbf_upper, 'k--')
plt.plot(X, y_rbf_lower, 'k--')
plt.plot(X, y_rbf_lowC, color='c', lw=2, label=r'RBF model (C=0.1, $\epsilon$ = 0.1)')
plt.plot(X, y_rbf_lowC_upper, 'c--')
plt.plot(X, y_rbf_lowC_lower, 'c--')
plt.scatter(X[svr_rbf.support_], y[svr_rbf.support_], facecolor="none", edgecolor="k", s=100)
# plt.scatter(X[svr_rbf_lowC.support_], y[svr_rbf_lowC.support_], facecolor="none", edgecolor="c", s=100)
plt.xlabel('X', fontsize=12)
plt.ylabel('Y', fontsize=12)
plt.title('Support Vector Regression with RBF Kernel')
plt.legend(fontsize=11)

# increase font size
plt.rcParams.update({'font.size': 16})
# increase legend size
plt.rcParams["legend.fontsize"] = 20

# Adjust the subplot layout

plt.tight_layout()
plt.show()
