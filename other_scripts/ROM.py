import torch as T
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA


dir_path = "C:\\Users\\clock\\Desktop\\Python\\Images\\statistics\\"
point = int(35)

# Load the values
load_dir = "stacks\\500_samples\\"
stack = T.load(dir_path+load_dir+"stack_train.pt") 
stack_test = T.load(dir_path+load_dir+"stack_test.pt") 
np_stack = stack_test.detach().numpy()
y_train = np.load(dir_path+load_dir+"training_targets.npy")
y_test = np.load(dir_path+load_dir+"test_targets.npy")
densities_targets = np.load(dir_path+load_dir+"densities_targets.npy")

# Get our matrix of k's
k_matrix = np_stack.mean(axis= 0)
print(np.shape(k_matrix))
# Get densities data
densities = densities_targets

# Do PCA analysis as unsupervised learning
pca = PCA(n_components = 2)
principalComponents = pca.fit_transform(densities)
# print(pca.singular_values_)
print("Explained variance ratios: ",pca.explained_variance_ratio_)

# Visualize k's as function projections/PC's  
fig = plt.figure(figsize = (8,8))
ax = plt.axes(projection ="3d")
# ax = ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_zlabel('k values', fontsize = 15)
# ax.set_title('2 component PCA', fontsize = 20)

labels = ['k1', 'k2', 'k3']
colors = ['r', 'g', 'b']
my_cmap = matplotlib.cm.jet
# my_cmap = plt.get_cmap('hsv')

for index, ki in enumerate(k_matrix.T):
    sctt= ax.scatter3D(principalComponents[:,0],principalComponents[:,1], ki,s=20,c= colors[index], marker = 'o')

ax.legend(labels)
# ax.set_xlabel('O2(x)', fontsize=15)
# ax.set_ylabel('O2(a)', fontsize=15)
# ax.set_zlabel('O(3P)', fontsize=15)
# fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 5)
ax.grid()
plt.show()