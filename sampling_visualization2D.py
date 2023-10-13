from  genFiles_O2_simple import Parameters
import sys, os
import numpy as np

# Disable print
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore print
def enablePrint():
    sys.stdout = sys.__stdout__

n_points = 100
param = Parameters(n_points)


k_colums = [0,1] 
k_true_values = [7.6e-22, 3E-44]
k_true_values = [1., 1.]
sample_range = [0.5, 2]

uniform_sampled = param.random_kset(k_true_values, k_colums, krange= sample_range, pdf_function='uniform')
log_sampled = param.random_kset(k_true_values, k_colums, krange= sample_range, pdf_function='log')
lh_sampled = param.latin_hypercube_kset(k_true_values, k_colums, krange= sample_range, log_uniform= True)


blockPrint()
morris_sampled = param.morris_kset(k_true_values, p =10 , r = int(n_points/4), k_range_type='lin', k_range= sample_range, kcolumns= k_colums)
enablePrint()

# print(morris_sampled.shape) # (2000, 3)

# Plot the results in 3d space
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig  = plt.figure(figsize=(10,10), facecolor='#cfe2f3')
# ax = fig.add_subplot(111, projection='3d', facecolor='#cfe2f3')

# plt.title("Sampling visualization")
# ax.scatter(uniform_sampled[:,0], uniform_sampled[:,1], uniform_sampled[:,2], c='b', marker='^', label='Uniform sampling')
plt.scatter(log_sampled[:,0], log_sampled[:,1], c='r', marker='o', label='Log sampling')
plt.scatter(lh_sampled[:,0], lh_sampled[:,1], c='g', marker='s', label='Latin hypercube sampling')
# ax.scatter(morris_sampled[:,0], morris_sampled[:,1], morris_sampled[:,2], c='k', marker='x', label='Morris sampling')
"""
# matplotlib built-in color cycle
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#FF8C00', '#8B4513', '#008080']  # add more if needed

# Reshape the array into shape (group, point, dimension)
morris_sampled_grouped = np.reshape(morris_sampled, (-1, 4, 3))

for i in range(len(morris_sampled_grouped)):
    ax.scatter(morris_sampled_grouped[i, :, 0], morris_sampled_grouped[i, :, 1], morris_sampled_grouped[i, :, 2], 
               color=colors[i % len(colors)], label=f'Traject. {i+1}')
    ax.plot(morris_sampled_grouped[i, :, 0], morris_sampled_grouped[i, :, 1], morris_sampled_grouped[i, :, 2], 
            color=colors[i % len(colors)])
"""
plt.xlabel(r'$k_{1}$', fontsize=16)
plt.ylabel(r'$k_{2}$', fontsize=16)
# ax.set_zlabel(r'$k_{3}$', fontsize=16, rotation= 90)

plt.legend(loc='upper left')
plt.show()
