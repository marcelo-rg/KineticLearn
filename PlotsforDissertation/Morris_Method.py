from  genFiles_O2_simple import Parameters
import sys, os
import numpy as np
import random

# Disable print
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore print
def enablePrint():
    sys.stdout = sys.__stdout__

n_points = 12
param = Parameters(n_points)


k_colums = [0,1] 
k_true_values = [7.6e-22, 3E-44, 4e-20]
k_true_values = [1., 1.]
sample_range = [0, 1.]


# New start from zero morris sampling
def MorrisSampler(k_real, p, r, k_range, indexes):
    k_size = len(k_real)

    # Define the region of experimentation w
    w = np.linspace(0, 1, p)
    
    # Define the incrmement delta
    delta = p/(2*(p-1))

    # Create starting nodes
    start_nodes = []
    for i in range(r):
        start_nodes.append(random.choices(w,k= k_size)) # maybe use random.sample instead to avoid duplicates
    # print("start_nodes: ", start_nodes)
    # print("r: ", r)

    trajectories = []
    mean_point = 0.5 
    for (traj_idx, start_node) in enumerate(start_nodes):
        trajectory= []
        trajectory.append(start_node)                  # add starting node
        order = random.sample(range(0,k_size), k_size) # generate updating order

        # add the remaining nodes
        current_node = start_node.copy()
        for i in order:
            new_node = current_node.copy()

            if(new_node[i]/mean_point>1): # if it lies in the second half of the range interval
                new_node[i] = new_node[i]-delta
            else:
                new_node[i] = new_node[i]+delta

            trajectory.append(new_node)
            current_node = new_node
        
        # save current trajectory
        trajectories.append(trajectory)
    print("r: ", r, "p: ", p, "k_size: ", k_size)
    print("trajectories shape: ", np.array(trajectories).shape)
    # print("trajectories: ", trajectories)

    reshaped_traj =  np.reshape(trajectories, (r*(k_size+1),-1))  # r * (k+1) sets of inputs
    print("trajectories shape: ", np.array(reshaped_traj).shape)
    
    n_inputs = r * (k_size+1)
    print("n_inputs: ", n_inputs)

    # Transform to the real distribution of k values
    k_set=[]
    for item in k_real:
        k_set.append(np.full(n_inputs, item))

    for (i, k_idx) in enumerate(indexes):
        k_set[k_idx] =  reshaped_traj[:,i]* k_real[k_idx]

    # print(np.transpose(np.array(k_set))[:,:3])
    return np.transpose(k_set)

# End of new start from zero morris sampling

uniform_sampled = param.random_kset(k_true_values, k_colums, krange= sample_range, pdf_function='uniform')
# log_sampled = param.random_kset(k_true_values, k_colums, krange= sample_range, pdf_function='log')
lh_sampled = param.latin_hypercube_kset(k_true_values, k_colums, krange= sample_range)


blockPrint()
# mymorris = MorrisSampler(k_true_values, p =10 , r = int(n_points/3), k_range= sample_range, indexes= k_colums)
morris_sampled = param.morris_kset(k_true_values, p =10 , r = int(n_points/3), k_range_type='lin', k_range= sample_range, kcolumns= k_colums)
enablePrint()

# print(morris_sampled.shape) # (2000, 3)
# print(mymorris)
# print(morris_sampled)


# Plot the results in 3d space
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# fig = plt.figure(figsize=(10,10), facecolor='#cfe2f3')
# ax = fig.add_subplot(111, projection='3d', facecolor='#cfe2f3')
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)


# plt.title("Sampling visualization")
# ax.scatter(uniform_sampled[:,0], uniform_sampled[:,1], uniform_sampled[:,2], c='b', marker='^', label='Uniform sampling')
# ax.scatter(log_sampled[:,0], log_sampled[:,1], log_sampled[:,2], c='r', marker='o', label='Log sampling')
# ax.scatter(lh_sampled[:,0], lh_sampled[:,1], lh_sampled[:,2], c='g', marker='s', label='Latin hypercube sampling')
# ax.scatter(morris_sampled[:,0], morris_sampled[:,1], morris_sampled[:,2], c='k', marker='x', label='Morris sampling')

# ax.scatter(log_sampled[:,0], log_sampled[:,1], c='r', marker='o', label='Log sampling')
# ax.scatter(lh_sampled[:,0], lh_sampled[:,1], c='g', marker='s', label='Latin hypercube sampling')
# ax.scatter(morris_sampled[:,0], morris_sampled[:,1], c='k', marker='x', label='Morris sampling')

# matplotlib built-in color cycle
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#FF8C00', '#8B4513', '#008080']  # add more if needed

# Reshape the array into shape (group, point, dimension)
print (morris_sampled.shape)
morris_sampled_grouped = np.reshape(morris_sampled, (-1, 3, 2))
print (morris_sampled_grouped.shape)
# exit()

for i in range(len(morris_sampled_grouped)):
    ax.scatter(morris_sampled_grouped[i, :, 0], morris_sampled_grouped[i, :, 1],
               color=colors[i % len(colors)], label=f'Traject. {i+1}')
    # ax.plot(morris_sampled_grouped[i, :, 0], morris_sampled_grouped[i, :, 1], 
    #         color=colors[i % len(colors)])

ax.set_xlabel(r'$k_{1}$', fontsize=16)
ax.set_ylabel(r'$k_{2}$', fontsize=16)
# ax.set_zlabel(r'$k_{3}$', fontsize=16, rotation= 90)

# Set value for p
p = 10

# Define the values for omega
omega = np.linspace(0, 1, p+1)

# Plotting the horizontal and vertical dashed lines
for value in omega:
    plt.axhline(y=value, color='gray', linestyle='--', linewidth=0.7)
    plt.axvline(x=value, color='gray', linestyle='--', linewidth=0.7)

plt.legend(loc='upper left')
plt.show()
