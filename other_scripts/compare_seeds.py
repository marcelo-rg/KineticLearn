import numpy as np
import matplotlib.pyplot as plt
import matplotlib

seed_1 =  np.loadtxt("seed_1.txt")
seed_2 = np.loadtxt("seed_2.txt")

# Configure matplotlib settings
A = 5 
plt.figure(figsize=(46.82 * .5**(.5 * A), 33.11 * .5**(.5 * A)))

matplotlib.rc('xtick', labelsize=15) 
matplotlib.rc('ytick', labelsize=15)
matplotlib.rcParams.update({'font.size': 20}) 

# Create a scatter plot of the two arrays against each other
for idx in range(len(seed_1[0])):
    filename = 'C:\\Users\\clock\\Desktop\\Python\\Images\\seeds\\seed_k_' + str(idx+1)+'.png'
    plt.clf()
    plt.scatter(seed_1[:,idx], seed_2[:,idx])
    # Add labels and a title
    plt.xlabel('k seed_1')
    plt.ylabel('k seed_2')
    plt.title('k'+str(idx+1))
    # Add a diagonal line representing perfect agreement
    plt.plot([0, 1], [0, 1], linestyle='--', color='k')
    plt.savefig(filename)





