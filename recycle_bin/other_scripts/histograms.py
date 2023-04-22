import matplotlib.pyplot as plt
import numpy as np
import random

k = np.array([6E-16,1.3E-15,9.6E-16,2.2E-15,7E-22,3E-44,3.2E-45,5.2,53]) # total of 9 reactions
n_react = k.size
n_trainSet = 1000
randOn = False

with open('C:\\Users\\clock\\Desktop\\Python\\datapoints.txt', 'r') as file :
    filedata = file.readlines()


dataset=[]
for line in filedata:
    dataset.append([float(item) for item in line.split()])


X = [item[:9] for item in dataset]
Y = [item[-3:] for item in dataset]



def uni_dist_k_set():
    k1 = np.linspace(k[0]*0.1, k[0]*10, n_trainSet)
    k_previous = k1

    for i in range(1, k.size, 1): #range(start, stop, step)
        ki = np.linspace(k[i]*0.1, k[i]*10, n_trainSet)   

        k_set =  np.vstack((k_previous,ki))
        k_previous = k_set

    if(randOn):
        for _ in k_set:
            random.shuffle(_)

    return np.transpose(k_set)

k_set = uni_dist_k_set()

plt.hist(k_set[:,0], bins=60)
plt.show()