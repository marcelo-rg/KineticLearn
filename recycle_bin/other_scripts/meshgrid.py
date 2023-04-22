import numpy as np
import matplotlib.pyplot as plt
import random

k = np.array([6E-16,1.3E-15,9.6E-16,2.2E-15,7E-22,3E-44,3.2E-45,5.2,53]) # total of 9 reactions
randOn= False
n_trainSet = 10

#-----------------------------------------------------------------

def uni_dist_k_set_mesh(index1, index2):
    uni_distr = np.linspace(1, 10, n_trainSet)
    
    list=[]
    for item in k:
        list.append(np.full(n_trainSet**2, item ))   
    
    array = np.array(list) # pass to array

    k1 = uni_distr*k[index1]
    k2 =  uni_distr*k[index2]

    K1, K2 = np.meshgrid(k1,k2)

    array[index1] = K1.flatten()
    array[index2] = K2.flatten()

    if(randOn):
        for _ in array:
            random.shuffle(_)

    return np.transpose(array)

#-------------------------Main-------------------------------------------

# npoints = 3

# k1 = np.linspace(1,10, npoints)
# k2 = np.linspace(1,5, npoints)
# k3 = np.linspace(1,10, npoints)

# K1, K2 = np.meshgrid(k1, k2)



# k1 = K1.flatten()
# k2 = K2.flatten()

# k1_shuffled = k1.copy()
# #k3 = K3.flatten()
# random.shuffle(k1_shuffled)

# for index in range(len(k1)):
#     print( "%.2f  %.2f %.2f" %(k1[index] ,k2[index], k1_shuffled[index]))

# print(len(k1), " datapoints")

#-----------------------------------------------------------------

index1 = 0
index2 = 2

k_set = np.transpose(uni_dist_k_set_mesh(index1, index2))
for index in range(len(k_set[index1])):
    print((k_set[index1][index],k_set[index2][index]))


