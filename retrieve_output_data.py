import numpy as np
import matplotlib.pyplot as plt
import random
random.seed(10)

#-----------------------------------------------------------------------------------------------------------------
# Generate k values (again, same k values used in simulation)
k = np.array([6E-16,1.3E-15,9.6E-16,2.2E-15,7E-22,3E-44,3.2E-45,5.2,53]) # total of 9 reactions
n_react = k.size
n_trainSet = 100
randOn = True

def log_dist_k_set():

    log_distr = np.logspace(-1, 1, n_trainSet, base = 10.) # 2 orders of magnitude [0.1-10]

    k1 = log_distr*k[0]
    k_previous = k1

    for i in range(1, k.size, 1): #range(start, stop, step)
        ki = log_distr*k[i]  
        k_set =  np.vstack((k_previous,ki))
        k_previous = k_set

    if(randOn):
        for _ in k_set:
            random.shuffle(_)

    return np.transpose(k_set)

#------------------------------------------------------------

def uni_dist_k_set(index1, index2):
    uni_distr = np.linspace(0.1, 10, n_trainSet)
    #log_distr = np.logspace(-1, 1, n_trainSet, base = 10.) # 2 orders of magnitude [0.1-10]
    
    list=[]
    for item in k:
        list.append(np.full(n_trainSet, item ))   
    
    array = np.array(list) # pass to array

    array[index1] = uni_distr*k[index1]
    array[index2]  =  uni_distr*k[index2]
    #array[index3]  =  uni_distr*k[index3]

    # k1 = uni_distr*k[0]
    # k_previous = k1

    # for i in range(1, k.size, 1): #range(start, stop, step)
    #     ki = uni_distr*k[i]  
    #     k_set =  np.vstack((k_previous,ki))
    #     k_previous = k_set

    if(randOn):
        for _ in array:
            random.shuffle(_)

    return np.transpose(array)


# ---------------------------------------------------------------------------------------------------------------

# k_set0 = log_dist_k_set(0)
# k_set1 = log_dist_k_set(1)
# k_set2 = log_dist_k_set(2)
# k_set3 = log_dist_k_set(3)
# k_set4 = log_dist_k_set(4)
# k_set5 = log_dist_k_set(5)
# k_set6 = log_dist_k_set(6)
# k_set7 = log_dist_k_set(7)
# k_set8 = log_dist_k_set(8)

# k_set = np.concatenate((k_set0,k_set1,k_set2, k_set3,k_set4,k_set5,k_set6,k_set7,k_set8))

# k_set1 = uni_dist_k_set(0,2)
# k_set2 = log_dist_k_set(0,2)
# k_set = np.concatenate((k_set1, k_set2))
k_set = uni_dist_k_set(0,2) # depois tirar isto esta a dar overwrite

"""
plt.hist(k_set[:,6], bins=50, edgecolor='black')

plt.style.use('seaborn-white')
plt.show()
exit()"""

#species = ['O2(X)','O2(a)', 'O2(b)', 'O(3P)', 'O(1D)']
species = ['O2(X)','O2(a)', 'O(3P)']

def read_output(file_address):

    with open(file_address, 'r') as file :
        filedata = file.readlines()

    list=[]
    s = 0
    for line in filedata:
        if(species[s] in line):
            list.append(line.split()[1]) # save density value
            s+=1  # note: species does not go out of index because filedata ends
    return list

# C:\\Users\\clock\\Desktop\\LoKI_v3.1.0\\Code\\Output
densities =[]
# Read data from all output folders
for i in range(len(k_set)):
    file_address = 'C:\\Users\\clock\\Desktop\\LoKI_v3.1.0\\Code\\Output\\OxygenSimplified_1_' + str(i) + '\\chemFinalDensities.txt'
    densities.append(read_output(file_address))


densities = np.array(densities)
#print(densities)


with open('C:\\Users\\clock\\Desktop\\Python\\datapoints.txt', 'w') as file:
    for i in range(len(k_set)):
        k_line = k_set[i]
        densitie_line = densities[i]

        file.write('  '.join( "{:.4E}".format(item) for item in k_line))
        file.write('  ')
        file.write('  '.join( "{:.4E}".format(float(item)) for item in densitie_line)+'\n')
    