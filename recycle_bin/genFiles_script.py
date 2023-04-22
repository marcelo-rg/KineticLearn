import numpy as np
import random

k = np.array([6E-16,1.3E-15,9.6E-16,2.2E-15,7E-22,3E-44,3.2E-45,5.2,53]) # total of 9 reactions
n_react = k.size
n_trainSet = 100
randOn = True

def log_dist_k_set():

    log_distr = np.logspace(0, 1, n_trainSet, base = 10.) # apenas 1 ordem de grandeza

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

def uni_dist_k_set():
    k1 = np.linspace(k[0], k[0]*2, n_trainSet)
    k_previous = k1

    for i in range(1, k.size, 1): #range(start, stop, step)
        ki = np.linspace(k[i]*0.1, k[i]*10, n_trainSet)   

        k_set =  np.vstack((k_previous,ki))
        k_previous = k_set

    if(randOn):
        for _ in k_set:
            random.shuffle(_)

    return np.transpose(k_set)

#--------------------------------------------------------------------------
# Generate k set
k_set = uni_dist_k_set() # log distributed k's


#------------------Generate the Chemistry + SetUp files--------------------------#

# Read in the files
with open('C:\\Users\\clock\\Desktop\\Python\\O2_simple_1.chem', 'r') as file :
    chemFiledata = file.read()

with open('C:\\Users\\clock\\Desktop\\Python\\setup_O2_simple.in', 'r') as file :
    setup_data = file.read()

# First substitution: replace the values of array k
for i in range(len(k_set[0])):
    # Replace the target string
    chemFiledata = chemFiledata.replace("{:.2E}".format(k[i]), "{:.2E}".format(k_set[0][i]))
    # Write the out file data again
    outfile = open('C:\\Users\\clock\\Desktop\\LoKI_v3.1.0\\Code\\Input\\SimplifiedOxygen\\O2_simple_1_0.chem', 'w')
    outfile.write(chemFiledata)

# First substitution of the SetUp file
setup_data = setup_data.replace('O2_simple_1.chem', 'O2_simple_1_0.chem')
setup_data = setup_data.replace('OxygenSimplified_1', 'OxygenSimplified_1_0')
outfile = open('C:\\Users\\clock\\Desktop\\LoKI_v3.1.0\\Code\\Input\\SimplifiedOxygen\\setup_O2_simple_0.in', 'w')
outfile.write(setup_data)

# Then replace for all k_set
for j in range (1, len(k_set), 1):
    for i in range(len(k_set[j])):
        # Replace the target string
        chemFiledata = chemFiledata.replace("{:.2E}".format(k_set[j-1][i]), "{:.2E}".format(k_set[j][i]))
        outfile = open('C:\\Users\\clock\\Desktop\\LoKI_v3.1.0\\Code\\Input\\SimplifiedOxygen\\O2_simple_1_' +str(j)+'.chem', 'w')
        outfile.write(chemFiledata)

    # Write out the setUp files
    setup_data = setup_data.replace('O2_simple_1_' +str(j-1)+'.chem', 'O2_simple_1_' +str(j)+'.chem') #replace chem file name to read
    setup_data = setup_data.replace('OxygenSimplified_1_' +str(j-1), 'OxygenSimplified_1_' +str(j)) #replace output folder name
    outfile = open('C:\\Users\\clock\\Desktop\\LoKI_v3.1.0\\Code\\Input\\SimplifiedOxygen\\setup_O2_simple_' +str(j)+'.in', 'w') 
    outfile.write(setup_data)

#---------------------------------------------------
# Save histograms of k's distr.