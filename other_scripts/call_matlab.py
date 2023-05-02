import numpy as np
import matplotlib.pyplot as plt
import matlab.engine
import random
import time
random.seed(10)

# eng.cd(r'C:\\Users\\clock\Desktop\\LoKI_v3.1.0', nargout=0)
# eng.ls(nargout=0)

#----------------------------------------------------------------------------------------------------------------------------------

k = np.array([6E-16,1.3E-15,9.6E-16,2.2E-15,7E-22,3E-44,3.2E-45,5.2,53]) # total of 9 reactions
n_react = k.size
n_trainSet = 5
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
    uni_distr = np.linspace(1, 10, n_trainSet)
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

#------------------------------------------------------------------------

def uni_dist_k_set_full():
    k1 = np.linspace(k[0], k[0]*2, n_trainSet)
    k_previous = k1

    for i in range(1, k.size, 1): #range(start, stop, step)
        ki = np.linspace(k[i], k[i]*10, n_trainSet)   

        k_set =  np.vstack((k_previous,ki))
        k_previous = k_set

    if(randOn):
        for _ in k_set:
            random.shuffle(_)

    return np.transpose(k_set)

#--------------------------------------------------------------------------
# Generate k set
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
k_set = uni_dist_k_set(0,2) 

print("k_set lenght: ", len(k_set))
#print("1: %f 2: %f 3: %f" %(len(k_set1), len(k_set2), len(k_set)))



# Read in the files
with open('C:\\Users\\clock\\Desktop\\Python\\O2_simple_1.chem', 'r') as file :
    chemFiledata = file.read()

with open('C:\\Users\\clock\\Desktop\\Python\\setup_O2_simple.in', 'r') as file :
    setup_data = file.read()

# print("Start")
# eng = matlab.engine.start_matlab()
# s = eng.genpath('C:\\Users\\clock\Desktop\\LoKI_v3.1.0')
# eng.addpath(s, nargout=0) # add loki code folder to search path of matlab
# eng.loki_loop(nargout=0)  # run the matlab script


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
        outfile.close()

    # Write out the setUp files
    setup_data = setup_data.replace('O2_simple_1_' +str(j-1)+'.chem', 'O2_simple_1_' +str(j)+'.chem') #replace chem file name to read
    setup_data = setup_data.replace('OxygenSimplified_1_' +str(j-1), 'OxygenSimplified_1_' +str(j)) #replace output folder name
    outfile = open('C:\\Users\\clock\\Desktop\\LoKI_v3.1.0\\Code\\Input\\SimplifiedOxygen\\setup_O2_simple_' +str(j)+'.in', 'w') 
    outfile.write(setup_data)
    outfile.close()


#----------------------------------------------------------------------------------------------------------------------------------------------------------------
print("Let's delay")
time.sleep(2)
print("Start")
eng = matlab.engine.start_matlab()
s = eng.genpath('C:\\Users\\clock\Desktop\\LoKI_v3.1.0')
eng.addpath(s, nargout=0) # add loki code folder to search path of matlab
eng.loki_loop(nargout=0)  # run the matlab script

print("Hello")