import numpy as np
import matplotlib.pyplot as plt
import matlab.engine
import random
import os
random.seed(10)

k = np.array([6E-16,1.3E-15,9.6E-16,2.2E-15,7E-22,3E-44,3.2E-45,5.2,53]) # total of 9 reactions
n_react = k.size
n_trainSet = 10
randOn = False


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

#------------------------------------------------------------------------

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
k_set = uni_dist_k_set_mesh(0,2) 

print("k_set lenght: ", len(k_set))
#print("1: %f 2: %f 3: %f" %(len(k_set1), len(k_set2), len(k_set)))


"""
to_plot = k_set[:,0]
#plt.hist(to_plot, bins=50, edgecolor='black')
print(to_plot)

plt.plot(np.arange(0,len(to_plot),1), to_plot)
plt.style.use('seaborn-white')
plt.show()
exit()"""



#------------------Generate the Chemistry + SetUp files--------------------------#

# Read in the files
with open('C:\\Users\\clock\\Desktop\\Python\\O2_simple_1.chem', 'r') as file :
    chemFiledata = file.read()

with open('C:\\Users\\clock\\Desktop\\Python\\setup_O2_simple.in', 'r') as file :
    setup_data = file.read()

# First substitution: replace the values of array k
for i in range(len(k_set[0])):
    # Replace the target string
    chemFiledata = chemFiledata.replace("{:.2E}".format(k[i]), "{:.5E}".format(k_set[0][i]))
    # Write the out chem file data again
    outfile = open('C:\\Users\\clock\\Desktop\\Python\\ChemFiles\\O2_simple_1_0.chem', 'w')
    #outfile = open('C:\\Users\\clock\\Desktop\\LoKI_v3.1.0\\Code\\Input\\SimplifiedOxygen\\O2_simple_1_0.chem', 'w')
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
        chemFiledata = chemFiledata.replace("{:.5E}".format(k_set[j-1][i]), "{:.5E}".format(k_set[j][i]))
        outfile = open('C:\\Users\\clock\\Desktop\\Python\\ChemFiles\\O2_simple_1_' +str(j)+'.chem', 'w')
        #outfile = open('C:\\Users\\clock\\Desktop\\LoKI_v3.1.0\\Code\\Input\\SimplifiedOxygen\\O2_simple_1_' +str(j)+'.chem', 'w')
        outfile.write(chemFiledata)
        outfile.close()

    # Write out the setUp files
    setup_data = setup_data.replace('O2_simple_1_' +str(j-1)+'.chem', 'O2_simple_1_' +str(j)+'.chem') #replace chem file name to read
    setup_data = setup_data.replace('OxygenSimplified_1_' +str(j-1), 'OxygenSimplified_1_' +str(j)) #replace output folder name
    outfile = open('C:\\Users\\clock\\Desktop\\LoKI_v3.1.0\\Code\\Input\\SimplifiedOxygen\\setup_O2_simple_' +str(j)+'.in', 'w') 
    outfile.write(setup_data)
    outfile.close()

exit()

#---------------------------------------------------
# Save histograms of k's distr.


#--------------------------------------Run the matlab script-------------------------------------------------------------------
os.chdir("C:\\Users\\clock\\Desktop\\LoKI_v3.1.0\\Code") # First change the working directory so that the relatives paths of loki work
eng = matlab.engine.start_matlab()
s = eng.genpath('C:\\Users\\clock\Desktop\\LoKI_v3.1.0')
eng.addpath(s, nargout=0) # add loki code folder to search path of matlab
eng.loki_loop(nargout=0)  # run the matlab script


#--------------------------------------------Retrieve output data-----------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------

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
