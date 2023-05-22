import numpy as np
import matlab.engine
import matplotlib.pyplot as plt
import matplotlib
from sklearn import preprocessing
import os

k = np.array([6E-16,1.3E-15,9.6E-16,2.2E-15,7E-22,3E-44,3.2E-45,5.2,53]) # total of 9 reactions
n_simulations = 100

#----------------------Read the k's to validate----------------------------
src_file = "C:\\Users\\clock\\Desktop\\Python\\predictions_test.txt"
all_xy = np.loadtxt(src_file, max_rows=None, delimiter=" ",
      comments="#", skiprows=0, dtype=np.float64)

k_changing = all_xy[:,[0,1,2]]
true_densities = all_xy[:,[3,4,5]]

list=[]
for item in k:
    list.append(np.full(n_simulations, item ))

# print(np.shape(list), np.shape(k_changing.T))
array_constant = np.array(list) # pass to array

k_changing = np.transpose(k_changing)

index1 = 0
index2 = 1
index3 = 2

array_constant[index1]  =  k_changing[index1]
array_constant[index2]  =  k_changing[index2]
array_constant[index3]  =  k_changing[index3]

# print(np.shape(array_constant), array_constant.T)
# exit()

k_set = array_constant.T

#------------------Generate the Chemistry + SetUp files--------------------------#

# Read in the files
with open('C:\\Users\\clock\\Desktop\\Python\\O2_simple_1.chem', 'r') as file :
    lines = []
    for line in file:
        lines.append(line.strip())
    chemFiledata = lines


with open('C:\\Users\\clock\\Desktop\\Python\\setup_O2_simple.in', 'r') as file :
    setup_data = file.read() # for the setup files we dont need to separate the string in lines

list_out = []
# First substitution: replace the values of array k
for i in range(len(k_set[0])): 
    # Replace the target string
    #chemFiledata = chemFiledata.replace("{:.2E}".format(k[i]), "{:.5E}".format(k_set[0][i]))
    line = chemFiledata[i].split()
    line[-2] = "{:.4E}".format(k_set[0][i])
    list_out.append('  '.join(line))


# Write the out chem file data again
#outfile = open('C:\\Users\\clock\\Desktop\\Python\\ChemFiles\\O2_simple_1_0.chem', 'w')
outfile = open('C:\\Users\\clock\\Desktop\\LoKI_v3.1.0\\Code\\Input\\SimplifiedOxygen\\O2_simple_1_0.chem', 'w')
outfile.write("\n".join(list_out))

# First substitution of the SetUp file
setup_data = setup_data.replace('O2_simple_1.chem', 'O2_simple_1_0.chem')
setup_data = setup_data.replace('OxygenSimplified_1', 'OxygenSimplified_1_0')
outfile = open('C:\\Users\\clock\\Desktop\\LoKI_v3.1.0\\Code\\Input\\SimplifiedOxygen\\setup_O2_simple_0.in', 'w')
outfile.write(setup_data)

# Then replace for all k_set
for j in range (1, len(k_set), 1): # for each datapoint 
    list_out = []
    for i in range(len(k_set[j])): # for each k (for each line in file)
        # Replace the target string
        #chemFiledata = chemFiledata.replace("{:.5E}".format(k_set[j-1][i]), "{:.5E}".format(k_set[j][i]))
        line = chemFiledata[i].split()
        line[-2] = "{:.4E}".format(k_set[j][i])
        list_out.append('  '.join(line))

    #outfile = open('C:\\Users\\clock\\Desktop\\Python\\ChemFiles\\O2_simple_1_' +str(j)+'.chem', 'w')
    outfile = open('C:\\Users\\clock\\Desktop\\LoKI_v3.1.0\\Code\\Input\\SimplifiedOxygen\\O2_simple_1_' +str(j)+'.chem', 'w')
    outfile.write("\n".join(list_out))
    outfile.close()

    # Write out the setUp files
    setup_data = setup_data.replace('O2_simple_1_' +str(j-1)+'.chem', 'O2_simple_1_' +str(j)+'.chem') #replace chem file name to read
    setup_data = setup_data.replace('OxygenSimplified_1_' +str(j-1), 'OxygenSimplified_1_' +str(j)) #replace output folder name
    outfile = open('C:\\Users\\clock\\Desktop\\LoKI_v3.1.0\\Code\\Input\\SimplifiedOxygen\\setup_O2_simple_' +str(j)+'.in', 'w') 
    outfile.write(setup_data)
    outfile.close()




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


with open('C:\\Users\\clock\\Desktop\\Python\\predictions_test_datapoints_test.txt', 'w') as file:
    for i in range(len(k_set)):
        k_line = k_set[i]
        densitie_line = densities[i]

        file.write('  '.join( "{:.4E}".format(item) for item in k_line))
        file.write('  ')
        file.write('  '.join( "{:.4E}".format(float(item)) for item in densitie_line)+'\n')

print(np.shape(true_densities), np.shape(densities))

A = 5  # We want figures to be A6
plt.figure(figsize=(46.82 * .5**(.5 * A), 33.11 * .5**(.5 * A)))

matplotlib.rc('xtick', labelsize=15) 
matplotlib.rc('ytick', labelsize=15)
matplotlib.rcParams.update({'font.size': 15}) 

ticks = np.arange(1,11,1)
plt.xticks(ticks)

# print(true_densities[:,0])
# print(densities[0,0], densities[1,0])
# print(true_densities[0,0], true_densities[1,0])

# plt.ylim([0.5e22, 1e23])
float_formatter = "{:.4e}".format
# densities = densities.tolist()
densities_list = []
for line in densities:
    densities_list.append([float_formatter(float(item)) for item in line])

densities = np.array(densities_list, dtype=float)
# print(type(densities[0,0]))
# print(true_densities)

species = ['O2(X)','O2(a)', 'O(3P)']
for index in range(len(true_densities[0])):
    plt.clf()
    a = true_densities[:,index] # target
    b = densities[:,index] # predicted
    ab = np.stack((a,b), axis=-1)
    sorted_ab = ab[ab[:,0].argsort()]
    plt.plot(np.arange(1, len(densities)+1), sorted_ab[:,0], 'bo', label='true densities')

    plt.plot(np.arange(1, len(densities)+1), sorted_ab[:,1], 'ro', label='densities (using pred. ks)')
    plt.title(species[index])
    plt.legend()
    plt.savefig('C:\\Users\\clock\\Desktop\\Python\\Images\\densities_validation_'+ str(index+1))


scaler_max_abs = preprocessing.MaxAbsScaler()
scaler_max_abs.fit(true_densities)
true_densities = scaler_max_abs.transform(true_densities)
densities = scaler_max_abs.transform(densities)
#Plot Correlation-Graphs: predicted densities vs true densities
for index in range(len(true_densities[0])):
    plt.clf()
    a = true_densities[:,index] # target
    b = densities[:,index] # predicted

    plt.plot(a, b, 'ro', label=species[index] +" density")
    plt.plot(a,a ,color="Blue" ,label= "Y=X line")
    
    rel_err = np.abs(np.subtract(a,b)/a)
    # print(rel_err)
    # print("stats: ",stats.chisquare(f_obs= b, f_exp= a))

    textstr = '\n'.join((
    r'$Mean\ \epsilon_{rel}=%.2f$%%' % (rel_err.mean()*100, ),
    r'$Max\ \epsilon_{rel}=%.2f$%%' % (max(rel_err)*100, )))

    # colour point o max error
    max_index = np.argmax(rel_err)
    plt.scatter(a[max_index],b[max_index] , color="gold", zorder= 2)

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', alpha=0.5) #, facecolor='none', edgecolor='none')

    # place a text box in upper left in axes coords
    plt.text(0.70, 0.25, textstr, fontsize=14,  transform=plt.gca().transAxes,
        verticalalignment='top', bbox=props)

    plt.xlabel("True values")
    plt.ylabel("Data with predicted coeffitients")
    plt.legend()
    # plt.show()
    plt.savefig('C:\\Users\\clock\\Desktop\\Python\\Images\\densities_validation_correlation'+ str(index+1))