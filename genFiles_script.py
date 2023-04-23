import numpy as np
import matlab.engine
import random
import os
random.seed(10)


#-----------------------------------------------------------------------------------------------------
def random_kset(index1, index2, index3): #, index3, index4, index5, index6, index7):
    k_range = 10
    
    list=[]
    for item in k:
        list.append(np.full(n_trainSet, item ))   
    
    array_constant = np.array(list) # pass to array


    array_random = np.random.uniform(1,10, size = (k.size, n_trainSet))
    # array_random = np.random.lognormal(0,3,n_trainSet)*k_range
    #print(array_random)

    array_constant[index1]  =  array_random[index1]*k[index1]
    array_constant[index2]  =  array_random[index2]*k[index2] 
    array_constant[index3]  =  array_random[index3]*k[index3]

    return np.transpose(array_constant)

#-----------------------------------------------------------------------------------------------------

class Parameters():
    def __init__(self, k ,npoints = 10, krange=[1,10]):
        self.n_points = npoints
        self.k = k
        self.krange = krange
        self.pressure = None
        self.temp = None

        # output
        self.k_set = None
        self.pressure_set = None
    
    def random_kset(self,kcolumns = None): 

        array_random = np.random.uniform(self.krange[0], self.krange[1], size = (k.size, self.n_points))
        
        if kcolumns is None:
            # use all colunms
            kcolumns = np.arange(k.size)
        else:
            # create an k_set full of the constant values in k
            self.k_set = np.full((self.n_points, k.size), self.k)

        for (idx, item) in enumerate(kcolumns):
            self.k_set[:,item] = array_random[idx]*self.k[item]


    
    def set_npoints(self, npoints):
        self.n_points = npoints

    


def runSimulations(k_set, filename = 'datapoints.txt'):
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


    with open('C:\\Users\\clock\\Desktop\\Python\\'+ filename, 'w') as file:
        for i in range(len(k_set)):
            k_line = k_set[i]
            densitie_line = densities[i]

            file.write('  '.join( "{:.12E}".format(item) for item in k_line))
            file.write('  ')
            file.write('  '.join( "{:.12E}".format(float(item)) for item in densitie_line)+'\n')



if __name__ == '__main__':
    # Definition of scheme 
    k = np.array([6E-16,1.3E-15,9.6E-16,2.2E-15,7E-22,3E-44,3.2E-45,5.2,53]) # total of 9 reactions
    k_columns = [0,1,2] # if none change all rate coefficients
    n_react = k.size
    n_trainSet = 4
    randOn = True

    # Create training dataset
    k_set_training = random_kset(0,1,2)
    # print("k_set lenght: ", len(k_set_training))

    parameters = Parameters(k, npoints=n_trainSet, krange=[1,10])
    parameters.random_kset(k_columns)

    # test if parameters.k_set is equal to k_set_training
    print("k_set_training: ", k_set_training)
    print("parameters.k_set: ", parameters.k_set)

    # runSimulations(k_set_training)
    # writeDataFiles(n_trainSet)

    # # Create test dataset
    # n_trainSet = 10
    # k_set_test = random_kset(0,1,2)
    # runSimulations(k_set_test, filename='datapoints_test.txt')
    # print("Finished creating test data file")

