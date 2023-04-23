import numpy as np
import matlab.engine
import random
import os
random.seed(10)

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

    


def runSimulations(k_set, loki_path):
    #------------------Generate the Chemistry + SetUp files--------------------------#
    # get the path of the current working directory
    cwd = os.getcwd()

    # Read in the example files
    with open(cwd + '\\O2_simple_1.chem', 'r') as file :
        lines = []
        for line in file:
            lines.append(line.strip())
        chemFiledata = lines


    with open(cwd + '\\setup_O2_simple.in', 'r') as file :
        setup_data = file.read() # (for the setup files we dont need to separate the string in lines)

    list_out = []
    # First substitution: replace the values of array k
    for i in range(len(k_set[0])): 
        # Replace the target string
        line = chemFiledata[i].split()
        line[-2] = "{:.4E}".format(k_set[0][i])
        list_out.append('  '.join(line))


    # Write the out chem file data again
    outfile = open(loki_path+ '\\Code\\Input\\SimplifiedOxygen\\O2_simple_1_0.chem', 'w')
    outfile.write("\n".join(list_out))

    # First substitution of the SetUp file
    setup_data = setup_data.replace('O2_simple_1.chem', 'O2_simple_1_0.chem') # change the name of the chem file to read
    setup_data = setup_data.replace('OxygenSimplified_1', 'OxygenSimplified_1_0') # change the name of the output folder
    outfile = open(loki_path+ '\\Code\\Input\\SimplifiedOxygen\\setup_O2_simple_0.in', 'w')
    outfile.write(setup_data)

    # Then replace for all k_set
    for j in range (1, len(k_set), 1): # for each datapoint 
        list_out = []
        for i in range(len(k_set[j])): # for each k (for each line in file)
            # Replace the target string
            line = chemFiledata[i].split()
            line[-2] = "{:.4E}".format(k_set[j][i])
            list_out.append('  '.join(line))

        outfile = open(loki_path+ '\\Code\\Input\\SimplifiedOxygen\\O2_simple_1_' +str(j)+'.chem', 'w')
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
def writeDataFile(npoints, species, loki_path, filename = 'datapoints.txt'):
    #species = ['O2(X)','O2(a)', 'O2(b)', 'O(3P)', 'O(1D)']
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

    densities =[]
    # Read data from all output folders
    for i in range(npoints):
        file_address = loki_path+ '\\Code\\Output\\OxygenSimplified_1_' + str(i) + '\\chemFinalDensities.txt'
        densities.append(read_output(file_address))


    densities = np.array(densities)
    #print(densities)

    # get the path of the current working directory
    cwd = os.getcwd()
    with open(cwd + filename, 'w') as file:
        for i in range(npoints):
            k_line = k_set[i]
            densitie_line = densities[i]

            file.write('  '.join( "{:.12E}".format(item) for item in k_line))
            file.write('  ')
            file.write('  '.join( "{:.12E}".format(float(item)) for item in densitie_line)+'\n')



if __name__ == '__main__':

    # Get current working directory
    cwd = os.getcwd()
    loki_path = "C:\\Users\\clock\\Desktop" + '\\LoKI_v3.1.0'

    # Definition of scheme 
    k = np.array([6E-16,1.3E-15,9.6E-16,2.2E-15,7E-22,3E-44,3.2E-45,5.2,53]) # total of 9 reactions
    species = ['O2(X)','O2(a)', 'O(3P)']
    k_columns = [0,1,2] # if none, changes all rate coefficients
    n_react = k.size
    n_trainSet = 10

    # Create training dataset
    parameters = Parameters(k, npoints=n_trainSet, krange=[1,10])
    parameters.random_kset(k_columns)

    # Run simulations
    runSimulations(parameters.k_set, loki_path)
    writeDataFile(species, loki_path, filename='datapoints.txt')

    # runSimulations(k_set_training)
    # writeDataFiles(n_trainSet)

    # # Create test dataset
    # n_trainSet = 10
    # k_set_test = random_kset(0,1,2)
    # runSimulations(k_set_test, filename='datapoints_test.txt')
    # print("Finished creating test data file")

