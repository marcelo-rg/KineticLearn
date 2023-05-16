import numpy as np
import random
import time
import pdb
import sys
import os

module_dir = os.path.abspath('/Users/matildevalente/Documents/PIC/')
sys.path.append(module_dir)
from src import SamplingLatinHypercube as lhs
from src import SamplingMorrisMethod as morris
from src import SamplingUniform as unif

random.seed(10)
np.random.seed(10) 

def runSimulations(k_set, filename):
    # 1. Generate the Chemistry + SetUp files

    # 1.1. Read in the files
    with open('/Users/matildevalente/Documents/PIC/Results_Part1/generationScripts/O2_simple_1.chem', 'r') as file :
        lines = []
        for line in file:
            lines.append(line.strip())
        chemFiledata = lines

    with open('/Users/matildevalente/Documents/PIC/Results_Part1/generationScripts/setup_O2_simple.in', 'r') as file :
        setup_data = file.read() # for the setup files we dont need to separate the string in lines

    list_out = []
    # 1.2. First substitution: replace the values of array k
    for i in range(len(k_set[0])): 
        # Replace the target string
        line = chemFiledata[i].split()
        line[-2] = "{:.4E}".format(k_set[0][i])
        list_out.append('  '.join(line))

    # 1.3. Write the out chem file data again
    outfile = open('/Users/matildevalente/Documents/PIC/Results_Part1/LoKI_v3.1.0-v2/Code/Input/SimplifiedOxygen/O2_simple_1_0.chem', 'w')
    outfile.write("\n".join(list_out))

    # 1.4. First substitution of the SetUp file
    setup_data = setup_data.replace('O2_simple_1.chem', 'O2_simple_1_0.chem')
    setup_data = setup_data.replace('OxygenSimplified_1', 'OxygenSimplified_1_0')
    outfile = open('/Users/matildevalente/Documents/PIC/Results_Part1/LoKI_v3.1.0-v2/Code/Input/SimplifiedOxygen/setup_O2_simple_0.in', 'w')
    outfile.write(setup_data)

    # 1.5. Then replace for all k_set
    for j in range (1, len(k_set), 1): # for each datapoint 
        list_out = []
        for i in range(len(k_set[j])): # for each k (for each line in file)
            # Replace the target string
            line = chemFiledata[i].split()
            line[-2] = "{:.4E}".format(k_set[j][i])
            list_out.append('  '.join(line))

        outfile = open('/Users/matildevalente/Documents/PIC/Results_Part1/LoKI_v3.1.0-v2/Code/Input/SimplifiedOxygen/O2_simple_1_' +str(j)+'.chem', 'w')
        outfile.write("\n".join(list_out))
        outfile.close()

        # Write out the setUp files
        setup_data = setup_data.replace('O2_simple_1_' +str(j-1)+'.chem', 'O2_simple_1_' +str(j)+'.chem') #replace chem file name to read
        setup_data = setup_data.replace('OxygenSimplified_1_' +str(j-1), 'OxygenSimplified_1_' +str(j)) #replace output folder name
        outfile = open('/Users/matildevalente/Documents/PIC/Results_Part1/LoKI_v3.1.0-v2/Code/Input/SimplifiedOxygen/setup_O2_simple_' +str(j)+'.in', 'w') 
        outfile.write(setup_data)
        outfile.close()
    
    print("Running stoped here. Please run the MatLab code to generate the outputs.")

    pdb.set_trace()
    # Prompt user to resume program
    resume_input = input("Press 'r' to resume program: ")
    if resume_input == 'r':
        print("Resuming running")

        """ 
        # 2. RUN THE MATLAB SCRIPT -- run directly on matlab
        os.chdir('/Users/matildevalente/Documents/PIC/LoKI_v3.1.0-v2/Code') # First change the working directory so that the relatives paths of loki work
        eng = matlab.engine.start_matlab()
        s = eng.genpath('/Users/matildevalente/Documents/PIC/LoKI_v3.1.0-v2')
        eng.addpath(s, nargout=0) # add loki code folder to search path of matlab
        eng.loki_loop(nargout=0)  # run the matlab script

        """
        #--------------------------------------------Retrieve output data-----------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------

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

        densities =[]
        # Read data from all output folders
        for i in range(len(k_set)):
            file_address = '/Users/matildevalente/Documents/PIC/Results_Part1/LoKI_v3.1.0-v2/Code/Output/OxygenSimplified_1_' + str(i) + '/chemFinalDensities.txt'
            densities.append(read_output(file_address))

        densities = np.array(densities)

        with open('/Users/matildevalente/Documents/PIC/Results_Part1/dataFolder/'+ filename, 'w') as file:
            for i in range(len(k_set)):
                k_line = k_set[i]
                densitie_line = densities[i]

                file.write('  '.join( "{:.12E}".format(item) for item in k_line))
                file.write('  ')
                file.write('  '.join( "{:.12E}".format(float(item)) for item in densitie_line)+'\n')
        densities = densities.astype(np.float64)

        return densities

# 1. Define Dataset Structure
start = time.time()
n_lines = 30

k_ref = np.array([6E-16,1.3E-15,9.6E-16,2.2E-15,7E-22,3E-44,3.2E-45,5.2,53]) # total of 9 reactions
k_range = [0.2,10] 


n_react = k_ref.size 
randOn = True

# 2. Generate samples
sampling_method = input("Which method to generate samples? \n 1. Uniform Sampling;\n 2. Morris Method;\n 3. Latin Hypercube Sampling;\nAnswer: ")
if(sampling_method == str(1)):
    print("Generating uniform sample.")
    uniform_sample = unif.UniformSampler(n_lines, n_react, k_range, k_ref)
    print("Done.")
    
    filename = 'Uniform/' + str(n_react) + 'k_' + str(n_lines) + '_NEW.txt'
    uniform_outputs = runSimulations(uniform_sample,filename)

elif(sampling_method == str(2)):
    print("Generating sample using Morris Method.")
    p = 8 
    k_range_lin= k_range
    k_range_log= [-1,1]
    k_range_type = "lin"
    r = int(n_lines/(n_react+1))      #nº of trajectories

    if(k_range_type == "log"):
        morris_sample = morris.MorrisSampler(k_ref, p, r, k_range_type, k_range_log,indexes=[0,1,2,3,4,5,6,7,8])
    elif(k_range_type == "lin"):
        morris_sample = morris.MorrisSampler(k_ref, p, r, k_range_type, k_range_lin,indexes=[0,1,2,3,4,5,6,7,8])
    
    #filename = 'Morris/Linear/p'+str(p)+'/' + str(n_react) + 'k_' + str(n_lines) + '_NEW.txt'
    #morris_outputs = runSimulations(morris_sample, filename)

elif(sampling_method == str(3)):
    print("Generating sample using Latin Hypercube Sampling.")
    
    # Get standard normal and standard uniform samples
    minRatio = 1
    uni,std = lhs.sample(n_react,n_lines,minRatio)
    uniformSample = lhs.getUniformSample(uni, k_ref, k_range)
    print("Done.")

    filename = 'LatinHypercube/' + str(n_react) + 'k_' + str(n_lines) + '_NEW.txt'
    unifSampleOutputs = runSimulations(uniformSample, filename)

else:
    print("Please select a valid answer!")

end = time.time()
print("LoKI took %.2f seconds to run. " % \
  (end-start))
