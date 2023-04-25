import numpy as np
import matlab.engine
import os
np.random.seed(10) # Recover reproducibility
import re

#-----------------------------------------------------------------------------------------------------

class Parameters():
    def __init__(self, k ,npoints):
        self.n_points = npoints
        self.k = k

        # Set of parameters
        self.k_set = None
        self.pressure_set = None
        self.radius_set = None
        self.electDensity_set  = None
    
    # protect these methods
    def random_kset(self,kcolumns = None, krange= [1,10]): 

        array_random = np.random.uniform(krange[0], krange[1], size = (k.size, self.n_points))
        
        if kcolumns is None:
            # use all colunms
            kcolumns = np.arange(k.size)
        else:
            # create an k_set full of the constant values in k
            self.k_set = np.full((self.n_points, k.size), self.k)

        for (idx, item) in enumerate(kcolumns):
            self.k_set[:,item] = array_random[idx]*self.k[item]
    
    def random_pressure_set(self,pressure ,pressure_range = [1, 10]):
        self.pressure_set = pressure*np.random.uniform(pressure_range[0], pressure_range[1], size = self.n_points)
    
    def random_radius_set(self,radius ,radius_range = [1, 10]):
        self.radius_set = radius*np.random.uniform(radius_range[0], radius_range[1], size = self.n_points)

    def random_electDensity_set(self,electDensity ,electDensity_range = [1, 10]):
        self.electDensity_set = electDensity*np.random.uniform(electDensity_range[0], electDensity_range[1], size = self.n_points)

#chamberRadius: 
    
    def set_npoints(self, npoints):
        self.n_points = npoints

#-----------------------------------------------------------------------------------------------------
    
class Simulations():
    def __init__(self, loki_path,  k ,npoints = 10):
        self.cwd = os.getcwd()
        self.loki_path = loki_path
        self.nsimulations = npoints
        self.parameters = Parameters(k, npoints)

        self._generateChemFiles= True

    # Compress Parameters methods
    def random_kset(self,kcolumns = None, krange= [1, 10]): 
        self.parameters.random_kset(kcolumns, krange)
    
    def random_pressure_set(self,pressure ,pressure_range = [1, 10]):
        self.parameters.random_pressure_set(pressure, pressure_range)
    
    def random_radius_set(self,radius ,radius_range = [1, 10]):
        self.parameters.random_radius_set(radius, radius_range)
  
    def random_electDensity_set(self,electDensity ,electDensity_range = [1, 10]):
        self.parameters.random_electDensity_set(electDensity, electDensity_range)

    # Private methods
    def _genChemFiles(self):
        if self.parameters.k_set is None:
            print('\nError: k_set is not defined. Please use the random_kset method, for example')
            exit()

        # Read in the example file
        with open(self.cwd + '\\O2_simple_1.chem', 'r') as file :  
            lines = []
            for line in file:
                lines.append(line.strip())
            chemFiledata = lines

        # Replace for all self.parameters.k_set
        for j in range (0,self.nsimulations, 1): # for each datapoint 
            list_out = []
            for i in range(len(self.parameters.k_set[j])): # for each k (for each line in file)
                line = chemFiledata[i]
                # the regular expression matches a decimal number in scientific notation
                line = re.sub(r'\d+.\d+E[+,-]?\d+', "{:.4E}".format(self.parameters.k_set[j][i]), line)
                list_out.append(line)

            # Write the out chem file 
            outfile = open(self.loki_path+ '\\Code\\Input\\SimplifiedOxygen\\O2_simple_1_' +str(j)+'.chem', 'w')
            outfile.write("\n".join(list_out))
            outfile.close()
            

    def _genSetupFiles(self):
        # Read in the example file
        with open(self.cwd + '\\setup_O2_simple.in', 'r') as file :
            setup_data = file.read() # (for the setup files we dont need to separate the string in lines)
        
        # Then replace for all self.parameters 
        for j in range (0, self.nsimulations, 1): 
            # replace the name of the chem file to read and the name of the output folder
            setup_data = setup_data.replace('O2_simple_1_' +str(j-1)+'.chem', 'O2_simple_1_' +str(j)+'.chem') #replace chem file name to read
            setup_data = setup_data.replace('OxygenSimplified_1_' +str(j-1), 'OxygenSimplified_1_' +str(j)) #replace output folder name

            if self.parameters.pressure_set is not None:
                # replace the pressure value in the setup file, that follows the string "gasPressure: "
                setup_data = re.sub(r'gasPressure: \d+.\d+', 'gasPressure: ' + "{:.4f}".format(self.parameters.pressure_set[j]), setup_data)

            if self.parameters.radius_set is not None:
                # replace the radius value in the setup file, that follows the string "chamberRadius: "
                setup_data = re.sub(r'chamberRadius: \d+.\d+', 'chamberRadius: ' + "{:.4f}".format(self.parameters.radius_set[j]), setup_data)

            if self.parameters.electDensity_set is not None:
                # replace the radius value in the setup file, that follows the string "electronDensity:  "
                setup_data = re.sub(r'electronDensity: \d+.\d+', 'electronDensity: ' + "{:.4f}".format(self.parameters.electDensity_set[j]), setup_data)

            # Write out the setUp files
            outfile = open(self.loki_path+ '\\Code\\Input\\SimplifiedOxygen\\setup_O2_simple_' +str(j)+'.in', 'w')
            outfile.write(setup_data)
            outfile.close()


    # Public methods
    def runSimulations(self):
        #------------------Generate the Chemistry + SetUp files--------------------------#
        if self._generateChemFiles:
            self._genChemFiles()
        self._genSetupFiles()

        #--------------------------------------Run the matlab script---------------------#
        outfile = open(self.loki_path + "\\n_simulations.txt", 'w')
        outfile.write(str(self.nsimulations))
        outfile.close()
        os.chdir(self.loki_path+ "\\Code") # First change the working directory so that the relatives paths of loki work
        eng = matlab.engine.start_matlab()
        s = eng.genpath(self.loki_path)
        eng.addpath(s, nargout=0) # add loki code folder to search path of matlab
        eng.loki_loop(nargout=0)  # run the matlab script
        exit()

    def writeDataFile(self, species, filename = 'datapoints.txt'):
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
        for i in range(self.nsimulations):
            file_address = self.loki_path+ '\\Code\\Output\\OxygenSimplified_1_' + str(i) + '\\chemFinalDensities.txt'
            densities.append(read_output(file_address))


        densities = np.array(densities)

        #------------------------------------------Write the data file----------------------#
        # get the path of the current working directory
        with open(self.cwd + filename, 'w') as file:
            for i in range(self.nsimulations):
                k_line = self.parameters.k_set[i]
                densitie_line = densities[i]

                file.write('  '.join( "{:.12E}".format(item) for item in k_line))
                file.write('  ')
                file.write('  '.join( "{:.12E}".format(float(item)) for item in densitie_line)+'\n')

    # Setters and getters
    def set_nsimulations(self, n):
        self.nsimulations = n
        self.parameters.set_npoints(n)

    def set_ChemFile_OFF(self):
        self._genChemFiles = False

    def set_ChemFile_ON(self):
        self._genChemFiles = True


if __name__ == '__main__': 

    # path to LoKI
    loki_path = "C:\\Users\\clock\\Desktop" + '\\LoKI_v3.1.0'

    # Definition of reaction scheme 
    k = np.array([6E-16,1.3E-15,9.6E-16,2.2E-15,7E-22,3E-44,3.2E-45,5.2,53]) # total of 9 reactions
    species = ['O2(X)','O2(a)', 'O(3P)']
    k_columns = [0,1,2] # if none, changes all rate coefficients
    n_trainSet = 5

    simul = Simulations(loki_path, k, n_trainSet)
    simul.random_kset(k_columns, krange=[1,10]) 
    simul.random_pressure_set(pressure= 133.322, pressure_range=[0.1,10]) # 1 Torr = 1133.322 Pa
    simul.random_radius_set(radius= 4e-3, radius_range=[1,5]) # [4e-3, 2e-2] 
    simul.random_electDensity_set(electDensity= 5e14, electDensity_range=[1,100]) # [5e14, 5e16]


    # Run simulations
    simul.runSimulations()
    simul.writeDataFile(species, filename='datapoints.txt')
