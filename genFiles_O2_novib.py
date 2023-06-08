import numpy as np
import matlab.engine
import os
import re
import SamplingMorrisMethod as morris
np.random.seed(10) # Recover reproducibility

#-----------------------------------------------------------------------------------------------------

class Parameters():
    def __init__(self, npoints):
        self.n_points = npoints

        # Set of parameters
        self.k_set = None
        self.pressure_set = None
        self.radius_set = None
        self.electDensity_set  = None
        self.kcolumns = None
    
    # protect these methods
    def morris_kset(self, k , p, r, k_range_type, k_range, kcolumns):
        if k is None:
            print('\nError: k is not defined. Please define k fixed values in the chem file or do not call gen k_set methods')
            exit()

        self.k_set =  morris.MorrisSampler(k, p, r, k_range_type, k_range, indexes= kcolumns)


    def random_kset(self, k ,kcolumns = None, krange= [1,10]): 
        k_size = len(k)
        self.kcolumns = kcolumns
        if k is None:
            print('\nError: k is not defined. Please define k fixed values in the chem file or do not call gen k_set methods')
            exit()
        array_random = np.random.uniform(krange[0], krange[1], size = (k_size, self.n_points))
        

        # create an k_set full of the constant values in k
        self.k_set = np.full((self.n_points, k_size), k)

        for (idx, item) in enumerate(kcolumns):
            self.k_set[:,item] = array_random[idx]*k[item]

    def fixed_kset(self, k):
        k_set = []
        for k_idx in range(k.size):
            k_set.append(k[k_idx]*np.ones(self.n_points))

        self.k_set = np.transpose(k_set)
    
    def random_pressure_set(self,pressure ,pressure_range = [1, 10]):
        self.pressure_set = pressure*np.random.uniform(pressure_range[0], pressure_range[1], size = self.n_points)

    def fixed_pressure_set(self, pressures):
        # make sure k_set is defined
        if self.k_set is None:
            print('\nError: k_set is not defined. Please define k_set values for each pressure '+
                  'or do not call fixed pressures method')
            exit()

        pressure_set = []
        for pressure in pressures:
            pressure_set.append(pressure*np.ones(self.n_points))
            
        self.pressure_set = np.array(pressure_set).flatten()

        # now make duplicates of random k_set to have the same size as pressure_set
        self.k_set = np.tile(self.k_set, (len(pressures),1))

        #update n_points
        self.n_points = self.pressure_set.size
    
    def random_radius_set(self,radius ,radius_range = [1, 10]):
        self.radius_set = radius*np.random.uniform(radius_range[0], radius_range[1], size = self.n_points)

    def random_electDensity_set(self,electDensity ,electDensity_range = [1, 10]):
        self.electDensity_set = electDensity*np.random.uniform(electDensity_range[0], electDensity_range[1], size = self.n_points)

    
    def set_npoints(self, npoints):
        self.n_points = npoints

#-----------------------------------------------------------------------------------------------------
    
class Simulations():
    def __init__(self, setup_file , chem_file, loki_path,npoints = 10):
        self.cwd = os.getcwd()
        self.loki_path = loki_path
        self.nsimulations = npoints
        self.parameters = Parameters(npoints)

        self._generateChemFiles= False
        # self.setup_file = setup_file
        self.outptFolder = chem_file[:-5]
        # create input folder if does not exist
        dir = self.loki_path+ '\\Code\\Input\\'+self.outptFolder
        if not os.path.exists(dir):
            os.makedirs(dir)


    def morris_kset(self, p, r, k_range_type, k_range, kcolumns=None):
        # read the k values from the chem file (to be used in the morris_kset method)
        if self._generateChemFiles:
            with open(self.cwd + '\\simulFiles\\' + chem_file, 'r') as file :
                values = []
                for line in file:
                    values.append(line.split()[-2])
            # create a numpy array with the k values of type float
            k = np.array(values)
            k = k.astype(float)
        else:
            k= None

        self.parameters.morris_kset(k, p, r, k_range_type, k_range, kcolumns)

    def random_kset(self, kcolumns , true_values , krange= [1, 10]): 
        self.parameters.random_kset(true_values , kcolumns, krange)

    def fixed_kset(self, k):
        self.parameters.fixed_kset(k)
    
    def random_pressure_set(self,pressure ,pressure_range = [1, 10]):
        self.parameters.random_pressure_set(pressure, pressure_range)

    def fixed_pressure_set(self,pressures):
        self.parameters.fixed_pressure_set(pressures)

        # update nsimualtions
        self.nsimulations = self.parameters.n_points
    
    def random_radius_set(self,radius ,radius_range = [1, 10]):
        self.parameters.random_radius_set(radius, radius_range)
  
    def random_electDensity_set(self,electDensity ,electDensity_range = [1, 10]):
        self.parameters.random_electDensity_set(electDensity, electDensity_range)



    # Private methods
    def _genChemFiles(self, indices):
        if self.parameters.k_set is None:
            print('\nError: k_set is not defined. Please use the random_kset method, for example')
            exit()

        def replace_values(filename, new_values, indices):
            assert len(new_values) == len(indices), "The length of new_values and indices must be the same."

            with open(self.cwd + '\\simulFiles\\' + chem_file, 'r') as file:
                content = file.readlines()

            matches = []
            flag = False
            for i, line in enumerate(content):
                if 'neutral species collisions' in line:
                    flag = True
                elif flag and 'constantRateCoeff' in line:
                    matches.append(i)

            for idx, value in zip(indices, new_values):
                if idx < len(matches):
                    line_parts = content[matches[idx]].split('|')
                    line_parts[2] = ' ' + "{:.4E}".format(value) + ' '
                    content[matches[idx]] = '|'.join(line_parts)

            # Write the updated content back to the file
            with open(filename, 'w') as file:
                file.writelines(content)


        # Replace for all self.parameters.k_set
        for j in range (0,self.nsimulations, 1): # for each datapoint 
            replace_values(filename= self.loki_path+ '\\Code\\Input\\'+self.outptFolder+'\\'+self.outptFolder+'_' +str(j)+'.chem',
                                new_values= self.parameters.k_set[j], indices= indices)
            



    def _genSetupFiles(self):
        # Read in the example file
        with open(self.cwd + '\\simulFiles\\' + setup_file, 'r') as file :
            setup_data = file.read() # (for the setup files we dont need to separate the string in lines)
        
        # Then replace for all self.parameters 
        for j in range (0, self.nsimulations, 1): 
            if self._generateChemFiles: 
                new_chemFile_name = self.outptFolder+ "\\\\"+ self.outptFolder+'_' +str(j)+ '.chem'
                setup_data = re.sub(r"chemFiles:\s*\n\s*- (.+?)\n", f"chemFiles:\n      - {new_chemFile_name}\n", setup_data) #replace chem file name
            setup_data = re.sub(r'folder:+\s+(\S+)', 'folder: ' + self.outptFolder +'_'+str(j), setup_data) #replace output folder name

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
            outfile = open(self.loki_path+ '\\Code\\Input\\'+self.outptFolder+'\\'+setup_file[:-3]+'_' +str(j)+'.in', 'w')
            outfile.write(setup_data)
            outfile.close()



    def _read_otpt_densities(self):

        def readFile(file_address):
            with open(file_address, 'r') as file :
                densities=[]
                for line in file:
                    if line.startswith(' '):
                        densities.append(line.split()[1])
            return densities
            

        densities =[]
        # Read data from all output folders
        for i in range(self.nsimulations):
            file_address = self.loki_path+ '\\Code\\Output\\'+self.outptFolder+'_' + str(i) + '\\chemFinalDensities.txt'
            densities.append(readFile(file_address))

        return np.array(densities)


    # Public methods
    def runSimulations(self):
        #------------------Generate the Chemistry + SetUp files--------------------------#
        if self._generateChemFiles:
            self._genChemFiles(self.parameters.kcolumns)
        else:
            print('\nChemistry files are not generated. The following file is used for all simulations: ' + chem_file)
            # read the example file and write it to the input folder
            with open(self.cwd + '\\simulFiles\\' + chem_file, 'r') as file:
                chemFiledata = file.read()
            outfile = open(self.loki_path+ '\\Code\\Input\\'+self.outptFolder+'\\'+chem_file[:-5] +'.chem', 'w')
            outfile.write(chemFiledata)
            outfile.close()


        self._genSetupFiles()
        

        #--------------------------------------Run the matlab script---------------------#
        outfile = open(self.loki_path + "\\loop_config.txt", 'w')
        outfile.write(str(self.nsimulations)) # save nsimul for matlab script
        outfile.write("\n"+ self.outptFolder+'\\'+setup_file[:-3]+'_') # save output folder name for matlab script
        outfile.close()
        os.chdir(self.loki_path+ "\\Code") # First change the working directory so that the relatives paths of loki work
        eng = matlab.engine.start_matlab()
        s = eng.genpath(self.loki_path)
        eng.addpath(s, nargout=0) # add loki code folder to search path of matlab
        eng.loki_loop(nargout=0)  # run the matlab script




    def writeDataFile(self, filename = 'datapoints.txt'):

        densities = self._read_otpt_densities()
        # electricCurrent = ...

        dir = self.cwd + '\\Data\\'
        if not os.path.exists(dir):
            os.makedirs(dir)

        # Write the data file
        with open(dir + filename, 'w') as file:
            for i in range(self.nsimulations):
                densitie_line = densities[i]

                # write the k values
                if self.parameters.k_set is not None:
                    k_line = self.parameters.k_set[i]
                    file.write('  '.join( "{:.12E}".format(item) for item in k_line))
                    file.write('  ')
                # write the pressure and radius values
                if self.parameters.pressure_set is not None:
                    file.write("{:.4E}".format(self.parameters.pressure_set[i]))
                    file.write('  ')
                if self.parameters.radius_set is not None:
                    file.write("{:.4E}".format(self.parameters.radius_set[i]))
                    file.write('  ')

                # write the densities values
                file.write('  '.join( "{:.12E}".format(float(item)) for item in densitie_line)+'\n')



    # Setters and getters
    def set_nsimulations(self, n):
        self.nsimulations = n
        self.parameters.set_npoints(n)

    def set_ChemFile_OFF(self):
        self._generateChemFiles = False

    def set_ChemFile_ON(self):
        self._generateChemFiles = True


if __name__ == '__main__': 

    # path to LoKI
    loki_path = "D:\\Marcelo" + '\\LoKI_v3.1.0'
    
    # Definition of reaction scheme and setup files
    chem_file = "oxygen_novib.chem" 
    setup_file = "oxygen_chem_setup_novib.in"

    k_columns = [0,1,2] 
    k_true_values = [7.6e-22, 3E-44, 4e-20] # WARNING: the order of the k's is too low to input into the scaler 
    pressures = [666.66] 
    n_simulations = 4

    simul = Simulations(setup_file, chem_file, loki_path, n_simulations)
    simul.set_ChemFile_ON() # turn off/on for fixed/changing values of k's
    simul.random_kset(k_columns, k_true_values, krange= [0.5,2])
    # simul.random_kset(kcolumns= k_columns, krange= [0.5,2]) # [0.5,2] range used in the Nsurrogates model
    # simul.fixed_pressure_set(pressures)

    # Run simulations
    simul.runSimulations()
    simul.writeDataFile(filename='datapoints_O2_novib_pressure_0.txt')