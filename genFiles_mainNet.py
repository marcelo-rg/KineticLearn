from genFiles_O2_simple import Simulations
import genFiles_O2_simple

if __name__ == "__main__":
    # path to LoKI
    loki_path = "C:\\Users\\clock\\Desktop" + '\\LoKI_v3.1.0'
    
    # Definition of reaction scheme and setup files
    chem_file = "O2_simple_1.chem" 
    setup_file = "setup_O2_simple.in"

    k_columns = [0,1,2] # if None, changes all columns
    pressures = [1333.32, 133.332] # P0, P1, P2, ... (in Pa)
    n_points = 500

    simul = Simulations(setup_file, chem_file, loki_path, n_points)
    simul.set_ChemFile_ON() # turn off/on for fixed/changing values of k's
    simul.random_kset(kcolumns= k_columns, krange= [0.5,2]) # [0.5,2] range used in the Nsurrogates model
    simul.fixed_pressure_set(pressures)

    # Run simulations
    simul.runSimulations()
    simul.writeDataFile(filename='datapoints_mainNet.txt')