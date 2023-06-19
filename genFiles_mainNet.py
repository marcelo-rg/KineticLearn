from genFiles_O2_novib import Simulations
# import genFiles_O2_novib

if __name__ == "__main__":
    # path to LoKI
    loki_path = "D:\\Marcelo" + '\\LoKI_v3.1.0'
    
    # Definition of reaction scheme and setup files
    chem_file = "oxygen_novib.chem" 
    setup_file = "oxygen_chem_setup_novib.in"

    k_columns = [0,1,2] # if None, changes all columns
    pressures = [133.332, 1333.32] # P0, P1, P2, ... (in Pa)
    n_points = 5
    k_true_values = [7.6e-22, 3E-44, 4e-20] # WARNING: the order of the k's is 

    simul = Simulations(setup_file, chem_file, loki_path, n_points)
    simul.set_ChemFile_ON() # turn off/on for fixed/changing values of k's
    simul.random_kset(k_columns, k_true_values, krange= [0.5,2])
    simul.fixed_pressure_set(pressures)

    print(simul.nsimulations)
    # Run simulations
    simul.runSimulations()
    # simul.writeDataFile(filename='datapoints_O2_novib_mainNet.txt')