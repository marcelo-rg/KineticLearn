from genFiles_O2_novib import Simulations
import numpy as np
# import genFiles_O2_simple


if __name__ == "__main__":
    # path to LoKI
    loki_path = 'C:\\Users\\clock\\Desktop' + '\\LoKI_v3.1.0'
    
    # Definition of reaction scheme and setup files
    chem_file = "oxygen_novib.chem" 
    setup_file = "oxygen_chem_setup_novib.in"

    k_columns = [0,1,2] # we can repeat the same index for more than one parameter in the same reaction
    # pressures = [133.332, 1333.32] # P0, P1, P2, ... (in Pa)

    # Generate 10 equally spaced points between 1 and 50 Torr
    pressures_torr = np.linspace(1, 10, 5)

    # Convert from Torr to Pascal
    pressures_pa = pressures_torr * 133.332

    n_points = 1 
    k_true_values = [7.6e-22, 3E-44, 4e-20] # WARNING: the order of the k's 

    simul = Simulations(setup_file, chem_file, loki_path, n_points)
    simul.set_ChemFile_OFF() # turn off/on for fixed/changing values of k's
    simul.random_kset(k_columns, k_true_values, krange= [0.5,2], pdf_function='uniform')
    simul.fixed_pressure_set(pressures_pa)

    print(simul.nsimulations)

    # Run simulations
    simul.runSimulations()

    # Change the k_set to the true values to be written in the data file
    simul.parameters.k_set = np.tile(k_true_values, (simul.nsimulations, 1))

    simul.writeDataFile(filename='predicted_pressure_curve_1_10Torr.txt')