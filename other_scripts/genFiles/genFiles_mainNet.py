from genFiles_O2_novib import Simulations
# import genFiles_O2_simple

if __name__ == "__main__":
    # path to LoKI
    loki_path = "D:\\Marcelo" + '\\LoKI_v3.1.0'
    
    # Definition of reaction scheme and setup files
    chem_file = "oxygen_novib.chem" 
    setup_file = "oxygen_chem_setup_novib.in"

    k_columns = [0,1,2,3,3] # we can repeat the same index for more than one parameter in the same reaction
    pressures = [133.332, 1333.32] # P0, P1, P2, ... (in Pa)
    n_points = 5
    k_true_values = [7.6e-22, 3E-44, 4e-20, 4e-20, 1e-16] # WARNING: the order of the k's 

    simul = Simulations(setup_file, chem_file, loki_path, n_points)
    simul.set_ChemFile_ON() # turn off/on for fixed/changing values of k's
    simul.random_kset(k_columns, k_true_values, krange= [0.5,2], pdf_function='uniform')
    # simul.latin_hypercube_kset(k_true_values, k_columns, krange= [0.5,2])
    simul.fixed_pressure_set(pressures)

    print(simul.nsimulations)
    # Run simulations
    simul.runSimulations()
    simul.writeDataFile(filename='dummy_5ks.txt')


    # # path to LoKI
    # loki_path = "D:\\Marcelo" + '\\LoKI_v3.1.0'
    
    # # Definition of reaction scheme and setup files
    # chem_file = "O2_simple_1.chem" 
    # setup_file = "setup_O2_simple.in"

    # # k_true = [6.0e-16, 1.3e-15, 9.6e-16]
    # k_columns = [0,1,2] # if None, changes all columns
    # pressures = [133.332, 1333.32] # P0, P1, P2, ... (in Pa)
    # n_points = 16

    # simul = Simulations(setup_file, chem_file, loki_path, n_points)
    # simul.set_ChemFile_ON() # turn off/on for fixed/changing values of k's
    # # simul.random_kset(k_columns, krange= [0.5,2], pdf_function='uniform')
    # # simul.latin_hypercube_kset(k_columns, krange= [0.5,2])
    # simul.morris_kset(p = 10 , r = int(n_points/4), k_range_type='lin', k_range= [0.5,2], kcolumns= k_columns)

    # simul.fixed_pressure_set(pressures)
    
    # print(simul.nsimulations)

    # # Run simulations
    # simul.runSimulations()
    # simul.writeDataFile(filename='O2_simple_morris.txt')