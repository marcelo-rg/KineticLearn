import os

dict = {
    'O2_simple': {
        'n_conditions': 3, 
        'k_columns': [0, 1, 2],
        'n_densities': 3,
        'species': ['O2(X)', 'O2(a)', 'O(3P)'],
        'main_dataset': os.path.join('data', 'datapoints_mainNet_2k.txt'),
        'main_dataset_test': os.path.join('data', 'datapoints_mainNet_test.txt'),
        'surrogate_dataset': os.path.join('data', 'datapoints_pressure_'),
    },
    'O2_novib': {
        'n_conditions': 2, 
        'k_columns': [0, 1, 2],
        'n_densities': 11,
        'species': ['O2(X)', 'O2(a1Dg)', 'O2(b1Sg+)', 'O2(A3Su+_C3Du_c1Su-)', 'O2(+,X)', 'O(3P)', 'O(1D)', 'O(+,gnd)', 'O(-,gnd)', 'O3(X)', 'O3(exc)'],
        'main_dataset': os.path.join('data', 'datapoints_O2_novib_mainNet_2surrog.txt'),
        'main_dataset_test': os.path.join('data', 'datapoints_O2_novib_mainNet_2surrog_test.txt'),
        'surrogate_dataset': os.path.join('data', 'datapoints_O2_novib_pressure_'),
    },
}

only_one_pressure = {
    'O2_simple': {
        'n_conditions': 1, 
        'k_columns': [0, 1, 2],
        'n_densities': 3,
        'species': ['O2(X)', 'O2(a)', 'O(3P)'],
        'main_dataset': os.path.join('data', 'datapoints_pressure_0.txt'),
        'main_dataset_test': os.path.join('data', 'datapoints_pressure_0_test.txt'),
        'surrogate_dataset': os.path.join('data', 'datapoints_pressure_'),
    },
    'O2_novib': {
        'n_conditions': 1, 
        'k_columns': [0, 1, 2],
        'n_densities': 11,
        'species': ['O2(X)', 'O2(a1Dg)', 'O2(b1Sg+)', 'O2(A3Su+_C3Du_c1Su-)', 'O2(+,X)', 'O(3P)', 'O(1D)', 'O(+,gnd)', 'O(-,gnd)', 'O3(X)', 'O3(exc)'],
        'main_dataset': os.path.join('data', 'datapoints_O2_novib_pressure_0.txt'),
        'main_dataset_test': os.path.join('data', 'datapoints_O2_novib_pressure_0_test.txt'),
        'surrogate_dataset': os.path.join('data', 'datapoints_O2_novib_pressure_'),
    },
}