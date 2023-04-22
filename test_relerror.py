import metrics_normalization as mn
import numpy as np

arr1 = np.array(([0.8, 1.2, 0.6], [0.8, 1.2, 0.6]))
arr2 = np.array(([0.9, 1.25, 0.78], [0.9, 1.25, 0.78]))

print("\n\nPrinting mean rel. err in validation:")
for idx, rel_err in enumerate(mn.rel_error(arr1, arr2)):
  print("k%d:   rel.err = %0.2f   " % \
      (idx+1, rel_err*100))