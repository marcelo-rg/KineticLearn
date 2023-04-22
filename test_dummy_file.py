import numpy as np

# create a sample array
reg = np.array([ 0.0003,  
  0.0129,  
  0.0025,  
  0.0431])
no_reg = np.array([ 0.0979,  
0.9588,  
0.1114, 
1.2046])

# normalize the array
reg_norm = np.linalg.norm(reg, ord=1)
no_reg_norm = np.linalg.norm(no_reg, ord= 1)

# print the normalized array
print(reg / reg_norm)
print(no_reg / no_reg_norm)
