import numpy as np
import math
import matplotlib.pyplot as plt
import torch as T
import torch.nn as nn
device = T.device("cpu")
from mlxtend.preprocessing import standardize
from mpl_toolkits import mplot3d
from sklearn import preprocessing



scaler = preprocessing.MinMaxScaler()
scaler_max_abs = preprocessing.MaxAbsScaler()
# -----------------------------------------------------------
class LoadDataset(T.utils.data.Dataset):
  # last 3 columns: densities (input)
  # first 9 columns: k's  (output)

  def __init__(self, src_file, m_rows=None):
    all_xy = np.loadtxt(src_file, max_rows=m_rows,
      usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12], delimiter="  ",
      # usecols=range(0,9), delimiter="\t",
      comments="#", skiprows=0, dtype=np.float64)

    tmp_x = all_xy[:,[0,1,2,9]] # [0,1,2]
    tmp_y = all_xy[:,[10,11,12]] 
    #[0,1,2,3,4,5,6,7,8]

    # self.my_standardize = mn.Standardize()
    # tmp_y = self.my_standardize.standardization(tmp_y)


    # Normalize data
    scaler_max_abs.fit(tmp_y)
    tmp_y = scaler_max_abs.transform(tmp_y)
    # scaler.fit(tmp_y) # standard scaler
    # tmp_y = scaler.transform(tmp_y)

    #scale k's
    scaler.fit(tmp_x) # standard scaler
    tmp_x = scaler.transform(tmp_x)





    #tmp_x = standardize(tmp_x)
    #tmp_y = np.log(tmp_y)
    #tmp_y = standardize(tmp_y)

    self.x_data = T.tensor(tmp_x, \
      dtype=T.float64).to(device)
    self.y_data = T.tensor(tmp_y, \
      dtype=T.float64).to(device)
    self.all_data = T.tensor(all_xy, \
      dtype=T.float64).to(device)


  def __len__(self):
    return len(self.x_data)

  def __getitem__(self, idx):
    densities = self.x_data[idx,:]  # or just [idx]
    coef = self.y_data[idx,:] 
    return (densities, coef)       # tuple of two matrices 



#------------------------------------------------------------------------------------




#------------------------------------------------------------------------------------


src_file = 'data\\datapoints_pressure_0.5to1.5.txt'
full_dataset = LoadDataset(src_file) #,m_rows=500) 

x_data = full_dataset.x_data
y_data = full_dataset.y_data



"""

# Trajectories

k_varia = y_data[:,index].numpy()

density0 = x_data[:,0].numpy()
density1 = x_data[:,1].numpy()
density2 = x_data[:,2].numpy()

#plt.hist(to_plot, edgecolor='black', bins=50)
#plt.clf() ['O2(X)','O2(a)', 'O(3P)']

plt.plot(k_varia, density0 ,"o",label= "O2(X)", markersize= 3.0)
plt.plot(k_varia, density1,"o", label= "O2(a)", markersize= 3.0)
plt.plot(k_varia, density2,"o",  label= "O(3P)", markersize= 3.0)

#print(np.sort(to_plot))
plt.title("k" + str(index+1))
plt.xlabel("k" + str(index+1))
plt.ylabel("density fraction")
plt.legend()
plt.savefig("C:\\Users\\clock\\Desktop\\Python\\Images\\treino_log\\k" + str(index+1)+".png")
#plt.show()
exit()"""

# index = 1
# k_varia = y_data[:,index].numpy()

# density0 = x_data[:,0].numpy()
# density1 = x_data[:,1].numpy()
# density2 = x_data[:,2].numpy()

# plt.plot(k_varia, density0 ,"-o",label= "O2(X)", markersize= 3.0)
# plt.plot(k_varia, density1,"-o", label= "O2(a)", markersize= 3.0)
# plt.plot(k_varia, density2,"-o",  label= "O(3P)", markersize= 3.0)
# plt.legend()
# plt.show()
# exit()


fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
plt.rcParams["figure.autolayout"] = True

density0 = x_data[:,0].numpy()
density1 = x_data[:,1].numpy()
density2 = x_data[:,2].numpy()

k1 = y_data[:,0].numpy()
k2 = y_data[:,1].numpy()
k3 = y_data[:,2].numpy()
pressure = x_data[:,3].numpy()


print( len(k1), " training points")

# K1, K2 = np.meshgrid(k0,k1)
# print(len(K1[0]))

# for index, value in enumerate(K1):
#   print(k0[index], "  ", value)

Label = ["O2(X)","O2(a)", "O(3P)"]
# add colormap hvs to the scatter 3D plot
ax.scatter3D(k1, k2, pressure, c=density0, cmap='hsv', label= Label[0])



#ax.contour3D(K1, K2, density0, color = "blue", label= "O2(X)")
# ax.scatter3D(k0, k1, density0, color = "blue", label= "O2(X)")
##
#ax.scatter3D(k02, k12, density02, color = "red", label= "O2(X)")
##
# ax.scatter3D(k0, k1, density1, color = "orange", label= "O2(a)")
# ax.scatter3D(k0, k1, density2, color = "green", label= "O(P)")
plt.title(Label[0])
ax.set_xlabel('k1', fontsize=15)
ax.set_ylabel('k2', fontsize=15)
ax.set_zlabel('pressure', fontsize=15)
plt.legend()
#plt.savefig("C:\\Users\\clock\\Desktop\\Python\\Images\\3dk1k3_unordered.pdf")
plt.show()





