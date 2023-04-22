import metrics_normalization as mn
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
import torch as T
import torch.nn as nn
device = T.device("cpu")
from mlxtend.preprocessing import standardize
from mpl_toolkits import mplot3d


# -----------------------------------------------------------
class LoadDataset(T.utils.data.Dataset):
  # last 3 columns: densities (input)
  # first 9 columns: k's  (output)

  def __init__(self, src_file, m_rows=None):
    all_xy = np.loadtxt(src_file, max_rows=m_rows,
      usecols=[0,1,2,3,4,5,6,7,8,9,10,11], delimiter="  ",
      # usecols=range(0,9), delimiter="\t",
      comments="#", skiprows=0, dtype=np.float64)
  

    tmp_x = all_xy[:,[9,10,11]] 
    tmp_y = all_xy[:,[0,1,2,3,4,5,6,7,8]]

    # Normalize data
    tmp_x = mn.densitie_fraction(tmp_x)
    #tmp_y = np.log(tmp_y)
    #tmp_y = standardize(tmp_y)
    #tmp_x = np.log(tmp_x)
    
    #tmp_x = standardize(tmp_x)

    self.x_data = T.tensor(tmp_x, \
      dtype=T.float64).to(device)
    self.y_data = T.tensor(tmp_y, \
      dtype=T.float64).to(device)


  def __len__(self):
    return len(self.x_data)

  def __getitem__(self, idx):
    densities = self.x_data[idx,:]  # or just [idx]
    coef = self.y_data[idx,:] 
    return (densities, coef)       # tuple of two matrices 

# ------------------------------------------------------------------------------




#------------------------------------------------------------------------------------


src_file = 'C:\\Users\\clock\\Desktop\\Python\\datapointsk1k2k3.txt' 
full_dataset = LoadDataset(src_file) #,m_rows=500) 

index1 = 0

index2 = 1
index3 = 2

x_data = full_dataset.x_data
y_data = full_dataset.y_data


# Trajectories


# index = 2
# k_varia = y_data[:,index].numpy()

# density0 = x_data[:,0].numpy()
# density1 = x_data[:,1].numpy()
# density2 = x_data[:,2].numpy()

# A = 5  # We want figures to be A6
# plt.figure(figsize=(46.82 * .5**(.5 * A), 33.11 * .5**(.5 * A)))

# matplotlib.rc('xtick', labelsize=15) 
# matplotlib.rc('ytick', labelsize=15)
# matplotlib.rcParams.update({'font.size': 20}) 

# plt.ylim(0)
# #plt.xlim([k_varia[0], k_varia[0]*100])
# plt.vlines(x=k_varia[0], ymin=0, ymax=density0[0], colors='teal', ls='--', lw=2)
# plt.text(-2.0*k_varia[0],density0[0] , "x = %.2e" % k_varia[0], rotation=90, verticalalignment='center', fontsize= 12, color = 'teal')
# plt.plot(k_varia, density0 ,label= "O2(X)", linewidth='2.5')
# plt.plot(k_varia, density1, label= "O2(a)", linewidth='2.5')
# plt.plot(k_varia, density2,  label= "O(3P)", linewidth='2.5')
# plt.xlabel("k" + str(index+1))
# plt.ylabel("Density fraction  ($n_i/n$)")
# plt.legend()
# plt.savefig("C:\\Users\\clock\\Desktop\\Python\\data\\variation_oneataTime\\k" + str(index+1)+".png")
# plt.show()
# exit()

#-------------------------------------------------------------

fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
plt.rcParams["figure.autolayout"] = True





density0 = x_data[:,0].numpy()
density1 = x_data[:,1].numpy()
density2 = x_data[:,2].numpy()



k0 = y_data[:,index1].numpy()
k1 = y_data[:,index2].numpy()
k2 = y_data[:,index3].numpy()




#
# density02 = x_data2[:,0].numpy()
# density12 = x_data2[:,1].numpy()
# density22 = x_data2[:,2].numpy()

# k02 = y_data2[:,index1].numpy()
# k12 = y_data2[:,index2].numpy()
#


print(len(k0), " training points")

# K1, K2 = np.meshgrid(k0,k1)
# print(len(K1[0]))

# for index, value in enumerate(K1):
#   print(k0[index], "  ", value)


# Creating color map
my_cmap = plt.get_cmap('hsv')

#--------------------------------------------------------------------------
# Color map for k's
sctt = ax.scatter3D(density0, density1, density2,c= k0, cmap = my_cmap)
plt.title("k"+str(index1+1))
ax.set_xlabel('O2(x)', fontsize=15)
ax.set_ylabel('O2(a)', fontsize=15)
ax.set_zlabel('O(3P)', fontsize=15)

#-----------------------------------------------------------------
#Color map for densities
# sctt = ax.scatter3D(k0, k1, k2,c= density2, cmap = my_cmap)
# ax.set_xlabel('k'+str(index1+1), fontsize=15)
# ax.set_ylabel('k'+str(index2+1), fontsize=15)
# ax.set_zlabel('k'+str(index3+1), fontsize=15)

##
#ax.scatter3D(k02, k12, density02, color = "red", label= "O2(X)")
##
# ax.scatter3D(k0, k1, density1, color = "orange", label= "O2(a)")
# ax.scatter3D(k0, k1, density2, color = "green", label= "O(P)")
# plt.title("k"+str(index1+1)+ " vs " "k"+str(index2+1))
# ax.set_xlabel('density'+str(index1+1), fontsize=15)
# ax.set_ylabel('density'+str(index2+1), fontsize=15)
# ax.set_zlabel('density'+str(index2+1), fontsize=15)

fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 5)
plt.legend()
#plt.savefig("C:\\Users\\clock\\Desktop\\Python\\Images\\3dk1k3_unordered.pdf")
plt.show()





