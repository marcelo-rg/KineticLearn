import metrics_normalization as mn
import numpy as np
import math
import matplotlib.pyplot as plt
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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # The Linear() class defines a fully connected network layer
        self.hid1 = nn.Linear(3, 20)  # hidden 1
        #self.hid2 = nn.Linear(10, 10) # hidden 2
        #self.hid3 = nn.Linear(50, 50) # hidden 3
        #self.hid4 = nn.Linear(50, 50) # hidden 4
        self.oupt = nn.Linear(20, 9)  # output

    # Missing initialization of weights
    #T.nn.init.uniform_(self.hid1.weight, -0.05, +0.05)

    def forward(self, x):
        z = T.relu(self.hid1(x)) # try also relu activ. f.
        #z = T.relu(self.hid2(z))
        #z = T.relu(self.hid3(z))
        #z = T.tanh(self.hid4(z))
        z = self.oupt(z)  # no activation
        return z


#------------------------------------------------------------------------------------




#------------------------------------------------------------------------------------


src_file = 'C:\\Users\\clock\\Desktop\\Python\\datapoints.txt'
full_dataset = LoadDataset(src_file) #,m_rows=500) 

x_data = full_dataset.x_data
y_data = full_dataset.y_data


#
full_dataset2 = LoadDataset('C:\\Users\\clock\\Desktop\\Python\\datapoints_mesh_random.txt') #,m_rows=500) 

x_data2 = full_dataset2.x_data
y_data2 = full_dataset2.y_data
#

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

index1 = 1
index2 = 2

k0 = y_data[:,index1].numpy()
k1 = y_data[:,index2].numpy()



#
density02 = x_data2[:,0].numpy()
density12 = x_data2[:,1].numpy()
density22 = x_data2[:,2].numpy()

k02 = y_data2[:,index1].numpy()
k12 = y_data2[:,index2].numpy()
#


print( len(k0), " training points")

# K1, K2 = np.meshgrid(k0,k1)
# print(len(K1[0]))

# for index, value in enumerate(K1):
#   print(k0[index], "  ", value)



#ax.contour3D(K1, K2, density0, color = "blue", label= "O2(X)")
ax.scatter3D(k0, k1, density0, color = "blue", label= "O2(X)")
##
#ax.scatter3D(k02, k12, density02, color = "red", label= "O2(X)")
##
ax.scatter3D(k0, k1, density1, color = "orange", label= "O2(a)")
ax.scatter3D(k0, k1, density2, color = "green", label= "O(P)")
plt.title("k"+str(index1+1)+ " vs " "k"+str(index2+1))
ax.set_xlabel('k'+str(index1+1), fontsize=15)
ax.set_ylabel('k'+str(index2+1), fontsize=15)
plt.legend()
#plt.savefig("C:\\Users\\clock\\Desktop\\Python\\Images\\3dk1k3_unordered.pdf")
plt.show()





