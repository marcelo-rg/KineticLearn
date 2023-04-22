import torch as T
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import clean

dir_path = "C:\\Users\\clock\\Desktop\\Python\\Images\\statistics\\"
point = int(35)

# Load the values
load_dir = "stacks\\500_samples\\"
stack = T.load(dir_path+load_dir+"stack_train.pt") 
stack_test = T.load(dir_path+load_dir+"stack_test.pt") 
np_stack = stack_test.detach().numpy()
y_train = np.load(dir_path+load_dir+"training_targets.npy")
y_test = np.load(dir_path+load_dir+"test_targets.npy")
densities_targets = np.load(dir_path+load_dir+"densities_targets.npy")

print(densities_targets.shape)
exit()

# Means and variances
mean_test = T.mean(stack_test, dim=0).detach().numpy()
var_test = T.var(stack_test, dim=0).detach().numpy()


A = 5  # We want figures to be A5
plt.figure(figsize=(46.82 * .5**(.5 * A), 33.11 * .5**(.5 * A)))

matplotlib.rc('xtick', labelsize=15) 
matplotlib.rc('ytick', labelsize=15)
matplotlib.rcParams.update({'font.size': 20}) 


hist_colours = ["orange","green","purple"]
# Plot histograms
for idx in range(len(np_stack[0][0])):
    plt.clf()
    plt.hist(np_stack[:,point,idx], color= hist_colours[idx], bins= 20)
    plt.title("k"+ str(idx+1)+ " distribution")
    plt.xlabel("k"+ str(idx+1))

    textstr = '\n'.join((
    "Mean %.4f" % (mean_test[point,idx], ),
    "Std %.4f" % (np.sqrt(var_test[point,idx]), )))

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', alpha=0.5) #, facecolor='none', edgecolor='none')

    # place a text box in upper left in axes coords
    plt.text(0.65, 0.75, textstr, fontsize=18,  transform=plt.gca().transAxes,
        verticalalignment='top', bbox=props)

    plt.savefig(dir_path +"hist_k" +str(idx+1)+".png")

# Plot all hist together
plt.clf()
for idx in range(len(np_stack[0][0])):
    plt.hist(np_stack[:,point,idx], color= hist_colours[idx], label="k"+str(idx+1), bins=20)
plt.legend()
plt.title("Distributions")
plt.savefig(dir_path+ "hist_all.png")

#----------------------------------------------------------------------------------
def plot_predict_target(predict, target, sort_by_target= False, y_err = None):
    npoints = len(predict)
    x_ = np.arange(0,npoints,1)
    a = target # target
    b = predict # predicted
    ab = np.stack((a,b), axis=-1)
    sorted_ab = ab[ab[:,0].argsort()]
    if (not sort_by_target):
        sorted_ab = ab
    plt.plot(x_, sorted_ab[:,1], 'ro', label='predicted')
    plt.plot(x_, sorted_ab[:,0], 'bo', label= 'target')

    if (y_err is not None):
        plt.clf()
        plt.errorbar(x_, sorted_ab[:,1],yerr= y_err, fmt='ro',ecolor="gold", label='predicted')
        plt.plot(x_, sorted_ab[:,0], 'bo', label= 'target')
    plt.legend()
#------------------------------------------------------------------------------------------------------

# Plot values of k's with std as error bars
for idx in range(len(mean_test[0])):
    filename = 'C:\\Users\\clock\\Desktop\\Python\\Images\\statistics\\ks\\k' + str(idx+1)+'_test.png'
    plt.clf()
    a = y_test[:,idx] # target
    b = mean_test[:,idx] # predicted
    err = np.sqrt(var_test[:,idx])
    plot_predict_target(b, a, sort_by_target=True, y_err=err)
    plt.title('k'+str(idx+1))
    plt.savefig(filename)



def load_checkpoint(checkpoint):
    # checkpoint = T.Load(file)
    print("=> Loading checkpoint")
    net_forward.load_state_dict(checkpoint['state_dict'])


# Load the surrogate model 
# Create and load surrogate NN
net_forward = clean.Net_forward().to(clean.device)
net_forward.to(T.double) # set model to float64
load_checkpoint(T.load("checkpoint_forward_k1k2k3_minmax.pth.tar"))
net_forward.eval() # set mode

# Get stack of test densities predictions with surrogate
for item in stack_test:
    new_tensor =  T.reshape(net_forward(item), (1, 100, 3))
    try:
        tensor =  T.cat((tensor, new_tensor), dim=0)
        # print("deu")
    except:
        # print("nao deu")
        tensor = new_tensor

mean_densities = T.mean(tensor, dim=0).detach().numpy()
var_densities = T.var(tensor, dim=0).detach().numpy()

def rel_err(a,b):
    return np.abs(np.subtract(a,b)/a)

# print(rel_err(mean_densities,densities_targets).mean(axis=0))
# print(var_densities.mean(axis=0))

# Create a scatter plot of the two arrays against each other
species = ['O2(X)','O2(a)', 'O(3P)']
for idx in range(len(mean_densities[6])):
    filename = 'C:\\Users\\clock\\Desktop\\Python\\Images\\statistics\\densities_validation_' + str(idx+1)+'.png'
    plt.clf()
    predictions = mean_densities[:,idx]
    # predictions = tensor.detach().numpy()[0,:,idx]
    # predictions = np_stack[0,:,idx]
    plt.scatter(densities_targets[:,idx],predictions)

    rel_error = rel_err(predictions, densities_targets[:,idx])
    textstr = '\n'.join((
    r'$Mean\ \epsilon_{rel}=%.2f$%%' % (rel_error.mean()*100, ),
    r'$Max\ \epsilon_{rel}=%.2f$%%' % (max(rel_error)*100, )))

    # colour point o max error
    max_index = np.argmax(rel_err)
    plt.scatter(a[max_index],b[max_index] , color="gold", zorder= 2)

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', alpha=0.5) #, facecolor='none', edgecolor='none')

    # place a text box in upper left in axes coords
    plt.text(0.70, 0.25, textstr, fontsize=14,  transform=plt.gca().transAxes,
        verticalalignment='top', bbox=props)


    # Add labels and a title
    plt.xlabel('True values')
    plt.ylabel('Mean of predicted values')
    plt.title(species[idx])
    # Add a diagonal line representing perfect agreement
    plt.plot([0, 1], [0, 1], linestyle='--', color='k')
    plt.savefig(filename)

# a = [[100,100,100],[100,100,100]]
# b= [[150, 100, 200],[200, 200, 200]]
# print(rel_err(a,b))
# print(rel_err(a,b).mean(axis=0))
# exit()


#---------------------------SAVE THE PREDICTED k'S TO BE INSERTED IN THE SIMULATION AGAIN--------------------------------
# I want to test if with the predicted k's we obtain these same densities
# 1. Load datasat tobe able to access the scalers 
src_file = 'C:\\Users\\clock\\Desktop\\Python\\datapointsk1k2k3_3k.txt' 
species = ['O2(X)','O2(a)', 'O(3P)']
k_columns = [0,1,2] # Set to None to read all reactions/columns in the file
full_dataset = clean.LoadDataset(src_file, nspecies= len(species), react_idx= k_columns) #(data already scaled)
# (mean_test, densities_targets)
array = np.hstack((full_dataset.scaler.inverse_transform(mean_test),full_dataset.scaler_max_abs.inverse_transform(densities_targets)))
np.savetxt("C:\\Users\\clock\\Desktop\\Python\\predictions_test.txt", array, fmt= "%.4e")

# Conlcusions: i) The values of different k's in the solution space are correlated, meaning minimizing the value of one k first will dictate the value of the other
# ii) The values of k1 and k2 also have an influence on the O(3P) densitie which makes sense but was not obvious to me until this point

# Let us now plot the values of k1 vs k2 to vizualize their relation (I am not sure if they also depend on the value of k3) 
plt.clf()
plt.scatter(mean_test[:,0], mean_test[:,1])

plt.plot([0, 1], [0, 1], linestyle='--', color='k')
plt.xlabel('k1')
plt.ylabel('k2')
plt.title('k2 VS k1')
plt.savefig("C:\\Users\\clock\\Desktop\\Python\\Images\\statistics\\k2VSk1_means.png")

# iii) k2 VS k1 is Not a function

# Let us now include k3 in the visualization
# Creating color map
my_cmap = plt.get_cmap('hsv')
plt.clf()
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
plt.rcParams["figure.autolayout"] = True 

# Color map for k's
sctt = ax.scatter3D(mean_test[:,0], mean_test[:,1], mean_test[:,2],c= densities_targets[:,0], cmap = my_cmap)
plt.title('k3 VS k2 VS k1')
ax.set_xlabel('k1', fontsize=15)
ax.set_ylabel('k2', fontsize=15)
ax.set_zlabel('k3', fontsize=15)
# plt.show()
plt.savefig("C:\\Users\\clock\\Desktop\\Python\\Images\\statistics\\k3VSk2VSk1.png")

# Let us now plot the values of k1 vs k2 to vizualize their relation (I am not sure if they also depend on the value of k3) 
plt.clf()
stack_test = stack_test.detach().numpy()
for idx in range(20):
    plt.scatter(stack_test[:,idx,0], stack_test[:,idx,1])
# plt.scatter(stack_test[:,60,0], stack_test[:,60,1], label= np.array2string(densities_targets[60], formatter={'float_kind':lambda x: "%.2f" % x}))
# plt.scatter(stack_test[:,70,0], stack_test[:,70,1], label= np.array2string(densities_targets[70], formatter={'float_kind':lambda x: "%.2f" % x}))
# plt.scatter(stack_test[:,50,0], stack_test[:,50,1], label= np.array2string(densities_targets[50], formatter={'float_kind':lambda x: "%.2f" % x}))
# plt.scatter(stack_test[:,71,0], stack_test[:,71,1], label= np.array2string(densities_targets[71], formatter={'float_kind':lambda x: "%.2f" % x}))
plt.plot([0, 1], [0, 1], linestyle='--', color='k')
plt.xlabel('k1')
plt.ylabel('k2')
plt.title('k2 VS k1')
# plt.legend(loc ='best')
plt.savefig("C:\\Users\\clock\\Desktop\\Python\\Images\\statistics\\k2VSk1.png")

print(np.shape(stack_test[:,point,0]))

# iv) So we conclude k2/k1 is constant for each set of densities
# Let us plot k2/k1 VS densities
plt.clf()
yolo = (stack_test[:,:,1]/stack_test[:,:,0]).mean(axis=0)
yolo2 = mean_test[:,1]/mean_test[:,0]
print(yolo[10], yolo2[10])
# yolo = (stack_test[:,:,2]).mean(axis=0)
print(np.shape(yolo))
plt.scatter(densities_targets[:,1], yolo)
# plt.show()

# Now lets look if we could create and injective map between densties and k2/k1
plt.clf()
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
plt.rcParams["figure.autolayout"] = True 

# Color map for k's
sctt = ax.scatter3D(densities_targets[:,0], densities_targets[:,1], yolo)#,c= yolo, cmap = my_cmap)
plt.title('k2/k1 as function of densities')
ax.set_xlabel('O2(X)', fontsize=15)
ax.set_ylabel('O2(a)', fontsize=15)
ax.set_zlabel('k2/k1', fontsize=15)
plt.show()

