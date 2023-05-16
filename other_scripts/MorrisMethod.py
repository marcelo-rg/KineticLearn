import numpy as np
import random
random.seed(101)



def Morris(k_real, p = 10, n_traj = 50, k_range= [0.5,1.5], indexes = None):

    if(indexes!=None):
        k_size = len(indexes)
    # k_array = k_array(indexes)

    # define region of experimentation
    w_linear = np.linspace(k_range[0], k_range[1] ,p)
    mean_point = (k_range[1]+k_range[0])/2
    print("w linear:" , w_linear)

    # delta 
    delta = p/(p-1)/2
    print("delta: ",delta)

    # create starting nodes
    start_nodes = []
    for i in range(n_traj):
        start_nodes.append(random.choices(w_linear,k= k_size)) # maybe use random.sample instead to avoid duplicates

    # create trajectories
    trajectories = []
    for (traj_idx, start_node) in enumerate(start_nodes):
        trajectory= []
        # unit_vector = random.choices([-1,1], k= 3)
        # unit_vector[0] =1; unit_vector[-1] = -1  # to make sure we dont build trajectories out of range
        
        # add starting node
        trajectory.append(start_node)    
        # print("start node: ", start_node)  

        # generate updating order
        order = random.sample(range(0,k_size), k_size)
        # print("order:", order)

        # add the remaining nodes
        current_node = start_node.copy()
        for i in order:
            new_node = current_node.copy()
            # new_node[i] = new_node[i]+delta*unit_vector[i]
            if(new_node[i]/mean_point>1): # if it lies in the second half of the range interval
                new_node[i] = new_node[i]-delta
            else:
                new_node[i] = new_node[i]+delta

            # print("new_node: " , new_node)
            trajectory.append(new_node)
            current_node = new_node
        
        # print("trajectory: ",trajectory[0])
        # save current trajectory
        trajectories.append(trajectory)
    
    # n_traj * (k+1) sets of inputs
    reshaped_traj =  np.reshape(trajectories, (n_traj*(k_size+1),-1))
    # print(np.shape(reshaped_traj))
    # print(reshaped_traj)

    #number of inputs
    n_inputs = n_traj*(k_size+1)

    # Transform to the real distribution of k values
    k_set=[]
    for item in k_real:
        k_set.append(np.full(n_inputs, item))

    # k_set = np.array(k_set) # pass to array

    for (i, k_idx) in enumerate(indexes):
        k_set[k_idx] =  reshaped_traj[:,i]* k_real[k_idx]
        # print(k_set[k_idx])

    # print(np.shape(k_set))
    # print(np.transpose(k_set))

    return np.transpose(k_set)
    







if __name__ == '__main__':
    k = np.array([6E-16,1.3E-15,9.6E-16,2.2E-15,7E-22,3E-44,3.2E-45,5.2,53]) # total of 9 reactions
    Morris(k, indexes=[0,1,2])
