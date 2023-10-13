import numpy as np
import random
import math
random.seed(101)

# 1. Morris Sampling - use the same fixed bounds for all inputs - part I
def MorrisSampler(k_real, p, r, k_range_type, k_range, indexes=None):

    k_size = len(indexes)

    # 1. define region of experimentation w
    w = []
    if(k_range_type == "log"):
        w_log = np.logspace(k_range[0], k_range[1] ,p, base=10)
        mean_point = 1
        print("w log:" , w_log)
        w = w_log

    elif((k_range_type == "lin")):
        w_linear = np.linspace(k_range[0], k_range[1] ,p)
        mean_point = (k_range[1]+k_range[0])/2
        print("w linear:" , w_linear)
        w = w_linear

    # 2. define delta 
    delta = p/(2*(p-1))

    # create starting nodes
    start_nodes = []
    for i in range(r):
        start_nodes.append(random.choices(w,k= k_size)) # maybe use random.sample instead to avoid duplicates
    # print("start_nodes: ", start_nodes)
    # print("start_nodes.shape: ", np.array(start_nodes).shape)
    # create trajectories
    trajectories = []

    for (traj_idx, start_node) in enumerate(start_nodes):
        trajectory= []
        trajectory.append(start_node)                  # add starting node
        order = random.sample(range(0,k_size), k_size) # generate updating order

        # add the remaining nodes
        current_node = start_node.copy()
        for i in order:
            new_node = current_node.copy()

            if(new_node[i]/mean_point>1): # if it lies in the second half of the range interval
                new_node[i] = new_node[i]-delta
            else:
                new_node[i] = new_node[i]+delta

            trajectory.append(new_node)
            current_node = new_node
        
        # save current trajectory
        trajectories.append(trajectory)

    reshaped_traj =  np.reshape(trajectories, (r*(k_size+1),-1))  # r * (k+1) sets of inputs
    print("trajectories shape: ", np.array(reshaped_traj).shape)
    
    n_inputs = r * (k_size+1)
    print("n_inputs: ", n_inputs)

    # Transform to the real distribution of k values
    k_set=[]
    for item in k_real:
        k_set.append(np.full(n_inputs, item))

    for (i, k_idx) in enumerate(indexes):
        k_set[k_idx] =  reshaped_traj[:,i]* k_real[k_idx]

    print(np.transpose(np.array(k_set))[:,:3])
    return np.transpose(k_set)



# 2. Morris Sampling - use different bounds for each input feature - part II
def MorrisSampler2(boundaries, p, r, k_range_type, n_inputs, input_ref, indexes):

    # 1. define region of experimentation w
    w, delta, mean_points = [], [], []
    
    if(k_range_type == "log"):
        w_log, mean_points = [], []
        for idx in range(n_inputs):
            w_log.append(np.logspace(boundaries[idx][0], boundaries[idx][1] ,p, base=10))
            mean_points.append((boundaries[idx][0] + boundaries[idx][1])/2)
        w = w_log

    elif((k_range_type == "lin")):
        w_linear, mean_points_lin, multiples, delta_lin = [], [], [], []
        for idx in range(n_inputs):
            w_linear.append(np.linspace(boundaries[idx][0], boundaries[idx][1] ,p))
            mean_points_lin.append((boundaries[idx][0] + boundaries[idx][1])/2)
            multiples.append((boundaries[idx][0] + boundaries[idx][1])/(p-1))
            delta_lin.append(math.ceil(mean_points_lin[idx] / multiples[idx]) * multiples[idx])
        w = w_linear
        delta = delta_lin
        mean_points = mean_points_lin
         
    #check
    print("\nw = " , w)                    # w           = [[...], [...], [...]]
    print("mean points = " , mean_points)  # mean_points = [4.95, 1.06, 0.44]
    print("delta = " , delta, "/n")        # delta       = [5.50, 5.30, 3.08]

    # create starting nodes
    start_nodes = np.zeros((r, r, n_inputs))
    for i in range(r):
        for j in range(r):
            for k in range(n_inputs):
                start_nodes[i][j][k] = np.random.uniform(w[k][0], w[k][p-1])
    
    # create trajectories
    trajectories = []
    for (traj_idx, start_node) in enumerate(start_nodes):
        trajectory= []
        trajectory.append(start_node)                  # add starting node
        order = random.sample(range(0,n_inputs), n_inputs) # generate updating order

        # add the remaining nodes
        for i in order:
            new_node = start_node.copy()
            if(new_node[traj_idx, i] > mean_points[i]): # if it lies in the second half of the range interval
                new_node[traj_idx, i] = new_node[traj_idx, i] - delta[i]
            else:
                new_node[traj_idx, i] = new_node[traj_idx, i] + delta[i]
            trajectory.append(new_node)
        trajectories.append(trajectory)
    
    print("trajectories = ", trajectories)
    reshaped_traj =  np.reshape(trajectories, (r*(n_inputs+1),-1))  # r * (k+1) sets of inputs
    print("trajectories shape: ", np.array(reshaped_traj).shape)
    
    N_inputs = r * (n_inputs+1)
    print("n_inputs: ", N_inputs)

    # Transform to the real distribution of k values
    result = [[0] * N_inputs for _ in range(n_inputs)]
    for (i, input_idx) in enumerate(indexes):
        result[input_idx] =  reshaped_traj[:,i]*input_ref[input_idx]

    return np.transpose(result)
    
       