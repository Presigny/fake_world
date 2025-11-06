# Copyright Charley Presigny 2025
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
import geopandas as gpd
import pandas as pd

def euclidean_distance_matrix(position):
    matrix_distance = np.zeros(shape=[len(position), len(position)])
    for j in range(len(position)):
        for i in range(len(position)):
            x0, y0 = position[i][0], position[i][1]
            x1, y1 = position[j][0], position[j][1]
            diff = (x1 - x0, y1 - y0)
            matrix_distance[i][j] = np.sqrt(np.dot(diff, diff))
    return matrix_distance

def soneira_peebles_model(lamb,eta,L,R,erase_nodes=False):
    position = [(0,0)]
    final_position = []
    rng = np.random.default_rng()
    for i in range(L):
        l_pos = []
        for x,y in position:
            l_random = (R/(lamb**i))*rng.random((eta,2))
            for x_rand,y_rand in l_random:
                l_pos.append((x_rand-x,y_rand-y))
        position = np.copy(l_pos)
        final_position = final_position + l_pos
    if type(erase_nodes) == int and erase_nodes >=0:
        position = list(position)
        while erase_nodes != 0:
            index = rng.choice(range(len(position)))
            position.pop(index)
            erase_nodes -= 1
    position = np.array(position)
    print(len(position))        
    distance_matrix = cdist(position, position)
    return np.transpose(position,(1,0)),distance_matrix

def sanity_check_dist_mat(mat): # check if no repeating elements
    for i in range(len(mat)):
        if len(np.unique(mat[i])) !=len(mat[i]):
            print ("matrix not suitable for numerical version of radiation model becuase repeating distances")
    return 0

# =============================================================================
# def soneira_peebles_extended(lamb,eta,L,R,erase_nodes=False):
#     position = [(0,0)] # where the positions of the i-th circles are stored
#     final_position = [] #where all the positions are stored
#     rng = np.random.default_rng()
#     for i in range(1,L+1):
#         l_pos = [] #
#         number_new_elem = eta
#         while number_new_elem !=0: # attribute randomly to the ith circles the i+1 circles
#             for x,y in position:
#                 number_in_circle = np.random.randint(0,number_new_elem+1) 
#                 number_new_elem -= number_in_circle
#                 l_random = (R/(lamb**i))*rng.random((number_in_circle,2))
#                 for x_rand,y_rand in l_random:
#                     l_pos.append((x_rand-x,y_rand-y))
#         final_position = final_position + l_pos
#         position = np.copy(l_pos)
#     distance_matrix = euclidean_distance_matrix(final_position)
#     final_position = np.transpose(final_position,(1,0))
#     return final_position,distance_matrix
# =============================================================================

def unfold_matrix(matrix):
    distribution = []
    for i in range(len(matrix)):
        for j in range(i):
            distribution.append(matrix[i,j])
    return distribution

def energy_of_system(distance_matrix,pop_distribution,l_selected):
    energy = 0
    for i in range(len(l_selected)):
        for j in range(i):
            energy += pop_distribution[i]*pop_distribution[j]/distance_matrix[l_selected[i],l_selected[j]]
    return energy

def norm_vector(vec):
    norm = np.sqrt(np.dot(vec,vec))
    if norm != 0:
        return [vec[i]/norm for i in range(len(vec))]
    else:
        return vec
    
def find_matching_point(position,x,y):
    distance = 1e99
    for i in range(len(position)):
        new_distance = euclidean(position[i], (x,y))
        if  new_distance < distance:
            new_position = position[i]
            index = i
            distance = new_distance
    return index,new_position
        

def antico_distance_population(position,pop_distribution,distance_matrix,number_step,percentile=90):
    N = len(pop_distribution)
    rng = np.random.default_rng()
    percentile_population = np.percentile(pop_distribution, percentile)
    position = np.transpose(position,(1,0))
    for step in range(number_step):
        proba = pop_distribution /np.sum(pop_distribution)
        i = rng.choice(range(N),p=proba)
        u_i =[0,0]
        x_i,y_i = position[i][0],position[i][1]
        for j in range(N):
            x_j,y_j = position[j][0],position[j][1]
            if i != j:
                distance_ij = distance_matrix[i,j]
                if (pop_distribution[i] >= percentile_population and pop_distribution[j] >= percentile_population):                    
                    u_i[0] += -(x_j-x_i)/ (distance_ij*distance_ij)
                    u_i[1] += -(y_j-y_i)/ (distance_ij*distance_ij)
        u_i = norm_vector(u_i)
        u_i = [np.mean(distance_matrix)*u_i[i] for i in range(len(u_i))]
        #print("u_i",u_i)
        new_x_i = position[i][0]+ u_i[0]
        new_y_i = position[j][1] + u_i[1]
        #print(new_x_i,new_y_i)
        new_index,new_pos = find_matching_point(position,new_x_i,new_y_i)
        pop_distribution[new_index],pop_distribution[i] = pop_distribution[i],pop_distribution[new_index]
    return pop_distribution

def v2antico_distance_population(position,pop_distribution,distance_matrix,number_step,percentile=90):
    N = len(pop_distribution)
    rng = np.random.default_rng()
    percentile_population = np.percentile(pop_distribution, percentile)
    position = np.transpose(position,(1,0))
    for step in range(number_step):
        proba = pop_distribution /np.sum(pop_distribution)
        i = rng.choice(range(N),p=proba)
        u_i =[0,0]
        x_i,y_i = position[i][0],position[i][1]
        for j in range(N):
            x_j,y_j = position[j][0],position[j][1]
            if i != j:
                distance_ij = distance_matrix[i,j]
                if (pop_distribution[i] >= percentile_population and pop_distribution[j] >= percentile_population):                    
                    u_i[0] += -(x_j-x_i)/ (distance_ij*distance_ij)
                    u_i[1] += -(y_j-y_i)/ (distance_ij*distance_ij)
                else:
                    u_i[0] += (x_j-x_i)/ (distance_ij*distance_ij)
                    u_i[1] += (y_j-y_i)/ (distance_ij*distance_ij)
        u_i = norm_vector(u_i)
        u_i = [np.mean(distance_matrix)*u_i[i] for i in range(len(u_i))]
        #print("u_i",u_i)
        new_x_i = position[i][0]+ u_i[0]
        new_y_i = position[j][1] + u_i[1]
        #print(new_x_i,new_y_i)
        new_index,new_pos = find_matching_point(position,new_x_i,new_y_i)
        pop_distribution[new_index],pop_distribution[i] = pop_distribution[i],pop_distribution[new_index]
    return pop_distribution


def co_distance_population(position,pop_distribution,distance_matrix,number_step,percentile=90):
    N = len(pop_distribution)
    rng = np.random.default_rng()
    percentile_population = np.percentile(pop_distribution, percentile)
    position = np.transpose(position,(1,0))
    for step in range(number_step):
        proba = pop_distribution /np.sum(pop_distribution)
        i = rng.choice(range(N),p=proba)
        u_i =[0,0]
        x_i,y_i = position[i][0],position[i][1]
        for j in range(N):
            x_j,y_j = position[j][0],position[j][1]
            if i != j:
                distance_ij = distance_matrix[i,j]
                if (pop_distribution[i] >= percentile_population and pop_distribution[j] >= percentile_population):
                    u_i[0] += (x_j-x_i)/ (distance_ij*distance_ij)
                    u_i[1] += (y_j-y_i)/ (distance_ij*distance_ij)
                elif (pop_distribution[i] < percentile_population and pop_distribution[j] < percentile_population):
                    u_i[0] += (x_j-x_i)/ (distance_ij*distance_ij)
                    u_i[1] += (y_j-y_i)/ (distance_ij*distance_ij)
        u_i = norm_vector(u_i)
        u_i = [np.mean(distance_matrix)*u_i[i] for i in range(len(u_i))]
        #print("u_i",u_i)
        new_x_i = position[i][0]+ u_i[0]
        new_y_i = position[j][1] + u_i[1]
        #print(new_x_i,new_y_i)
        new_index,new_pos = find_matching_point(position,new_x_i,new_y_i)
        #print("new_pos", new_pos)
        
        #print(new_pos,pop_distribution[i])
        pop_distribution[new_index],pop_distribution[i] = pop_distribution[i],pop_distribution[new_index]
    return pop_distribution

    ##################################
    
def quantify_correlation(percentile,pop_distribution,distance_matrix):
    extract = [i for i in range(len(pop_distribution)) if pop_distribution[i] >= np.percentile(pop_distribution,percentile)]
    list_distance = []
    for i in range(len(extract)):
        for j in range(i):
            index_i = extract[i]
            index_j = extract[j]
            list_distance.append(distance_matrix[index_i,index_j])
    return np.mean(list_distance)
        
        

if __name__ == "__main__":         
    import methods_two_point_correlation as mtpc
    ####PARAMETERS #############################################
    eta = 9 #density, number of circles by steps
    l = 6 #fraction of space covered by the circles
    L = 4  #size of circles, number of steps
    gamma = 2 - np.log(eta)/np.log(l)
    print("gamma", gamma)
    print("fractal dimension",np.log(eta)/np.log(l))
    print("l = ", l)
    R = 1322000 # radius max
    
    min_rad = R/(l**(L-1))
    nbins=20
    rmin=0
    N_run= 10000
    size=5
    print("mimimum radius",min_rad )
    position,distance_matrix = soneira_peebles_model(l,eta,L,R)
    d = {}
    d["x"] = position[0]
    d["y"] = position[1]
    #df = pd.DataFrame((position))
    # gdf = gpd.GeoDataFrame(
    #      d, geometry=gpd.points_from_xy(position[0], position[1]), crs="EPSG:4326")
    # r_edges,xi = mtpc.compute_two_point_correlation(gdf,gdf_edge,N_run,size,rmin,nbins=nbins,crs="EPSG:4326")
    # distance_distribution = mtpc.compute_DD(gdf)
#     distribution = np.triu(distance_matrix)
#     distribution = distribution[distribution !=0]
#     sns.histplot(distribution,stat="probability",bins=40)
#    #  N_patch = len(distance_matrix[0])
#    #  max_size = 1e6
#    #  number_step = 100
#    #  pop_distribution = [max_size*(r**-1) for r in range(1,N_patch+1)]
#    #  pop_distribution[1],pop_distribution[2],pop_distribution[3] =1e6,1e6,1e6
#    #  #pop_distribution  = antico_distance_population(position,pop_distribution,distance_matrix,number_step,99)
#    #  pop_distribution  = antico_distance_population(position,pop_distribution,distance_matrix,number_step,99)
#    #  #rng = np.random.default_rng()
#    #  #rng.shuffle(pop_distribution)
#    #  distribution = unfold_matrix(distance_matrix)
#    #  log_size = [np.log(pop_distribution[i])for i in range(len(pop_distribution))]
#    #  print("minimum analytics,", min_rad)
#    #  mean_max = quantify_correlation(99,pop_distribution,distance_matrix)
#    #  print("mean distance of bigger points,", mean_max)
#    #  print("average distance of points", np.mean(distance_matrix))
#    # # distribution = distribution /  np.max(distribution)
#    #  plt.scatter(position[0],position[1],s=log_size,c=pop_distribution,alpha=0.5)
#    #  plt.show()
# # =============================================================================
#    # emp_matrix = np.load("it.npy")
#     #emp_distribution = unfold_matrix(emp_matrix)
#     #emp_distribution = emp_distribution /  np.max(emp_distribution)
# #     #emp_matrix_2 = np.load("fr.npy")
# #     #emp_distribution_2 = unfold_matrix(emp_matrix_2)
# #     #emp_distribution_2 = emp_distribution_2 /  np.max(emp_distribution_2)
    
#     #sns.histplot(emp_distribution,stat="probability",binwidth=0.05,kde=True)
    
#     plt.xscale("log")
#     plt.vlines(min_rad,ymax=0.06,ymin=0)
#     plt.grid()
#     plt.show()
#     plt.scatter(position[0],position[1],color="red",alpha=0.5,s=0.1)
#     plt.grid()
#     plt.show()
# =============================================================================
    
    #np.save("synthetic_data/position_sp_model",position)


# =============================================================================
#     import fitter
#     
#     f= fitter.Fitter(distribution,xmin=80,distributions="powerlaw")
#     f.fit()
#     f.summary()
#     f.fitted_param
# =============================================================================
        
    
    
    
    
    

    
        
    
        
    

