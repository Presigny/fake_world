import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt


def analytical_radiation_model(pop_distribution,distance_matrix):
    N_patch = len(pop_distribution)
    T = np.zeros([N_patch,N_patch])
    total_pop = np.sum(pop_distribution)
    for i in range(N_patch):
        p_i = pop_distribution[i]
        for j in range(N_patch):
            if i != j:
                p_j = pop_distribution[j]
                radius_ij = distance_matrix[i,j]
                l_index = [k for k in range(len(distance_matrix[i])) if distance_matrix[i,k] <= radius_ij and k != j and k != i]
                s_ij = 0
                for index in l_index:
                    s_ij += pop_distribution[index]
                norm = 1/(1-(p_i/(total_pop)))
                T[i,j] = norm*(p_i*p_j)/((p_i+s_ij)*(p_i+p_j+s_ij))
    return T

def ar_model_pool(i,pop_distribution,distance_matrix):
    N_patch = len(pop_distribution)
    T = np.zeros([1,N_patch])
    total_pop = np.sum(pop_distribution)
    p_i = pop_distribution[i]
    for j in range(N_patch):
        if i != j:
            p_j = pop_distribution[j]
            radius_ij = distance_matrix[i,j]
            l_index = [k for k in range(len(distance_matrix[i])) if distance_matrix[i,k] <= radius_ij and k != j and k != i]
            s_ij = 0
            for index in l_index:
                s_ij += pop_distribution[index]
            T[0,j] = (p_i*p_j)/((p_i+s_ij)*(p_i+p_j+s_ij))
    T = T/np.sum(T)
    return T
    

def multiprocessing_radiation_model(pop_distribution,distance_matrix):
    N_patch = len(pop_distribution)
    T = np.zeros([N_patch,N_patch])
    args = [[i,pop_distribution,distance_matrix] for i in range(N_patch)]
    with Pool(12) as pool:
        result = pool.starmap(ar_model_pool,args)
    for i in range(N_patch):
        T[i] = result[i]
    return T


def plotter(metapopulation, name_file, t_variables, t_color, MC_sim=False):
    Y = []
    l_mean = []
    l_std = []
    T = metapopulation.get_variables("T")
    print(t_variables)
    for tup_name, tup_point in t_variables:
        print(tup_name, tup_point)
        Y.append(metapopulation.get_variables(tup_name, which_point=tup_point))
        mean, std = metapopulation.average_MonteCarlo(
            T, tup_name, which_point=tup_point)
        l_mean.append(mean)
        l_std.append(std)
    if MC_sim:
        Y_MC = []
        t_MC = metapopulation.get_variables_Montecarlo("T")
        #name_file = name_file+"_MC"
        for tup_name, tup_point in t_variables:
            Y_MC = metapopulation.get_variables_Montecarlo(
                tup_name, which_point=tup_point)
            for i in range(len(Y_MC)):
                plt.scatter(t_MC[i], Y_MC[i], s=1, alpha=1)

    for i in range(len(Y)):
        plt.errorbar(T, l_mean[i], yerr=l_std[i], label="MC point = "+str(
            t_variables[i][1])+" "+str(t_variables[i][0]), color=t_color[i][0])
        plt.plot(T, Y[i], label="ODEs point = "+str(t_variables[i]
                 [1])+" "+str(t_variables[i][0]), color=t_color[i][1])
    plt.xlabel("time (days)")
    plt.ylabel("number of individuals")
    #plt.xlim([40,365])
    #plt.ylim([0,4])
    plt.legend()
    plt.show()


def ODE_plotter(metapopulation, name_file, t_variables, t_color, MC_sim=False):
    Y = []
    l_mean = []
    l_std = []
    T = metapopulation.get_variables("T")
    print("time", T[-1])
    for tup_name, tup_point in t_variables:
        print(tup_name, tup_point)
        Y.append(metapopulation.get_variables(tup_name, which_point=tup_point))
    for i in range(len(Y)):
        plt.plot(T, Y[i], label="ODEs point = "+str(t_variables[i]
                 [1])+" "+str(t_variables[i][0]), color=t_color[i][1])
    plt.xlabel("time (days)")
    plt.ylabel("number of individuals")
    #plt.xlim([40,365])
    #plt.ylim([0,4])
    #plt.yscale("log")
    plt.legend()
    
def initial_configuration(population,proportion,number_infected):
    # M ratio mosquito over humans
    N_patch = len(population)
    #K = [0]*(N_patch+1)
    Y0 = np.zeros(shape=[N_patch,3])
    fraction_infected = int(N_patch*proportion)
    l_index = range(N_patch)
    index_infected = np.random.choice(l_index,size=fraction_infected,replace=False)
    Y0[:,0] = population 
    for i in index_infected:
            Y0[i][0] -= number_infected #set the human pop
            Y0[i][1] = number_infected #set the number of infected 
    return Y0