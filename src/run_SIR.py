from CLASS_SIR_model import *

import matplotlib.pyplot as plt






    
if __name__ == "__main__":
    ################################ PARAMETERS ###############################
    path = "/home/utente/Documenti/PRIN/Epidemics_models/joint_model_PARMA/data/2060/SSP1-2.6/ssp1_2060_provinces_IT.dat"
    d_load = load_results(path)
    mobility_matrix = d_load["mobility_matrix"]
    for i in range(len(mobility_matrix)):
        mobility_matrix[i][i] = 0.5
    population = d_load["population_distribution"]
    #population = [100]*len(population)
    time_step = 100
    time_span = [0,150] #[0,365.99] for 2040,2060,2080
    l_t = np.linspace(time_span[0], time_span[1], time_step)
    Model = Mobility_Matrix_Model  # The model
    parameter = {}
    tau = 1/5/np.median(population)
    gamma = 1/6
    proportion = 0.01
    number_infected = 1
    parameter["tau"] = tau
    parameter["gamma"] = gamma
    parameter["mobility_matrix"] = mobility_matrix
    parameter["population"] = population
    
    Y0 = initial_configuration(population,proportion,number_infected)
    
    
    # #####################RUN####################################################
    epidemics = Metapopulation(Y0, parameter, Model)
    epidemics.run_model(time_span, l_t)
    # # ###SAVE
    # #save_results(path_save, epidemics)
   
    # ##############PLOT###########################################
    l_R = []
    for i in range(1,107):
        t_var = [("Rh",i)]
        t_color = (('lightskyblue', 'blue'), ('lightsalmon', 'darkorange'),
              ('limegreen', 'green'), ('brown', 'red'))
        ODE_plotter(epidemics, None, t_var, t_color, MC_sim=False)
        plt.show()
        plt.close()
    for i in range(len(mobility_matrix)):
        l_R.append(epidemics.get_variables("Rh",i)[-1])
    print("sum people infected ", np.sum(l_R))
    
# =============================================================================

    ###### RUN  MONTECARLO ####
    N_sim =12
    leap = 0.5
    seed = [None for i in range(N_sim)]  # set the seed
    #l_t = np.linspace(time_span[0], time_span[1], time_span[1]*2)
    args = [(1, time_span, l_t, seed[i], leap) for i in range(N_sim)]
    l_T_MC, Y_MC = multiprocessing_MC(epidemics, args)
    print("time (in minutes): ", (time()-start)/60)
    for i in range(107):
        t_var = [('Ih', i)]
        t_color = (('lightskyblue', 'blue'), ('lightsalmon', 'darkorange'),
                    ('limegreen', 'green'), ('brown', 'red'))
        plotter(epidemics, None, t_var, t_color, MC_sim=False)
        #plotter(epidemics, None, t_var, t_color, MC_sim=True)
    sanity_check_total_population(epidemics,pop_distribution)

