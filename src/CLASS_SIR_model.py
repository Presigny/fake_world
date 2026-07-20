# /// script
# dependencies = [
#  "numpy == 1.26.4",
#  "matplotlib == 3.8.4",
#  "scipy == 1.13.1",
# ]
# ///
import numpy as np
import matplotlib.pyplot as plt
import sys
import datetime
import copy
import pickle
import pprint
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
from scipy.stats import expon,poisson
from scipy.stats import zipf
from multiprocessing import Pool
from time import time

class Spatial_Kernel_Model:
    def __init__(self,Y0,parameter):
        self.Y0 = Y0  # initial conditions for solving the ODEs
        self.parameter = parameter  # dictionnary of parameters
        self.rates = [] * len(self.Y0) # rates for Gillespie algo
        self.simu_time_MC = []  # where each time step Montecarlo simulation is stored
        self.simu_Y_MC = [] # where each simulation of Montecarlo simulation is stored
        
        
    @staticmethod
    def SK_susceptible_host(y,index,tau,m_mobility,nb_point):
        """Input:
        Nh - total number of host
        T_lh - host life span
        Sh - total number of susceptible host
        Iv - total number of infectious vector
        c_vh - effective contact rate vector to host (transmission rate*biting rate) """
        # allows to retake the 7 variables of the SEIR associated to point index
        #print(y)
        Sh_i, Ih_i, Rh_i = y[3*index:3*(index+1)]
        Nh_i = Sh_i + Ih_i + Rh_i
        OUTFLOW = 0 #where the result dS/dt is stored
        if Nh_i != 0: #ensure that you have people at point index
            for j in range(nb_point): #integrate the distance-dependent term for exposition
                if j != index:
                    # allows to retake the 7 variables of the SEIR associated to point index
                    Sh_j, Ih_j, Rh_j = y[3*j:3*(j+1)]
                    Nh_j = Sh_j+ Ih_j+ Rh_j
                    #contribution of other points to the outglow of Sh_i toward Eh_i
                    OUTFLOW += Ih_j*((m_mobility[j][index]/Nh_j)+(m_mobility[index][j]/Nh_i))
                else:
                    OUTFLOW += (Ih_i/Nh_i)*(m_mobility[index][index]+m_mobility[index][index])
            OUTFLOW *= (Sh_i*tau)
            return -OUTFLOW,OUTFLOW
        else:
            return 0,0


    @staticmethod
    def SK_infected_host(y,index,flow_Sh,gamma,nb_point):
        """Input:
            Eh - total number of exposed host
            T_iit - total number of infectious vector
            Ih - total number of exposed host
            T_id - host infection duration
            T_lh - host life span
            """
        # allows to retake the 7 variables of the SEIR associated to point index
        Sh_i, Ih_i, Rh_i, = y[3*index:3*(index+1)]
        return flow_Sh- Ih_i*gamma,Ih_i*gamma

    
class Mobility_Matrix_Model:
    def __init__(self,Y0,parameter):
        self.Y0 = Y0  # initial conditions for solving the ODEs
        self.parameter = parameter  # dictionnary of parameters
        self.rates = [] * len(self.Y0) # rates for Gillespie algo
        self.simu_time_MC = []  # where each time step Montecarlo simulation is stored
        self.simu_Y_MC = [] # where each simulation of Montecarlo simulation is stored

    @staticmethod
    def MM_system(t,y,tau,gamma,m_mobility,nb_point):
        """Compute the and store the evolution of the system according to the ODEs
        Input:
            t - current time
            y - array of current conditions of the system y(t) (initial conditions when initialized)
            parameters - all the parameters of the system (vectorized parameters i.e. on eby met
        Output: 
            OUTPUT - array of dy(t+delta_t)/dt"""
        OUTPUT = []
        for point in range(nb_point): 
            dSh_dt,flow_Sh = Spatial_Kernel_Model.SK_susceptible_host(y,point,tau,m_mobility,nb_point)
            dIh_dt,dRh_dt = Spatial_Kernel_Model.SK_infected_host(y,point,flow_Sh,gamma,nb_point)
            OUTPUT.append([dSh_dt,dIh_dt,dRh_dt])
        #print("time:",t)
        #print("point:",point)
        #print("flow:", flow_Sh)
        OUTPUT = np.array(OUTPUT).flatten()
        return OUTPUT

    def solve(self,T,t_eval):
        """Compute the solution of the ODEs of the class with scipy.solve_ivp
        Input:
            T - time bounds where the solution is computed
            t_eval - list of times where the ODEs system is evaluated
         Output:
            self.l_t - list of times where the ODEs system is evaluate
            self.Y - solution for each variable in order vars_point0,vars_point1,..."""
        time_span = [min(T),max(T)]
        tau = self.parameter["tau"]
        gamma = self.parameter["gamma"]
        m_mobility = self.parameter["mobility_matrix"]
        Y0_flatten = np.ravel(self.Y0) #the initial conditions are flattened to be used in solve_ivp
        nb_point = len(self.Y0)
        ODEs_system = solve_ivp(Mobility_Matrix_Model.MM_system, time_span, Y0_flatten,args=(tau,gamma,m_mobility,nb_point), t_eval=t_eval,rtol=1e-2)
        print("One run is done")
        #ODEs_system = odeint(Mobility_Matrix_Model.MM_system, y0= Y0_flatten,args=(tau,gamma,m_mobility,nb_point), t=t_eval)
        self.Y = ODEs_system.y
        self.l_t = ODEs_system.t
        return self.l_t,self.Y
    
    def __Gillespie_onestep(self,rng):
        """Compute the stochastic step,"""
        flatten_rate = np.ravel(self.rates) #flatten the self.rates to be able to compare each rate at each point
        sum_rate = np.sum(flatten_rate)
        #print("sum rate ", sum_rate)
        proba_event = flatten_rate / sum_rate #change rate in proba of event
        # print(proba_event)
        time_event = expon.rvs(scale=1 / sum_rate,random_state=rng) #time for soemthing to happen
        #print(time_event)
        index = [i for i in range(len(flatten_rate))]
        rd_index = rng.choice(index, p=proba_event) #choose the event (a certain action at a certain point)
        point = rd_index // len(self.rates[0])  # eucldiean division with number of variables per points to find point
        index_event = rd_index % len(self.rates[0])  # find the event in the point "point"
        return time_event, point, index_event


    def mobility_transmission(self, YT,point):
        # U is the factor dependent of the popualtion of a givn point
        transmission_rate = 0
        m_mobility = self.parameter["mobility_matrix"]
        for j in range(len(YT)):
            Sh_j, Ih_j, Rh_j = YT[j]
            transmission_rate += Ih_j*(m_mobility[j][point]+m_mobility[point][j])
        return transmission_rate
    

    def __update_rate_MM(self, YT):
        # Sh_to_Ih, Ih_to_Rh
        number_event = 2
        self.rates = np.zeros(shape=[len(YT),number_event]) #[0] * len(YT) 
        tau = self.parameter["tau"]
        gamma = self.parameter["gamma"]
        for point in range(len(YT)): # everything except the wolrd patch that is placed at the end 
            Sh_i, Ih_i, Rh_i = YT[point]
            self.rates[point] = [(Sh_i * tau) * self.mobility_transmission(YT,point),Ih_i/gamma]
        return self.rates

    def __update_event_MM(self, YT, point, index_event):
        # The SEIR model + birth-death has 14 differents events whose order is associated with the rates of
        # self.rates. Events 0-1: Sh_to_Ih,Ih_to_Rh
        intermediate_YT = copy.copy(YT)
        if index_event == 0:  # Sh_to_Ih
            YT[point][1] += 1
            YT[point][0] -= 1
        elif index_event == 1:  # Ih_to_Rh
            YT[point][2] += 1
            YT[point][1] -= 1
        if -1 in YT:
            return intermediate_YT #ensure there is no forbidden evetn (negative population)
        else:
            return YT
    
    def simulate_MonteCarlo(self, T,seed):
        t = T[0] #starting time
        YT_MC = [] #solution of Motecarlo simulation for each variable
        T_MC = [] # Montecarlo time
        T_MC.append(T[0])
        step_Y = copy.copy(self.Y0)
        YT_MC.append(copy.copy(step_Y))
        rng = np.random.default_rng(seed)
        while t < T[1]: #run until end time
            self.rates = self.__update_rate_MM(step_Y) #update the rates of each event
            delta_t, point, index_event = self.__Gillespie_onestep(rng) #choose time_step,at which point the event occurs and what occurs
            step_Y = self.__update_event_MM(step_Y, int(point), int(index_event)) #update the states of each variable
            #print(step_Y)
            YT_MC.append(copy.copy(step_Y))
            t += delta_t #update the time
            T_MC.append(t)
        return np.array(T_MC), np.transpose(YT_MC, (2, 1, 0))
    
    def __update_leaping_event_MM(self, YT, point, index_event,event_leap):
        # The SEIR model + birth-death has 14 differents events whose order is associated with the rates of
        # self.rates. Events 0-6: deaths of Sh_i, Eh_i, Ih_i, Rh_i, Sv_i, Ev_i, Iv_i respectively
        # Event 7-8: birth of susceptible host and susceptible vector respectively
        # Event 9-11: host transformations, Sh_to_Eh,Eh_to_Ih,Ih_to_Rh respectively
        # Event 12-13: vector transformations, Sv_to_Ev,Ev_to_Iv respectively
        #
        intermediate_YT = copy.copy(YT)
        if index_event == 0:  # Sh_to_Ih
           YT[point][1] += event_leap
           YT[point][0] -= event_leap
        elif index_event == 1:  # Ih_to_Rh
           YT[point][2] += event_leap
           YT[point][1] -= event_leap
        if np.any(YT <0):
            print(YT)
            return intermediate_YT #ensure there is no forbidden evetn (negative population)
        else:
            return YT
    
    def __Tau_leaping_onestep(self,rng,leap,YT):
        """Compute the stochastic step,"""
        for point in range(len(self.rates)):
            for index in range(len(self.rates[point])):
                event_leap = poisson.rvs(leap*self.rates[point][index],random_state=rng)
                YT = self.__update_leaping_event_MM(YT, point, index, event_leap)
                #otherwise the chnage is no counted
        return YT
    
    def simulate_TauLeaping(self,T,seed,leap):
        t = T[0] #starting time
        YT_MC = [] #solution of Motecarlo simulation for each variable
        T_MC = [] # Montecarlo time
        T_MC.append(T[0])
        step_Y = copy.copy(self.Y0)
        YT_MC.append(copy.copy(step_Y))
        #print(seed)
        rng = np.random.default_rng(seed)
        while t < T[1]: #run until end time
            #print(t)
            self.rates = self.__update_rate_MM(step_Y) #update the rates of each event
            step_Y = self.__Tau_leaping_onestep(rng,leap,step_Y) #update the states of each variable
            #print(step_Y)
            #print(step_Y)
            YT_MC.append(copy.copy(step_Y))
            t += leap #update the time
            T_MC.append(t)
        return np.array(T_MC), np.transpose(YT_MC, (2, 1, 0))
    
class Metapopulation:
    """Create a new object of class Metapopulation """
    def __init__(self,Y0,parameter,model):
        self.Y0 = Y0  # flattening for subsequent integration, when getter reshape
        self.parameter = parameter  # dictionnary of parameters
        self.model = model(Y0,parameter)
        self.l_t = None
        self.Y_t = None
        self.simu_time_MC = None
        self.simu_Y_MC = None
        self.d_var = {} #associate name of variables to a digit that is used everywhere for indexing
        self.d_var["Sh"],self.d_var["Ih"],self.d_var["Rh"] = 0,1,2

    def run_model(self,T,t_eval):
        """Run the ODEs model according to the parameters
        Input:
            T:
            t_eval:
        Output:
            self.l_t
            self.Y_t"""
        self.l_t,self.Y_t = self.model.solve(T,t_eval)
        self.Y_t = self.__reshape_ODE_result()
        return self.l_t,self.Y_t
    
    def run_montecarlo(self,N_sim,T,T_under=None,seed=None,leap=None):
        """N_sim - number of simulation (set to one if multiprocessing)
        T - time interval where to do the simulation
        T_under - all the times where the data needs to be sampled"""
        self.simu_time_MC = [] * N_sim
        self.simu_Y_MC = [] * N_sim
        now = datetime.datetime.now()
        print(now.strftime("%Y-%m-%d %H:%M:%S"))
        for i in range(N_sim): #serial processing
            if leap: # if leap has a value do a Tau Leaping MC
                #print("tau leaping activated")
                a, b = self.model.simulate_TauLeaping(T,seed,leap)
            else:
                a, b = self.model.simulate_MonteCarlo(T,seed)
            self.simu_time_MC.append(a)
            self.simu_Y_MC.append(b)
        if np.array(T_under).any():
            self.simu_time_MC,self.simu_Y_MC =self.__global_undersampling_MonteCarlo(T_under)
        return self.simu_time_MC,self.simu_Y_MC

    def save_multiprocessing_montecarlo(self,l_T_MC,Y_MC):
        self.simu_Y_MC = Y_MC
        self.simu_time_MC = l_T_MC


    def __reshape_ODE_result(self):
        Y_shape = np.zeros(shape=[len(self.Y0),len(self.Y0[0]),len(self.l_t)])
        for index in range(len(Y_shape)):
            Y_shape[index] = self.Y_t[3*index:3*(index+1)]
        self.Y_t = Y_shape
        return self.Y_t
    
    def __get_number_host_ODE(self,which_point):
        Nh = [0]*len(self.Y_t[which_point][0])
        for var_index in range(3): #count jist number of host 0,1,2,3
            a = self.Y_t[which_point][var_index]
            Nh = [Nh[t]+a[t] for t in range(len(self.Y_t[which_point][var_index]))]
        return Nh


    def get_variables(self,which_var,which_point=0):
        """getter to obtain the computed variables
        Input:
            which_var: name of variables (use the d_var dictionnary code), see initializer
            which_point: name of the point where to get the variable"""
        if which_var == "T": # T to extract the time 
            return self.l_t
        elif which_var == "Nh":
            return self.__get_number_host_ODE(which_point)
        else:
            var_index = self.d_var[which_var]
            return self.Y_t[which_point][var_index]
        
        
    def __get_number_host_Montecarlo(self,which_point):
        Nh = [0]*len(self.simu_Y_MC)
        for i in range(len(self.simu_Y_MC)):
            b = [0]*len(self.simu_Y_MC[i][0][0])
            for var_index in range(3):
                a = self.simu_Y_MC[i][var_index][which_point]
                #print(b)
                b = [b[t]+a[t] for t in range(len(self.simu_Y_MC[i][var_index][which_point]))]
            Nh[i] = b
        return Nh
        
            
    def get_variables_Montecarlo(self,which_var,which_point=0):
        if which_var =="T":
            return [self.simu_time_MC[i][0] for i in range(len(self.simu_time_MC))]
        elif which_var == "Nh":
            return self.__get_number_host_Montecarlo(which_point)
        else:
            var_index = self.d_var[which_var]
            return [self.simu_Y_MC[i][var_index][which_point] for i in range(len(self.simu_Y_MC))]
    
    def __global_undersampling_MonteCarlo(self,T_target):
            l_sample = np.zeros(shape=[len(self.simu_Y_MC),len(self.simu_Y_MC[0]),len(self.simu_Y_MC[0][0]),len(T_target)])
            l_t_sample = np.zeros(shape=[len(self.simu_time_MC),len(T_target)])
            i = 0
            for n in range(len(self.simu_Y_MC)):
                l_t_sample[n] = T_target
                for t in range(len(T_target)):
                    while self.simu_time_MC[0][i] < T_target[t]:
                        i += 1
                    for index in range(len(self.simu_Y_MC[0])):
                        for point in range(len(self.simu_Y_MC[0][0])):
                            l_sample[n][index][point][t] = self.simu_Y_MC[n][index][point][i]
            #print(T_target)
            return l_t_sample,l_sample
        
    def undersampling_MonteCarlo(self,T_target,which_var,which_point):
        l_sample = np.zeros(shape=[len(T_target),len(self.simu_Y_MC)])
        var_index = self.d_var[which_var]
        for n in range(len(self.simu_Y_MC)):
            i = 0
            for t in range(len(T_target)):
                while self.simu_time_MC[n][0][i] < T_target[t]:
                    i += 1
                l_sample[t][n] = self.simu_Y_MC[n][var_index][which_point][i]
        return l_sample

    def average_MonteCarlo(self,T_target,which_var,which_point,undersample=False):
        """Not very well coded the else here"""
        if undersample:
            l_sample = self.undersampling_MonteCarlo(T_target,which_var,which_point)
        else:
            l_sample = np.zeros(shape=[len(T_target),len(self.simu_Y_MC)])
            if which_var == "Nh":
                l_sample = self.__get_number_host_Montecarlo(which_point)
                l_sample = np.transpose(l_sample,(1,0))
            elif which_var == "Nv":
                l_sample = self.__get_number_vector_Montecarlo(which_point)
                l_sample = np.transpose(l_sample,(1,0))
            elif which_var == "ratio_Nh":
                var_index = self.d_var["Ih"]
                for n in range(len(self.simu_Y_MC)):
                        for t in range(len(T_target)):
                            l_sample[t][n] = self.simu_Y_MC[n][var_index][which_point][t]
                l_Nh = self.__get_number_host_Montecarlo(which_point)
                l_Nh = np.transpose(l_Nh,(1,0))
                for t in range(len(l_sample)):
                    #print("l_sqmple", l_sample[t])
                    #print("l_Nh",l_Nh[t])
                    l_sample[t] = np.array([l_sample[t][n]/l_Nh[t][n] for n in range(len(l_sample[t]))])
            else:
                var_index = self.d_var[which_var]
                for n in range(len(self.simu_Y_MC)):
                    for t in range(len(T_target)):
                        l_sample[t][n] = self.simu_Y_MC[n][var_index][which_point][t]
        l_mean = np.zeros(shape=len(l_sample))
        l_std = np.zeros(shape=len(l_sample))
        N_sim = len(self.simu_time_MC)
        for t in range(len(l_sample)):
            l_mean[t] = np.mean(l_sample[t])
            l_std[t] = np.std(l_sample[t])
        l_std = [l_std[i] / np.sqrt(N_sim) for i in range(len(l_std))]
        return l_mean,l_std