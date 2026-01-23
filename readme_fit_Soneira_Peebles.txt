Author: Charley Presigny
This file is intended to explain the use of the notebook /notebooks/fit_Soneira_Peebles.ipynb.
The notebook finds the parameters of the Soneira-Peebles + the proportion of random points needed to generate synthetic data
whose 2pcf fits the one of the realted empirical system.

PART 0: Parameters

It loads the files related to a given system as well as the results obtained by 2pcf.!pynb
dimish_radius: this variable is to be used <1 if the Soneira-Peebles does not fit into the border of the system. First try with it equal to 1.
Try to not diminish it too much otherwise the SP part is concentrated on a smaller part of the system.


PART I: Select the best combination of parameters
How does the fir work:
1- For each value of cluster = eta in the l_cluster file, the algorithm finds the value L and the number of random points 
that fit the empirical 2pcf. FOr that it generates "iteration_pure_SP" time a SOneira-Peebles with the paramter eta and L
in order to find the right correction coefficient that is the mean of all the try that is then related to the number of random point needed to
make this correction
2- Then, for each combination of parameter found by the system, "number_of_try_for_each_parameter" 2pcf are produced and the one htat fir the best 
the empirical distribution (as assessed by an euclidean cost fucntion) is selected to be plotted and the value of the cost function saved.


PART III: Compute the Kullback-Leibler divergence
The set of parameters that minimize the cost fucntion are selected to compute the KL divergence.
n_iteration SP+random models (and random models) are produced and their distance distribution are compared with the related empirical system.
Two types of KL are computed: one on all the distribution, one reduced in the first "n_short" bins of the distributions



PART IV: Save the fitted parameters

Self-explanatory + Plot of 2pcf with optimal paraemters + plot of the point distribution on the maps
