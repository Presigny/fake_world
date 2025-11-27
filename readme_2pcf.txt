Author: Charley Presigny
This file is intended to explain the use of the notebook /notebooks/2pcf.ipynb.
The notebook computes the two point correlaton function (2pcf) for any given system under the form of a 
Geopandas dataframe with CRS (coordinate reference system) with the physical border of the system. 
Format:
Geopandas dataframe for data loads csv files.
Geopandas dataframe for physical borders loads geojson files.


IMPORTANT NOTE: The way we bin the distances in our system has a big impact on the outcome of the code.
A linear scale will give emphasis on the bigger scale of the system.
A logscale will give the same importance to eahc scale in the system.

PRO TIP: I think we should do both because I guess the linear scale could be more adapted to have a fitting 
distance distribution (what we are ultimately interested in) with linear bins and similarly with logscale  

PRO TIP:  20-30 bins seems nice

PART I: Compute the 2pcf for random catalog

In order to have unbiased estimation of the 2pcf, we need to compute the 2pcf (using the Landy-Szalay estimator) for a random system
within the border of the system we consider and with the same number of points. It is a sanity check to see
if the split random catalog (see Keihanen et al. 2019) that we build is robust enough. It is the case when 
the 2pcf for the random system is more or less zero everywhere (lets consider pcf(r) <0.05 a good value)
PRO TIP: Take a combination N_run*size > 20-50* number of points in the system

IMPORTANT NOTE: In all the plots (except for fitting SP model cf Part III), the first bins is not represented 
because it includes the smaller possible scale.

PART II: Compute the 2pcf for the system

In this part we compute the 2pcf for the system of interests. 
So far with the city dataset I observe that the 2pcf has a random behavior (oscillating around zero) for high values of distance range.

PART III: Fit the 2pcf to find the gamma of the Soneira-Peebles

Fit the curve using the variance through a XXX method. Possibility of reducing the interval of fit.

PRO TIP: Fitting only on the positive part of the curve considering that the oscillation around zero represent
a scale at which there is no more effect of the SP model

PART IV: Estimate the best eta to be tested

By using a KMeans clustering and the silhouette score, this part of the notebook gives the 4 best partitions
of the system that can be used as a values for eta.

PART V: Estimation of the correlation dimension

The correlation dimension is something very close to the measure of uniformity but the values seem very noisy
(method of fit identicla to the fit of gamma for SP). Just put it here for fun !
