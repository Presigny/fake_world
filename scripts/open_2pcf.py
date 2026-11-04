import sys
from pathlib import Path
path = Path().cwd().parent / "src" #to add the src directory to the path regognized by Python
sys.path.append(str(path))
import methods_two_point_correlation as mtpc
import save_load_pickle as slp
import argparse
import numpy as np
from scipy.optimize import curve_fit
    
# Bookkeeping of important path in the project
path_data = Path("/media/utente/LaCie/fake_world_data/")
data_dir = path_data#Path().cwd().parent/Path("data")
data_city = data_dir / "cities"
data_map = data_dir / "map"
data_2pcf = data_dir/"2pcf"
out_dir = path_data #Path().cwd().parent/Path("out")
plot_dir = out_dir / Path("plot_powerlaw")
##Create directories and path to save data


# parser = argparse.ArgumentParser(description="Fit a power law to the 2pcf")

# parser.add_argument("--name", type=str, default="Sweden")
# parser.add_argument("--threshold", type=int, default=1)
# parser.add_argument("--N_run", type=int, default=10)
# parser.add_argument("--size", type=int, default=1000)
# parser.add_argument("--k", type=int, default=3)
# parser.add_argument("--rmax", type=float, default=None)
# parser.add_argument("--scale", type=str, default="log", choices=["log", "linear"])
# parser.add_argument("--nbins", type=int, default=20)

# args = parser.parse_args()
name = "Austria"
threshold = 1
N_run = 10
size = 4500
k = 1
rmax = None
scale ="log"
nbins = 20

name_load = data_2pcf/ Path(name) / Path(f"{threshold}_{N_run}_{size}_{k}__{rmax}_{scale}_{nbins}")
name_save = data_2pcf/ Path(name) / Path(f"powerlawfit_{threshold}_{N_run}_{size}_{k}__{rmax}_{scale}_{nbins}")
d_save = slp.load_results(name_load)
r_edges = d_save["r_edges"] 
l_xi = d_save["xi"]
distance_distribution = d_save["distance_distribution"] 

a,b = 0,len(l_xi) #the two extremal index between which fitting the SP"
xi = np.mean(l_xi,axis=0)
error = np.std(l_xi,axis=0) #vertical error due to statistical variance
print(xi,error)
xi_reduced = xi[a:b]
r_reduced = r_edges[a:b]
error_reduced = error[a:b]
popt,pcov = curve_fit(mtpc.power_law,r_reduced,xi_reduced,sigma=error_reduced)
# popt[0] is fitted gamma, popt[1] is fitted r0
#pcov is the covariance matrix
d_save = {}
d_save["r_edges"] = r_edges
d_save["xi"] = l_xi
d_save["distance_distribution"] = distance_distribution
d_save["fit_SP"] = popt
d_save["covariance_fit"] = pcov
#d_save["number_points"] = len(gdf_projected)
slp.save_results(name_save,d_save)

mtpc.plot_2pcf_with_fit(name_save,plot_dir,popt,pcov,scale)