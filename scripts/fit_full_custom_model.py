import sys
from pathlib import Path
path = Path().cwd().parent / "src" #to add the src directory to the path regognized by Python
sys.path.append(str(path))
import methods_two_point_correlation as mtpc
import fit_custom_Soneira_Peebles as fcsp
import save_load_pickle as slp
import argparse
import numpy as np
import geopandas as gpd
import os
    
# Bookkeeping of important path in the project
path_data = Path("/media/utente/LaCie/fake_world_data/")
data_dir = path_data#Path().cwd().parent/Path("data")
data_city = data_dir / "cities"
data_map = data_dir / "map"
data_2pcf = data_dir/"2pcf"
data_fit = data_dir /"fit"
out_dir = path_data #Path().cwd().parent/Path("out")
plot_dir = out_dir / Path("plot_powerlaw")
##Create directories and path to save data


parser = argparse.ArgumentParser(description="Fit a power law to the 2pcf")

parser.add_argument("--name", type=str, default="Austria")
parser.add_argument("--threshold", type=int, default=1)
parser.add_argument("--N_run", type=int, default=10)
parser.add_argument("--size", type=int, default=4500)
parser.add_argument("--k", type=int, default=3)
parser.add_argument("--rmax", type=float, default=None)
parser.add_argument("--scale", type=str, default="log", choices=["log", "linear"])
parser.add_argument("--nbins", type=int, default=20)

args = parser.parse_args()

name = args.name
threshold = args.threshold
N_run = args.N_run
size = args.size
k = args.k
rmax = args.rmax
scale = args.scale
nbins = args.nbins

diminish_radius = 1 # multiplication factor for the maximum radius, 
#to be lowered if the SP does not fit inside the considered border quick enough

#Load the polygon file for border
path_border = data_map / f"{name}.geojson"
gdf_edge = gpd.read_file(path_border)

name_load = data_2pcf/ Path(name) / Path(f"clusters_{threshold}_{N_run}_{size}_{k}__{rmax}_{scale}_{nbins}")
name_save = data_fit/ Path(name) / Path(f"model_{threshold}_{N_run}_{size}_{k}__{rmax}_{scale}_{nbins}")
d_load = slp.load_results(name_load)
r_edges = d_load["r_edges"] 
l_xi = d_load["xi"]
distance_distribution = d_load["distance_distribution"] 
l_cluster = d_load["clustering"]
N = d_load["number_points"] #number of point sin the dataset
gamma, r0 = d_load["fit_SP"] #fitted gamma and R0
crs = mtpc.crs_selector(name) #cooridnate reference system associated with the datset
gdf_edge = gdf_edge.to_crs(crs) # Polygon of the border of the country

eta = l_cluster[0] #density, number of circles by steps
l = eta**(1/(2-gamma)) #fraction of space covered by the circles
L = int(np.log(N)/np.log(eta)+1)-1 #size of circles, number of steps
R = 0.5*gdf_edge.length[0]/(2*np.pi)*diminish_radius # radius max
min_rad = R/(l**(L-1))
rmin= np.min(r_edges)


iteration_pure_SP = 1 #number of try to set the basis of the SOneira-Peebles 2pcf
number_of_try_for_each_parameter = 1 #number of generated 2pcf SP+random by set of paraemters to compare with the reference

save_dist,save_L,save_random = fcsp.find_the_fit_parameters_SP_random(r_edges,l_xi,l_cluster,N,R,gdf_edge,crs,rmax,number_of_try_for_each_parameter,iteration_pure_SP,gamma,N_run,size,k,scale,nbins,rmin,name)

d_load["save_dist"] = save_dist
d_load["L"] = save_L
d_load["eta"] = l_cluster
d_load["save_random"] = save_random
d_load["iteration_pure_SP"] = iteration_pure_SP
d_load["number_of_try_for_each_parameter"] = number_of_try_for_each_parameter
if not(name_save.parent):
    os.mkdir(name_save.parent)
slp.save_results(name_save,d_load)