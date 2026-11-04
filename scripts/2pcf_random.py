import sklearn.cluster as sk
import itertools
import sys
import matplotlib.pyplot as plt
#from kneed import KneeLocator
from sklearn.metrics import silhouette_score
from pathlib import Path
import geopandas as gpd
path = Path().cwd().parent / "src" #to add the src directory to the path regognized by Python
sys.path.append(str(path))
import methods_two_point_correlation as mtpc
import save_load_pickle as slp
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.optimize import curve_fit


def plot_2pcf(load_file,scale):
    """Plot and save the 2pcf store in load_file"""
    i=1
    name = load_file.parent.name
    fig, ax = plt.subplots()
    sns.set(style="whitegrid")
    d = slp.load_results(load_file)
    r_edges, l_xi = d["r_edges"],d["xi"]
    xi = np.mean(l_xi,axis=0)
    error = np.std(l_xi,axis=0)
    if len(xi) != len(r_edges): #when we choose a rmax define by user the code put an extra category above
        xi = xi[0:-1]
        error = error[0:-1]
    color = sns.color_palette("viridis")[0]
    r_edges = r_edges[1:]
    xi = xi[1:]
    error = error[1:]
    end = np.zeros([1])
    end[0] = 1.1*r_edges[-1]
    r_edges_zero = np.concatenate((r_edges,end))
    width = np.diff(r_edges_zero) #need to find a way to encode properly the first alignement
    ax.bar(r_edges,xi,width=width,align="center",edgecolor="black",color=color,alpha=0.5)
    plt.errorbar(r_edges, xi, yerr=error, fmt="o", color="r",ms=1)
    title = "2 points correlation function "+name
    plt.title(title)
    plt.ylabel(r"$\xi(r)$")
    plt.xlabel(r"$r$ (meters)")
    if scale == "lin":
        plt.xscale("linear")
    else:
        plt.xscale(scale)
    #plt.yscale("log")
    #plt.legend()
    plt.savefig(plot_dir / title, format="png",transparent=True)
    plt.show()
    
# Bookkeeping of important path in the project
path_data = Path("/media/utente/LaCie/fake_world_data/")
data_dir = path_data#Path().cwd().parent/Path("data")
data_city = data_dir / "cities"
data_map = data_dir / "map"
out_dir = path_data #Path().cwd().parent/Path("out")
plot_dir = out_dir / Path("plot")



name = "Poland" #name of the system considered
name_random = f"Random{name}" #How the name of the random catalo will be saved
threshold = 10000 # minimal number of inhabitants for the city to be considered
N_run = 10 # number of random catalog generated for computing the 2PCF
size = 5000 #number of points each random catalog has
k = 10  #number of time the 2PCF computing process is repeated to estimate variance
rmax = None#3e5 #maximal distance between two points considered, if None, taken the max from data
scale = "log" #scale at which to bin the data, recommended "log"
nbins = 20 # number of bins to consider in the computation of 2PCF (betweem rmin and rmax)


##Preprocessing
crs = mtpc.crs_selector(name) # Coordinate reference system of the country considered
path_city = data_city / f"{name}_cities.csv" #Path of the cities 
path_border = data_map / f"{name}.geojson" # Path of the multipolygon of the territory of the country
gdf_city = mtpc.load_df_to_gdf(path_city,threshold) # Geopandas of the dataset
print("number of cities considered: ",len(gdf_city))
gdf_edge = gpd.read_file(path_border) #Geopandas of the polygon border of the dataset
gdf_projected = gdf_city.to_crs(crs) # Projection in the right coordinate system
coord = gdf_projected.get_coordinates() # Coordinates of the points in dataset


##Create directories and path to save data
path_save_random = out_dir/Path("2pcf_random") /Path(f"{name_random}/")
path_save_random.mkdir(parents=True,exist_ok=True)

name_save_random = path_save_random/Path(f"{threshold}_{N_run}_{size}_{k}_{rmax}_{scale}_{nbins}")
# pay attention to waht each number means 
gdf_projected_random= mtpc.generate_random_point(gdf_edge,len(gdf_city),crs,check_gpd=True)
r_edges,l_xi = mtpc.PCF_with_variance(gdf_projected_random,gdf_edge,crs,N_run,size,k,rmax,scale,nbins=nbins)
distance_distribution = mtpc.compute_DD(gdf_projected_random)
#store the random test
d_save = {}
d_save["r_edges"] = r_edges
d_save["xi"] = l_xi
d_save["distance_distribution"] = distance_distribution
slp.save_results(name_save_random,d_save)

plot_2pcf(name_save_random,scale)