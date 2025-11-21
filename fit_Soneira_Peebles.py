import geopandas as gpd
import sys
from pathlib import Path
path = Path().cwd() / "src" #to add the src directory to the path regognized by Python
sys.path.append(str(path))
import methods_two_point_correlation as mtpc
import save_load_pickle as slp
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.spatial import distance
import matplotlib.pyplot as plt
import sklearn.cluster as sk
import sys
from sklearn.metrics import silhouette_score
from shapely.geometry import Polygon

def soneira_peebles_model(lamb,eta,L,R,erase_nodes=False):
    position = [(0,0)]
    final_position = []
    rng = np.random.default_rng()
    for i in range(L):
        l_pos = []
        for x,y in position:
            r = (R/(lamb**i))*np.random.uniform(low=0, high=1, size=eta)  # radius
            theta = np.random.uniform(low=0, high=2*np.pi, size=eta)  # angle
            l_x_rand = r * np.cos(theta)
            l_y_rand = r * np.sin(theta)
            #l_random = (R/(lamb**i))*rng.random((eta,2))
            for x_rand,y_rand in zip(l_x_rand,l_y_rand):
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
    return position
from shapely.geometry import Point
def soneira_peebles_border(lamb,eta,L,R,gdf_edge,erase_nodes=False):
    position = gdf_edge.centroid.get_coordinates().to_numpy()
    final_position = []
    rng = np.random.default_rng()
    for i in range(L):
        l_pos = []
        for x,y in position:
            r = (R/(lamb**i))*np.random.uniform(low=0, high=1, size=eta)  # radius
            theta = np.random.uniform(low=0, high=2*np.pi, size=eta)  # angle
            l_x_rand = [r[i] * np.cos(theta[i]) for i in range(eta)]
            l_y_rand = [r[i] * np.sin(theta[i]) for i in range(eta)]
            #l_random = (R/(lamb**i))*rng.random((eta,2))
            for x_rand,y_rand in zip(l_x_rand,l_y_rand):
                #s = geopandas.GeoSeries(point)
                l_pos.append((x-x_rand,y-y_rand))
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
    return position

def plot_2pcf(r_edges,l_xi):
    fig, ax = plt.subplots()
    sns.set(style="whitegrid")
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
    plt.xscale("log")
    #plt.yscale("log")
    #plt.legend()
    #plt.savefig(plot_dir / title, format="png",transparent=True)
    plt.show()

def plot_two_2pcf(r_edges,l_xi,l_xi_SP):
    fig, ax = plt.subplots()
    sns.set(style="whitegrid")
    xi = np.mean(l_xi,axis=0)
    error = np.std(l_xi,axis=0)
    xi_SP = np.mean(l_xi_SP,axis=0)
    error = np.std(l_xi,axis=0)
    if len(xi) != len(r_edges): #when we choose a rmax define by user the code put an extra category above
        xi = xi[0:-1]
        error = error[0:-1]
        xi_SP = xi_SP[0:-1]
    color = sns.color_palette("viridis")[0]
    color_SP = sns.color_palette("viridis")[1]
    r_edges = r_edges[1:]
    xi = xi[1:]
    xi_SP = xi_SP[1:]
    error = error[1:]
    end = np.zeros([1])
    end[0] = 1.1*r_edges[-1]
    r_edges_zero = np.concatenate((r_edges,end))
    width = np.diff(r_edges_zero) #need to find a way to encode properly the first alignement
    ax.bar(r_edges,xi,width=width,align="center",edgecolor="black",color=color,alpha=0.5,label=f"{name}")
    ax.bar(r_edges,xi_SP,width=width,align="center",edgecolor="black",color=color_SP,alpha=0.5,label="SP")
    plt.errorbar(r_edges, xi, yerr=error, fmt="o", color="r",ms=1)
    title = "2 points correlation function "+name
    plt.title(title)
    plt.ylabel(r"$\xi(r)$")
    plt.xlabel(r"$r$ (meters)")
    plt.xscale("log")
    #plt.yscale("log")
    plt.legend()
    #plt.savefig(plot_dir / title, format="png",transparent=True)
    plt.show()

path = Path().cwd()
sys.path.append(str(path))
data_dir = Path("data/cities")
map_dir = Path("data/map")

# 
# path_city = Path.cwd()/ data_dir / f"{name}_cities.csv"

# gdf_projected = gdf_city.to_crs(crs)
# coord = gdf_projected.get_coordinates()

name = "Belgium"
path_border = Path.cwd() / map_dir / f"{name}.geojson"
# gdf_city = mtpc.load_df_to_gdf(path_city,threshold)
gdf_edge = gpd.read_file(path_border)

out_dir = Path.cwd() /Path(f"out/2pcf/{name}")
file = out_dir / "1_5_7000_3__100000.0_log_20"
dico_results = slp.load_results(file)
r_edges = dico_results["r_edges"]
gamma, r0 = dico_results["fit_SP"]
l_xi = dico_results["xi"]
country_distribution = dico_results["distance_distribution"]
l_cluster = dico_results["clustering"]
N = dico_results["number_points"]
crs = mtpc.crs_selector(name)
gdf_edge = gdf_edge.to_crs(crs)

eta = 3#l_cluster[] #density, number of circles by steps
l = eta**(1/(2-gamma)) #fraction of space covered by the circles
L = int(np.log(N)/np.log(eta)+1) #size of circles, number of steps

print("gamma", gamma)
print("fractal dimension",np.log(eta)/np.log(l))
print("l = ", l)
print("L = ", L)
R = gdf_edge.length[0]/(2*np.pi)#np.max(country_distribution) # radius max
#R = np.min(r_edges)*(l**(L-1))
min_rad = R/(l**(L-1))
nbins=20
rmin=0
N_run= 5
size= 3000
print("mimimum radius",min_rad )
position = soneira_peebles_model(l,eta,L,1)
#position = soneira_peebles_border(l,eta,L,R,gdf_edge)
distance_distribution = distance.pdist(position)

d = {}
d["x"] = position.T[0]
d["y"] = position.T[1]
#df = pd.DataFrame((position))
gdf = gpd.GeoDataFrame(
      d, geometry=gpd.points_from_xy(position.T[0], position.T[1]),crs=crs)

convex_hull_polygon = gdf.union_all()
polygon = convex_hull_polygon.convex_hull
geoserie = gpd.GeoSeries(polygon)
squares_gdf=gpd.GeoDataFrame({'geometry': geoserie, 'df':[1]},crs=crs)
# polygon = convex_hull_polygon#.convex_hull
# xmax = np.max(position.T[0])
# xmin = np.min(position.T[0])
# ymin = np.min(position.T[1])
# ymax = np.max(position.T[1])
              
# # 1. Define coordinates for a square
# coords = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]

# # 2. Create a Shapely Polygon
# square = Polygon(coords)

# # 3. Create a GeoSeries from the polygon
# squares_gs = gpd.GeoSeries([square])

# # 4. Create a GeoDataFrame from the GeoSeries
# squares_gdf = gpd.GeoDataFrame({'geometry': squares_gs},crs=crs)
fig,ax = plt.subplots()
squares_gdf.plot(ax=ax)
#gdf_edge.plot(ax=ax,color="black")
gdf.plot(ax=ax,color="red",markersize=0.5)
# N_SP = eta**L
k=3
rmax = 1/(np.max(country_distribution)/1e5)#np.max(r_edges)
rmin = 0#np.min(r_edges)
scale="log"
# sns.histplot(distance_distribution)
# gdf_projected_random= mtpc.generate_random_point(squares_gdf,N_SP,crs,check_gpd=True)
# r_edges_random_SP,l_xi_random_SP = mtpc.PCF_with_variance(gdf_projected_random,squares_gdf,crs,N_run,size,k,rmax,scale,nbins=nbins)
# distance_distribution_random = mtpc.compute_DD(gdf_projected_random)
# plot_2pcf(r_edges_random_SP,l_xi_random_SP)

r_edges_SP,l_xi_SP = mtpc.PCF_with_variance_SP(gdf,squares_gdf,crs,N_run,size,k,rmax,scale,nbins=nbins,rmin=rmin)
plot_2pcf(r_edges_SP,l_xi_SP)
plot_2pcf(r_edges,l_xi)
#plot_two_2pcf(r_edges,l_xi,l_xi_SP)
from scipy.optimize import curve_fit
def power_law(r, gamma,r0):
     model = (r/r0)**(-gamma)
     return model
a,b = 0,19
xi_SP = np.mean(l_xi_SP,axis=0)
error = np.std(l_xi_SP,axis=0)
xi_reduced = xi_SP[a:b]
r_reduced = r_edges_SP[a:b]
error_reduced = error[a:b]
popt,pcov = curve_fit(power_law,r_reduced,xi_reduced,sigma=error_reduced,p0=(gamma,0.1))
print(popt)
print(np.sqrt(pcov[0][0]))
# position = (r0/popt[1])*position

# d = {}
# d["x"] = position.T[0]
# d["y"] = position.T[1]
# #df = pd.DataFrame((position))
# gdf = gpd.GeoDataFrame(
#       d, geometry=gpd.points_from_xy(position.T[0], position.T[1]),crs=crs)
# convex_hull_polygon = gdf.union_all()
# polygon = convex_hull_polygon.convex_hull
# geoserie = gpd.GeoSeries(polygon)
# squares_gdf=gpd.GeoDataFrame({'geometry': geoserie, 'df':[1]},crs=crs)
# xmax = np.max(position.T[0])
# xmin = np.min(position.T[0])
# ymin = np.min(position.T[1])
# ymax = np.max(position.T[1])
              
# # # 1. Define coordinates for a square
# # coords = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]

# # # 2. Create a Shapely Polygon
# # square = Polygon(coords)

# # # 3. Create a GeoSeries from the polygon
# # squares_gs = gpd.GeoSeries([square])

# # # 4. Create a GeoDataFrame from the GeoSeries
# # squares_gdf = gpd.GeoDataFrame({'geometry': squares_gs},crs=crs)
# fig,ax = plt.subplots()
# squares_gdf.plot(ax=ax)
# gdf.plot(ax=ax,color="red",markersize=0.5)
# N_SP = eta**L
# k=3
# rmax = np.max(r_edges)
# rmin = np.min(r_edges)
# scale="lin"
# # sns.histplot(distance_distribution)
# # gdf_projected_random= mtpc.generate_random_point(squares_gdf,N_SP,crs,check_gpd=True)
# # r_edges_random_SP,l_xi_random_SP = mtpc.PCF_with_variance(gdf_projected_random,squares_gdf,crs,N_run,size,k,rmax,scale,nbins=nbins)
# # distance_distribution_random = mtpc.compute_DD(gdf_projected_random)
# # plot_2pcf(r_edges_random_SP,l_xi_random_SP)

# r_edges_SP,l_xi_SP = mtpc.PCF_with_variance(gdf,squares_gdf,crs,N_run,size,k,rmax,scale,nbins=nbins,rmin=rmin)
# plot_2pcf(r_edges_SP,l_xi_SP)
# plot_two_2pcf(r_edges,l_xi,l_xi_SP)
# #distance_distribution = mtpc.compute_DD(gdf_projected)
# ##generate a Soenir
# from scipy.optimize import curve_fit
# def power_law(r, gamma,r0):
#      model = (r/r0)**(-gamma)
#      return model
# a,b = 1,19
# xi_SP = np.mean(l_xi_SP,axis=0)
# error = np.std(l_xi_SP,axis=0)
# xi_reduced = xi_SP[a:b]
# r_reduced = r_edges[a:b]
# error_reduced = error[a:b]
# popt,pcov = curve_fit(power_law,r_reduced,xi_reduced,sigma=error_reduced,p0=(gamma,r0))
# print(popt)