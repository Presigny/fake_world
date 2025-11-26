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
from scipy.stats import entropy

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
            l_point = [Point((x_rand,y_rand)) for x_rand,y_rand in zip(l_x_rand,l_y_rand)] 
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
    #l_point = [Point(position[i]) for i in range(len(position))]
    s = gpd.GeoDataFrame(
          position, geometry=gpd.points_from_xy(position.T[0], position.T[1]),crs=crs)
    points_with_join = gpd.sjoin(s, gdf_edge, how="left", predicate="within")
    points_with_join['is_inside'] = points_with_join['index_right'].notna()
    point_outside = points_with_join[points_with_join["is_inside"] == False]
    return position,point_outside

def check_inside(lamb,eta,L,R,gdf_edge,erase_nodes=False):
    position,point_outside = soneira_peebles_border(lamb,eta,L,R,gdf_edge,erase_nodes)
    while len(point_outside) >= 1:
        print("Entering the loop")
        position,point_outside=soneira_peebles_border(lamb,eta,L,R,gdf_edge,erase_nodes)
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

def compare_linear_distribution(n_iteration,l,eta,L,R,gdf_edge,crs,number_point,random_point,erase_nodes,k,rmax,scale,nbins):
    KL_distanceSP = 1000
    KL_distance_random =1000
    positionSP,position_random = 0,0
    for i in range(n_iteration):
        position = check_inside(l,eta,L,R,gdf_edge,erase_nodes)
        d = {}
        d["x"] = position.T[0]
        d["y"] = position.T[1]
        #df = pd.DataFrame((position))
        gdf = gpd.GeoDataFrame(
              d, geometry=gpd.points_from_xy(position.T[0], position.T[1]),crs=crs)
        gdf_random_square = gdf_edge.sample_points(random_point)
        gdf_random_points = gpd.GeoDataFrame(gdf_random_square.explode().get_coordinates(),geometry=gdf_random_square.explode(),crs=crs)
        gdf = gpd.GeoDataFrame(pd.concat( [gdf,gdf_random_points], ignore_index=True))
        gdf_projected_random= mtpc.generate_random_point(gdf_edge,number_point,crs,check_gpd=True)
        r_edges_random_SP,l_xi_random_SP = mtpc.PCF_with_variance(gdf_projected_random,gdf_edge,crs,N_run,size,k,rmax,scale,nbins=nbins)
        distance_distribution_random = mtpc.compute_DD(gdf_projected_random)
        distance_distribution_SP = mtpc.compute_DD(gdf)
        KL_distanceSP_inter = entropy(country_distribution+1,distance_distribution_SP+1)
        KL_distance_random_inter = entropy(country_distribution+1,distance_distribution_random+1)
        if KL_distanceSP_inter <KL_distanceSP:
            KL_distanceSP = KL_distanceSP_inter 
        if KL_distance_random_inter <KL_distance_random:
           KL_distance_random= KL_distance_random_inter
    print(KL_distanceSP,KL_distance_random)
    return 0

def compare_linear_distribution(n_iteration,l,eta,L,R,gdf_edge,crs,number_point,random_point,erase_nodes,k,rmax,scale,nbins):
    KL_distanceSP = 1000
    KL_distance_random =1000
    nbins=20
    r_bins = np.logspace(100,np.max(country_distribution)+1,nbins)
    country_distribution_log = mtpc.binning_data(country_distribution,nbins,r_bins)
    positionSP,position_random = 0,0
    for i in range(n_iteration):
        position = check_inside(l,eta,L,R,gdf_edge,erase_nodes)
        d = {}
        d["x"] = position.T[0]
        d["y"] = position.T[1]
        #df = pd.DataFrame((position))
        gdf = gpd.GeoDataFrame(
              d, geometry=gpd.points_from_xy(position.T[0], position.T[1]),crs=crs)
        gdf_random_square = gdf_edge.sample_points(random_point)
        gdf_random_points = gpd.GeoDataFrame(gdf_random_square.explode().get_coordinates(),geometry=gdf_random_square.explode(),crs=crs)
        gdf = gpd.GeoDataFrame(pd.concat( [gdf,gdf_random_points], ignore_index=True))
        gdf_projected_random= mtpc.generate_random_point(gdf_edge,number_point,crs,check_gpd=True)
        #r_edges_random_SP,l_xi_random_SP = mtpc.PCF_with_variance(gdf_projected_random,gdf_edge,crs,N_run,size,k,rmax,scale,nbins=nbins)
        distance_distribution_random = mtpc.compute_DD(gdf_projected_random)
        distance_distribution_SP = mtpc.compute_DD(gdf)
        distance_distribution_SP = mtpc.binning_data(distance_distribution_SP,nbins,r_bins)[0:nbins]
        distance_distribution_random = mtpc.binning_data(distance_distribution_random,nbins,r_bins)[0:nbins]
        KL_distanceSP_inter = entropy(country_distribution+1,distance_distribution_SP+1)
        KL_distance_random_inter = entropy(country_distribution+1,distance_distribution_random+1)
        if KL_distanceSP_inter <KL_distanceSP:
            KL_distanceSP = KL_distanceSP_inter 
        if KL_distance_random_inter <KL_distance_random:
           KL_distance_random= KL_distance_random_inter
    print(KL_distanceSP,KL_distance_random)
    return 0

def compare_log_distribution(n_iteration,l,eta,L,R,gdf_edge,crs,number_point,random_point,erase_nodes,k,rmax,scale,nbins,country_distribution):
    KL_distanceSP = 1000
    KL_distance_random =1000
    positionSP,position_random = 0,0
    nbins=20
    r_bins = np.logspace(np.log10(100),np.log10(np.max(country_distribution)+1),nbins)
    country_distribution_log = mtpc.binning_data(country_distribution,nbins,r_bins)
    for i in range(n_iteration):
        position = check_inside(l,eta,L,R,gdf_edge,erase_nodes)
        d = {}
        d["x"] = position.T[0]
        d["y"] = position.T[1]
        #df = pd.DataFrame((position))
        gdf = gpd.GeoDataFrame(
              d, geometry=gpd.points_from_xy(position.T[0], position.T[1]),crs=crs)
        gdf_random_square = gdf_edge.sample_points(random_point)
        gdf_random_points = gpd.GeoDataFrame(gdf_random_square.explode().get_coordinates(),geometry=gdf_random_square.explode(),crs=crs)
        gdf = gpd.GeoDataFrame(pd.concat( [gdf,gdf_random_points], ignore_index=True))
        gdf_projected_random= mtpc.generate_random_point(gdf_edge,number_point,crs,check_gpd=True)
        #r_edges_random_SP,l_xi_random_SP = mtpc.PCF_with_variance(gdf_projected_random,gdf_edge,crs,N_run,size,k,rmax,scale,nbins=nbins)
        distance_distribution_random = mtpc.compute_DD(gdf_projected_random)
        distance_distribution_SP = mtpc.compute_DD(gdf)
        distance_distribution_SP = mtpc.binning_data(distance_distribution_SP,nbins,r_bins)[0:nbins]
        distance_distribution_random = mtpc.binning_data(distance_distribution_random,nbins,r_bins)[0:nbins]
        KL_distanceSP_inter = entropy(country_distribution_log+1,distance_distribution_SP+1)
        KL_distance_random_inter = entropy(country_distribution_log+1,distance_distribution_random+1)
        if KL_distanceSP_inter <KL_distanceSP:
            KL_distanceSP = KL_distanceSP_inter 
        if KL_distance_random_inter <KL_distance_random:
           KL_distance_random= KL_distance_random_inter
    print(KL_distanceSP,KL_distance_random)
    return 0
        

# 
# path_city = Path.cwd()/ data_dir / f"{name}_cities.csv"

# gdf_projected = gdf_city.to_crs(crs)
# coord = gdf_projected.get_coordinates()

name = "Belgium"
path_border = Path.cwd() / map_dir / f"{name}.geojson"
# gdf_city = mtpc.load_df_to_gdf(path_city,threshold)
gdf_edge = gpd.read_file(path_border)

out_dir = Path.cwd() /Path(f"out/2pcf/{name}")
file = out_dir / "1_5_7000_3__100000.0_log_30"
dico_results = slp.load_results(file)
r_edges = dico_results["r_edges"]
gamma, r0 = dico_results["fit_SP"]
l_xi = dico_results["xi"]
country_distribution = dico_results["distance_distribution"]
l_cluster = dico_results["clustering"]
number_point = dico_results["number_points"]
N = dico_results["number_points"]
crs = mtpc.crs_selector(name)
gdf_edge = gdf_edge.to_crs(crs)

eta = 3#l_cluster[] #density, number of circles by steps
l = eta**(1/(2-gamma)) #fraction of space covered by the circles
L = int(np.log(N)/np.log(eta)+1)-1 #size of circles, number of steps
#L=12
#L=3
erase_nodes = 400
print(eta**L)
#l=1.42
print("gamma", gamma)
print("fractal dimension",np.log(eta)/np.log(l))
print("l = ", l)
print("L = ", L)
R = 0.5*gdf_edge.length[0]/(2*np.pi)#np.max(country_distribution) # radius max
#R = np.min(r_edges)*(l**(L-1))
min_rad = R/(l**(L-1))
nbins=30
rmin=0
N_run= 5
size= 3000
random_point = number_point-eta**L+erase_nodes
print("mimimum radius",min_rad )
print("random_point=", random_point)
n_iteration=1000
rmax=0
k = N_run
scale="log"
compare_linear_distribution(n_iteration,l,eta,L,R,gdf_edge,crs,number_point,random_point,erase_nodes,k,rmax,scale,nbins)
#compare_log_distribution(n_iteration,l,eta,L,R,gdf_edge,crs,number_point,random_point,erase_nodes,k,rmax,scale,nbins,country_distribution)
#position = soneira_peebles_model(l,eta,L,R,erase_nodes=0)
# position = check_inside(l,eta,L,R,gdf_edge,erase_nodes) #soneira_peebles_border(l,eta,L,R,gdf_edge,erase_nodes=erase_nodes)
# distance_distribution = distance.pdist(position)

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

# gdf_random_square = squares_gdf.sample_points(random_point)
# #gdf_random_square = gdf_random_square.sample_points(random_point)
# gdf_random_square = gdf_edge.sample_points(random_point)
# gdf_random_points = gpd.GeoDataFrame(gdf_random_square.explode().get_coordinates(),geometry=gdf_random_square.explode(),crs=crs)
# gdf = gpd.GeoDataFrame(pd.concat( [gdf,gdf_random_points], ignore_index=True))
# fig,ax = plt.subplots()
# squares_gdf.plot(ax=ax)
# gdf_edge.plot(ax=ax,color="black")
# gdf.plot(ax=ax,color="red",markersize=0.5)
# #N_SP = eta**L
# k=3
# rmax = 1e5#R/(np.max(country_distribution)/1e5)#np.max(r_edges)
# rmin = 0#np.min(r_edges)
# scale="log"
# # sns.histplot(distance_distribution)
# gdf_projected_random= mtpc.generate_random_point(gdf_edge,number_point,crs,check_gpd=True)
# r_edges_random_SP,l_xi_random_SP = mtpc.PCF_with_variance(gdf_projected_random,gdf_edge,crs,N_run,size,k,rmax,scale,nbins=nbins)
# distance_distribution_random = mtpc.compute_DD(gdf_projected_random)
# distance_distribution_SP = mtpc.compute_DD(gdf)
# # plot_2pcf(r_edges_random_SP,l_xi_random_SP)


# r_edges_SP,l_xi_SP = mtpc.PCF_with_variance_SP(gdf,squares_gdf,crs,N_run,size,k,rmax,scale,nbins=nbins,rmin=rmin)

# 
# xi = np.mean(l_xi,axis=0)
# xi_SP =np.mean(l_xi_SP,axis=0)
# xi_random_SP = np.mean(l_xi_random_SP,axis=0)
# KL_2pcf = entropy(xi,xi_SP)
# KL_2pcf_random = entropy(xi,xi_random_SP)
# nbins=20
# r_bins = np.logspace(np.log10(100),np.log10(np.max(country_distribution)+1),nbins)
# # country_distribution = mtpc.binning_data(country_distribution,nbins,r_bins)
# # distance_distribution_SP = mtpc.binning_data(distance_distribution_SP,nbins,r_bins)[0:nbins]
# # distance_distribution_random = mtpc.binning_data(distance_distribution_random,nbins,r_bins)[0:nbins]
# KL_distanceSP = entropy(country_distribution+1,distance_distribution_SP+1)
# KL_distance_random = entropy(country_distribution+1,distance_distribution_random+1)

# country_distribution = mtpc.binning_data(country_distribution,nbins,r_bins)
# distance_distribution_SP = mtpc.binning_data(distance_distribution_SP,nbins,r_bins)[0:nbins]
# distance_distribution_random = mtpc.binning_data(distance_distribution_random,nbins,r_bins)[0:nbins]
# KL_distanceSP_log = entropy(country_distribution+1,distance_distribution_SP+1)
# KL_distance_random_log = entropy(country_distribution+1,distance_distribution_random+1)

# plot_2pcf(r_edges_SP,l_xi_SP)
# plot_2pcf(r_edges,l_xi)


# sns.histplot(country_distribution,bins=nbins,label="original",legend=True,log_scale=True)
# sns.histplot(distance_distribution_SP,bins=nbins,label="SP",legend=True,log_scale=True)
# sns.histplot(distance_distribution_random,bins=nbins,label="random",legend=True,log_scale=True)
# plt.yscale("log")
# plt.legend()
# plt.show()
# #plot_two_2pcf(r_edges,l_xi,l_xi_SP)
# from scipy.optimize import curve_fit
# def power_law(r, gamma,r0):
#      model = (r/r0)**(-gamma)
#      return model
# a,b = 1,15
# xi_SP = np.mean(l_xi_SP,axis=0)
# error = np.std(l_xi_SP,axis=0)
# xi_reduced = xi_SP[a:b]
# r_reduced = r_edges_SP[a:b]
# error_reduced = error[a:b]
# popt,pcov = curve_fit(power_law,r_reduced,xi_reduced,sigma=error_reduced,p0=(gamma,r0))
# print(popt)
# print(np.sqrt(pcov[0][0]))
# s=(r0/popt[1])
# position = (1/s)*position
# print("s=",s)
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

# fig,ax = plt.subplots()
# squares_gdf.plot(ax=ax)
# #gdf_edge.plot(ax=ax,color="black")
# gdf.plot(ax=ax,color="red",markersize=0.5)
# # N_SP = eta**L
# k=3
# rmax = 1e5#s*np.max(distance_distribution)#/(np.max(country_distribution)/1e5)#np.max(r_edges)
# rmin = 0#np.min(r_edges)
# scale="log"
# # sns.histplot(distance_distribution)
# # gdf_projected_random= mtpc.generate_random_point(squares_gdf,N_SP,crs,check_gpd=True)
# # r_edges_random_SP,l_xi_random_SP = mtpc.PCF_with_variance(gdf_projected_random,squares_gdf,crs,N_run,size,k,rmax,scale,nbins=nbins)
# # distance_distribution_random = mtpc.compute_DD(gdf_projected_random)
# # plot_2pcf(r_edges_random_SP,l_xi_random_SP)

# r_edges_SP,l_xi_SP = mtpc.PCF_with_variance_SP(gdf,squares_gdf,crs,N_run,size,k,rmax,scale,nbins=nbins,rmin=rmin)
# plot_2pcf(r_edges_SP,l_xi_SP)
# plot_2pcf(r_edges,l_xi)
# #plot_two_2pcf(r_edges,l_xi,l_xi_SP)
# from scipy.optimize import curve_fit
# def power_law(r, gamma,r0):
#      model = (r/r0)**(-gamma)
#      return model
# a,b = 4,12
# xi_SP = np.mean(l_xi_SP,axis=0)
# error = np.std(l_xi_SP,axis=0)
# xi_reduced = xi_SP[a:b]
# r_reduced = r_edges_SP[a:b]
# error_reduced = error[a:b]
# popt,pcov = curve_fit(power_law,r_reduced,xi_reduced,sigma=error_reduced,p0=(gamma,5e3))
# print(popt)
# print(np.sqrt(pcov[0][0]))



