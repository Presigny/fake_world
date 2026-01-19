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
import os 
from matplotlib.colors import LogNorm
from tqdm import tqdm
from shapely.geometry import MultiPolygon, Polygon

data_dir = Path().cwd().parent/Path("data")
out_dir = Path().cwd().parent/Path("out")


threshold = 1 # minimal number of inhabitants for the city to be considered
N_run = 5 # number of random catalog generated for computing the 2PCF
size = 7000 #number of points each random catalog has
k =  3  #number of time the 2PCF computing process is repeated to estimate variance
rmax = 1.6e5 #maximal distance between two points considered, if None, taken the max from data
scale = "log" #scale at which to bin the data, recommended "log"
nbins = 20 # number of bins to consider in the computation of 2PCF (betweem rmin and rmax)
bootstrap_size = 10_000
bootstrap_N_real = 10

parameters = {"threshold":threshold,"N_run":N_run,"size":size,"k":k,"rmax":rmax,"scale":scale,"nbins":nbins,"bootstrap_size":bootstrap_size,"bootstrap_N_real":bootstrap_N_real }


def plot_2pcf(load_file,scale, ax):
    """Plot and save the 2pcf store in load_file"""
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
    plt.ylabel(r"$\xi(r)$")
    plt.xlabel(r"$r$ (meters)")
    if scale == "lin":
        plt.xscale("linear")
    else:
        plt.xscale(scale)

def power_law(r, gamma,r0):
    """The power law form of the SP model"""
    model = (r/r0)**(-gamma)
    return model

def plot_2pcf_with_fit(load_file,popt,pcov,scale, ax):
    """Plot and save the 2pcf of the system together with the SP fit
    The 4 curves represent several combination of parameters within 1-sigma of the parameters
    """
    sns.set(style="whitegrid")
    d = slp.load_results(load_file)
    r_edges, l_xi = d["r_edges"],d["xi"]
    xi = np.mean(l_xi,axis=0)
    error = np.std(l_xi,axis=0)
    color = sns.color_palette("viridis")[0]
    r_edges = r_edges#[1:]
    xi = xi#[1:]
    error = error#[1:]
    if len(xi) != len(r_edges): #when we choose a rmax define by user the code put an extra category above
        xi = xi[0:-1]
        error = error[0:-1]
    end = np.zeros([1])
    end[0] = 1.1*r_edges[-1]
    r_edges_zero = np.concatenate((r_edges,end))
    width = np.diff(r_edges_zero) #need to find a way to encode properly the first alignement
    ax.bar(r_edges,xi,width=width,align="center",edgecolor="black",color=color,alpha=0.5)
    plt.errorbar(r_edges, xi, yerr=error, fmt="o", color="r",ms=1)
    delta_gamma,delta_r0=np.sqrt(np.diag(pcov))
    popt1 = [popt[0]+delta_gamma,popt[1]+delta_r0]
    popt2 = [popt[0]-delta_gamma,popt[1]-delta_r0]
    popt3 = [popt[0]+delta_gamma,popt[1]-delta_r0]
    popt4 = [popt[0]-delta_gamma,popt[1]+delta_r0]
    plt.plot(r_edges, power_law(r_edges, *popt), '--',label=rf"$\gamma={np.round(popt[0],3)} +- {np.round(delta_gamma,3)},r_0={np.round(popt[1],0)} +- {np.round(delta_r0,0)}$")
    plt.plot(r_edges, power_law(r_edges, *popt1), '--',color="red")
    plt.plot(r_edges, power_law(r_edges, *popt2), '--',color="orange")
    plt.plot(r_edges, power_law(r_edges, *popt3), '--',color="green")
    plt.plot(r_edges, power_law(r_edges, *popt4), '--',color="purple")
    plt.ylabel(r"$\xi(r)$")
    plt.xlabel(r"$r$ (meters)")
    if scale == "lin":
        plt.xscale("linear")
    else:
        plt.xscale("log")
    plt.legend()


def plot_silhouette(kmax,l_silhouette,ax):
    """Plot and save the silhouette plot"""
    # fig, ax = plt.subplots(figsize=(10,10))
    # plt.style.use("fivethirtyeight")
    plt.plot(range(2, kmax), np.mean(l_silhouette,axis=0))
    plt.errorbar(range(2, kmax), np.mean(l_silhouette,axis=0), yerr=np.std(l_silhouette,axis=0), fmt="o", color="r",ms=5)
    plt.xticks(range(2, kmax))
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Coefficient")
    ax.set_aspect(1. / ax.get_data_ratio(), adjustable='box')

from tqdm import tqdm
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

def compute_silhouette_with_subsampling(
    gdf_projected,
    kmax=20,
    Nsample=5000,
    n_repeats=5
):
    """
    gdf_projected : GeoDataFrame projeté (pour coord = .get_coordinates())
    kmax : nombre max de clusters testés
    Nsample : taille du sous-échantillon
    n_repeats : nombre de sous-échantillons pour moyenner
    """

    # Coordonnées complètes (mais on ne s’en sert que pour les samples)
    coords_full = gdf_projected.get_coordinates()

    silhouette_means = []   # silhouette moyen par k
    silhouette_stds = []    # écart type pour tracer un shaded plot éventuellement

    for k in tqdm(range(2, kmax)):

        scores_k = []  # scores obtenus sur différentes répétitions

        for r in range(n_repeats):

            # --- Sous-échantillonnage ---
            gdf_sample = gdf_projected.sample(Nsample, replace=False)
            coords = gdf_sample.get_coordinates().values

            # --- KMeans ---
            kmeans = KMeans(
                n_clusters=k,
                tol=1e-8,
                max_iter=200,
                n_init=10
            ).fit(coords)

            # --- Silhouette ---
            score = silhouette_score(coords, kmeans.labels_)
            scores_k.append(score)

        # Moyenne + écart type pour ce k
        silhouette_means.append(np.mean(scores_k))
        silhouette_stds.append(np.std(scores_k))

    return silhouette_means, silhouette_stds


def plot4(name, data_type, path_points_csv, path_border_geojson, threshold_variable, param = parameters):

    threshold = param["threshold"]
    N_run = param["N_run"]
    size = param["size"]
    k = param["k"]
    rmax = param["rmax"]
    scale = param["scale"]
    nbins = param["nbins"]
    bootstrap_size = param["bootstrap_size"]
    bootstrap_N_real = param["bootstrap_N_real"]

    crs = mtpc.crs_selector(name) # Coordinate reference system of the country considered
    gdf_points = mtpc.load_df_to_gdf(path_points_csv,threshold, threshold_variable) # Geopandas of the dataset
    gdf_edge = gpd.read_file(path_border_geojson) #Geopandas of the polygon border of the dataset

    geom = gdf_edge.geometry.iloc[0]

    # isoler le plus grand polygone
    if isinstance(geom, MultiPolygon):
        largest_poly = max(geom.geoms, key=lambda p: p.area)
    else:
        largest_poly = geom

    gdf_edge = gpd.GeoDataFrame(geometry=[largest_poly], crs=gdf_edge.crs)


    gdf_projected = gdf_points.to_crs(crs) # Projection in the right coordinate system
    coord = gdf_projected.get_coordinates() # Coordinates of the points in dataset
    print("number of data points : ", len(gdf_points))
    plt.figure(figsize=(12,12))
    print("step 1: 2pcf data")
    ax1 = plt.subplot(221)

    path_save = out_dir /Path("2pcf") /Path(f"{name}/")
    
    name_save = path_save/Path(f"{threshold}_{N_run}_{size}_{k}_{rmax}_{scale}_{nbins}_{name}_{data_type}")
    if len(gdf_projected) < bootstrap_size:
        r_edges,l_xi = mtpc.PCF_with_variance(gdf_projected,gdf_edge,crs,N_run,size,k,rmax,scale,nbins=nbins)
    else:
        r_edges,l_xi,  = mtpc.compute_two_point_correlation_bootstrap(gdf_projected,gdf_edge,crs,N_run,size,False,rmax,scale,nbins, bootstrap_size,bootstrap_N_real)
    print("2pcf done")
    d_save = {}
    d_save["r_edges"] = r_edges
    d_save["xi"] = l_xi
    os.makedirs(path_save, exist_ok=True)
    slp.save_results(name_save,d_save)


    a,b = 1,19 #the two extremal index between which fitting the SP"
    xi = np.mean(l_xi,axis=0)
    error = np.std(l_xi,axis=0) #vertical error due to statistical variance
    xi_reduced = xi[a:b]
    r_reduced = r_edges[a:b]
    error_reduced = error[a:b]
    try:
        popt,pcov = curve_fit(power_law,r_reduced,xi_reduced,sigma=error_reduced)
        plot_2pcf_with_fit(name_save,popt,pcov,scale, ax1)
    except RuntimeError:
        print("Fit failed!")
        plot_2pcf(name_save,scale, ax1)


    ax2 = plt.subplot(222)
    print("step 2: plot points")
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    gdf_points.plot(
        ax = ax2,
        markersize= 1,
        alpha= 0.25,
        # column="population_density",
        # norm=LogNorm(),
        # cmap="viridis",      # tu peux changer la colormap
        # legend=True,         # affiche la barre de couleurs
    )

    ax3 = plt.subplot(223)
    print("step 3: 2pcf random")

    path_save_random = out_dir/Path("2pcf_random") /Path( f"Random{name}/")
    name_save_random = path_save_random/Path(f"{threshold}_{N_run}_{size}_{k}_{rmax}_{scale}_{nbins}_{name}_{data_type}")

    gdf_projected_random= mtpc.generate_random_point(gdf_edge,len(gdf_points),crs,check_gpd=True)
    gdf_projected_random = gdf_projected_random.explode(ignore_index=True)
    if len(gdf_projected_random) < bootstrap_size:
        r_edges,l_xi = mtpc.PCF_with_variance(gdf_projected_random,gdf_edge,crs,N_run,size,k,rmax,scale,nbins=nbins)
    else:
        r_edges,l_xi,  = mtpc.compute_two_point_correlation_bootstrap(gdf_projected_random,gdf_edge,crs,N_run,size,False,rmax,scale,nbins, bootstrap_size,bootstrap_N_real)
    # print(l_xi)


    d_save = {}
    d_save["r_edges"] = r_edges
    d_save["xi"] = l_xi
    # d_save["distance_distribution"] = distance_distribution
    os.makedirs(path_save_random, exist_ok=True)
    slp.save_results(name_save_random,d_save)



    plot_2pcf(name_save_random,scale, ax3)


    ax4 = plt.subplot(224)
    print("step 4: clustering")
    if len(gdf_projected) < bootstrap_size:
        coord = gdf_projected.get_coordinates()
        kmax = 20 # maximum number of cluster to consider 
        l_silhouette = []
        silhouette_coefficients = []
        for k in tqdm(range(2, kmax)):
            kmeans = sk.KMeans(n_clusters=k,tol=1e-8,max_iter=100,n_init=10).fit(coord)#max_iter=1000,n_init=100
            score = silhouette_score(coord, kmeans.labels_)
            silhouette_coefficients.append(score)
        l_silhouette.append(silhouette_coefficients)

        plot_silhouette(kmax,l_silhouette, ax4)

    else:
        sil_means, sil_stds = compute_silhouette_with_subsampling(
            gdf_projected,
            kmax=20,
            Nsample=bootstrap_size,
            n_repeats=bootstrap_N_real
        )

        l_silhouette = [sil_means]  # Pour respecter ta signature existante
        plot_silhouette(20, l_silhouette, ax4)


        ax4.plot(range(2,20), sil_means)
        ax4.fill_between(range(2,20),
                        np.array(sil_means)-np.array(sil_stds),
                        np.array(sil_means)+np.array(sil_stds),
                        alpha=0.2)


    plt.show()




def plot5(name, data_type, gdf_points, path_border_geojson, threshold_variable, param = parameters):

    threshold = param["threshold"]
    N_run = param["N_run"]
    size = param["size"]
    k = param["k"]
    rmax = param["rmax"]
    scale = param["scale"]
    nbins = param["nbins"]
    bootstrap_size = param["bootstrap_size"]
    bootstrap_N_real = param["bootstrap_N_real"]

    crs = mtpc.crs_selector(name) # Coordinate reference system of the country considered
    gdf_edge = gpd.read_file(path_border_geojson) #Geopandas of the polygon border of the dataset

    geom = gdf_edge.geometry.iloc[0]

    # isoler le plus grand polygone
    if isinstance(geom, MultiPolygon):
        largest_poly = max(geom.geoms, key=lambda p: p.area)
    else:
        largest_poly = geom

    gdf_edge = gpd.GeoDataFrame(geometry=[largest_poly], crs=gdf_edge.crs)


    gdf_projected = gdf_points.to_crs(crs) # Projection in the right coordinate system
    coord = gdf_projected.get_coordinates() # Coordinates of the points in dataset
    print("number of data points : ", len(gdf_points))
    plt.figure(figsize=(12,12))
    print("step 1: 2pcf data")
    ax1 = plt.subplot(221)

    path_save = out_dir /Path("2pcf") /Path(f"{name}/")
    
    name_save = path_save/Path(f"{threshold}_{N_run}_{size}_{k}_{rmax}_{scale}_{nbins}_{name}_{data_type}")
    if len(gdf_projected) < bootstrap_size:
        r_edges,l_xi = mtpc.PCF_with_variance(gdf_projected,gdf_edge,crs,N_run,size,k,rmax,scale,nbins=nbins)
    else:
        r_edges,l_xi,  = mtpc.compute_two_point_correlation_bootstrap(gdf_projected,gdf_edge,crs,N_run,size,False,rmax,scale,nbins, bootstrap_size,bootstrap_N_real)
    print("2pcf done")
    d_save = {}
    d_save["r_edges"] = r_edges
    d_save["xi"] = l_xi
    os.makedirs(path_save, exist_ok=True)
    slp.save_results(name_save,d_save)


    a,b = 1,19 #the two extremal index between which fitting the SP"
    xi = np.mean(l_xi,axis=0)
    error = np.std(l_xi,axis=0) #vertical error due to statistical variance
    xi_reduced = xi[a:b]
    r_reduced = r_edges[a:b]
    error_reduced = error[a:b]
    try:
        popt,pcov = curve_fit(power_law,r_reduced,xi_reduced,sigma=error_reduced)
        plot_2pcf_with_fit(name_save,popt,pcov,scale, ax1)
    except RuntimeError:
        print("Fit failed!")
        plot_2pcf(name_save,scale, ax1)


