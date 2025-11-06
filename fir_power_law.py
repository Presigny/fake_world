import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys
path = Path().cwd()
sys.path.append(str(path /"src"))
import save_load_pickle as slp
from scipy.optimize import minimize
import methods_two_point_correlation as mtpc
import geopandas as gpd 


def compute_error(load_dir):
    l_xi = []
    for item in load_dir.iterdir():
        d = slp.load_results(item)
        r_edges, xi = d["r_edges"],d["xi"]
        l_xi.append(xi)
    l_xi = np.array(l_xi).transpose()
    l_error = []
    for i in range(len(l_xi)):
        l_error.append(3*np.std(l_xi[i]))
    return l_error

def plot_two_point_correlation(load_dir,l_error,save_dir):
    i=1
    name,rmin,nbins=load_dir.name.split("_")
    fig, ax = plt.subplots()
    sns.set(style="whitegrid")
    for item in load_dir.iterdir():
        d = slp.load_results(item)
        Nsim,N_run,N_point_random,threshold_city = item.name.split("_")
        r_edges, xi = d["r_edges"],d["xi"]
        color = sns.color_palette("viridis", i)[0]
        i+=1
        width = np.concatenate((np.array([2000]),np.diff(r_edges))) #need to find a way to encode properly the first alignement
        ax.bar(r_edges,xi,width=width,align="center",edgecolor="black",color=color,alpha=0.5)
    plt.errorbar(r_edges, xi, yerr=l_error, fmt="o", color="r",ms=1)
    title = "2 points correlation function "+name
    plt.title(title)
    plt.ylabel(r"$\xi(r)$")
    plt.xlabel(r"$r$ (meters)")
    plt.xscale("log")
    plt.yscale("log")
    #plt.legend()
    plt.savefig(save_dir / title, format="png",transparent=True)
    plt.show()

def each_plot_two_point_correlation(load_dir,l_error,save_dir):
    i=1
    name,rmin,nbins=load_dir.name.split("_")
    for item in load_dir.iterdir():
        d = slp.load_results(item)
        Nsim,N_run,N_point_random,threshold_city = item.name.split("_")
        r_edges, xi = d["r_edges"],d["xi"]
        print(r_edges[0])
        color = sns.color_palette("viridis", i)[0]
        i+=1
        #fig, ax = plt.subplots()
        sns.set(style="whitegrid")
        width = np.concatenate((np.array([2000]),np.diff(r_edges))) #need to find a way to encode properly the first alignement
        plt.bar(r_edges,xi,width=width,align="center",edgecolor="black",color=color,alpha=0.5)
    #plt.errorbar(r_edges, xi, yerr=l_error, fmt="o", color="r",ms=1)
        title = "2 points correlation function "+name
        plt.title(title)
        plt.ylabel(r"$\xi(r)$")
        plt.xlabel(r"$r$ (meters)")
        plt.xscale("log")
        plt.yscale("linear")
        #plt.legend()
       # plt.savefig(save_dir / title, format="png",transparent=True)
        plt.show()
    
def plot_distance_distribution(load_dir,save_dir,r_dist_min=False):
    i=1
    name,rmin,nbins=load_dir.name.split("_")
    fig, ax = plt.subplots()
    sns.set(style="whitegrid")
    for item in load_dir.iterdir():
        d = slp.load_results(item)
        Nsim,N_run,N_point_random,threshold_city = item.name.split("_")
        distance_distribution,r_edges = d["distance_distribution"],d["r_edges"]
        if r_dist_min:
            binrange = [r_edges[0],np.max(distance_distribution)]
        else:
            binrange = [np.min(distance_distribution),np.max(distance_distribution)]

        color = sns.color_palette("viridis", i)[0]
        i+=1
        sns.histplot(distance_distribution,stat="probability",bins=3*int(nbins),binrange=binrange)
        title = "Distance distribution "+name
        plt.title(title)
        plt.xlabel(r"$r$ (meters)")
        plt.xscale("log")
        #plt.yscale("log")
        # #plt.legend()
        plt.savefig(save_dir / title, format="png",transparent=True)
        plt.show()
        return 0

def xi_bin_avg(r1, r2, r0, gamma):
    # average (r/r0)^(-gamma) over bin weighted by r^2 volume
    num = (r2**(2-gamma) - r1**(2-gamma)) / (2-gamma)
    den = (r2**2 - r1**2) / 2.0
    return (r0**gamma) * num / den

def model_xi_bins(r_edges, r0, gamma):
    nbins = len(r_edges)-1 
    xi = np.zeros(nbins)
    for i in range(nbins):
        xi[i] = xi_bin_avg(r_edges[i], r_edges[i+1], r0, gamma)
    return xi

def neg_log_likelihood(params, DD, RR, r_edges):
    r0, gamma = params
    if r0 <= 0 or gamma <= 0:  # physical bounds
        return np.inf

    C = len(DD)/len(RR)#(ND*(ND-1.0)) / (NR*(NR-1.0))
    xi_model = model_xi_bins(r_edges, r0, gamma)
    mu = C * RR * (1.0 + xi_model)
    mu = np.maximum(mu, 1e-12)  # avoid log(0)

    # Poisson log-likelihood (dropping constant terms)
    nll = np.sum(mu - DD * np.log(mu))
    return nll

save_dir = Path("data/two_point_correlation")
data_dir = Path("data")
number_xi = 10
N_run = 5
size = 10000
threshold = 5000
name = "France"
rmin = 1000
nbins = 20
crs = mtpc.crs_selector(name)
path_city = Path.cwd() / data_dir / "france_cities.csv"
path_border = Path.cwd() / data_dir / "map/France.geojson"

gdf_city = mtpc.load_df_to_gdf(path_city,threshold)
gdf_edge = gpd.read_file(path_border)
gdf_projected = gdf_city.to_crs(crs)

# path_save = Path.cwd().parent / save_dir /Path(name+"_"+str(rmin)+"_"+str(nbins))
# path_save.mkdir(exist_ok=True)

r_edges,xi,RR = mtpc.compute_two_point_correlation_with_RR(gdf_projected,gdf_edge,crs,N_run,size,rmin,nbins=nbins)
DD = mtpc.compute_DD(gdf_projected)

print("fit begins")
# Example fit
mask = (xi >0) & (r_edges >10000) & (r_edges <1e6) #where it fits is the most probable
r_edges = r_edges[mask]
r_edges = np.logspace(np.log10(r_edges[0]),np.log10(r_edges[-1]),20)
RR = mtpc.binning_data(RR, len(r_edges), r_edges)
DD = mtpc.binning_data(DD, len(r_edges), r_edges)
DD = DD[1:]
DD = DD[0:(len(DD)-1)]
RR = RR[1:]
RR = RR[0:(len(RR)-1)]
p0 = [70000, 1.8]  # initial guess for (r0, gamma)
res = minimize(neg_log_likelihood, p0, args=(DD, RR, r_edges),method='Nelder-Mead')
print("Best-fit parameters:", res.x)
    

#each_plot_two_point_correlation(load_dir,l_error,save_dir)
# plot_two_point_correlation(load_dir,l_error,save_dir)
# plot_distance_distribution(load_dir,save_dir)