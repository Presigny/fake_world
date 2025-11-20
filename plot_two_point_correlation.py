import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys
path = Path().cwd().parent
sys.path.append(str(path /"src"))
import save_load_pickle as slp
import methods_two_point_correlation as mtpc


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

def plot_2pcf(load_dir):
    i=1
    name,rmin,nbins=load_dir.name.split("_")
    fig, ax = plt.subplots()
    sns.set(style="whitegrid")
    for item in load_dir.iterdir():
        print(item)
        d = slp.load_results(item)
        Nsim,N_run,N_point_random,threshold_city = item.name.split("_")
        r_edges, xi = d["r_edges"],d["xi"]
        color = sns.color_palette("viridis", i)[0]
        r_edges = r_edges[1:]
        xi = xi[1:]
        i+=1
        print(xi)
        width = np.concatenate((np.array([2000]),np.diff(r_edges))) #need to find a way to encode properly the first alignement
        ax.bar(r_edges,xi,width=width,align="center",edgecolor="black",color=color,alpha=0.5)
    l_error = l_error[1:]
    print(l_error)
    plt.errorbar(r_edges, xi, yerr=l_error, fmt="o", color="r",ms=1)
    title = "2 points correlation function "+name
    plt.title(title)
    plt.ylabel(r"$\xi(r)$")
    plt.xlabel(r"$r$ (meters)")
    plt.xscale("log")
    #plt.yscale("log")
    #plt.legend()
    plt.savefig(save_dir / title, format="png",transparent=True)
    plt.show()

def plot_two_point_correlation(load_dir,l_error,save_dir):
    i=1
    name,rmin,nbins=load_dir.name.split("_")
    fig, ax = plt.subplots()
    sns.set(style="whitegrid")
    for item in load_dir.iterdir():
        print(item)
        d = slp.load_results(item)
        Nsim,N_run,N_point_random,threshold_city = item.name.split("_")
        r_edges, xi = d["r_edges"],d["xi"]
        color = sns.color_palette("viridis", i)[0]
        r_edges = r_edges[1:]
        xi = xi[1:]
        i+=1
        print(xi)
        width = np.concatenate((np.array([2000]),np.diff(r_edges))) #need to find a way to encode properly the first alignement
        ax.bar(r_edges,xi,width=width,align="center",edgecolor="black",color=color,alpha=0.5)
    l_error = l_error[1:]
    print(l_error)
    plt.errorbar(r_edges, xi, yerr=l_error, fmt="o", color="r",ms=1)
    title = "2 points correlation function "+name
    plt.title(title)
    plt.ylabel(r"$\xi(r)$")
    plt.xlabel(r"$r$ (meters)")
    plt.xscale("log")
    #plt.yscale("log")
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
        print(xi)
        color = sns.color_palette("viridis", i)[0]
        i+=1
        #fig, ax = plt.subplots()
        sns.set(style="whitegrid")
        width = np.concatenate((np.array([2000]),np.diff(r_edges))) #need to find a way to encode properly the first alignement
        plt.bar(r_edges,xi,width=width,align="center",edgecolor="black",color=color,alpha=0.5)
    #plt.errorbar(r_edges, xi, yerr=l_error, fmt="o", color="r",ms=1)
        title = "2 points correlation function less points "+name
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

def scatter_point_correlation(load_dir,l_error,save_dir):
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
        powerlaw = [(r/4500)**(-0.8) for r in r_edges]
        #width = np.concatenate((np.array([2000]),np.diff(r_edges))) #need to find a way to encode properly the first alignement
        ax.scatter(r_edges,xi,edgecolor="black",color=color,alpha=0.5)
        ax.plot(r_edges,powerlaw)
    plt.errorbar(r_edges, xi, yerr=l_error, fmt="o", color="r",ms=1)
    title = "2 points correlation function "+name
    plt.title(title)
    plt.ylabel(r"$\xi(r)$")
    plt.xlabel(r"$r$ (meters)")
    plt.xscale("log")
    plt.yscale("log")
    #plt.legend()
    #plt.savefig(save_dir / title, format="png",transparent=True)
    plt.show()

def shot_noise(load_dir,l_error,save_dir):
    for item in load_dir.iterdir():
        d = slp.load_results(item)
        r_edges, xi,distance_distribution = d["r_edges"],d["xi"],d["distance_distribution"]
        print(len(d["distance_distribution"]))
        hist_DD = mtpc.binning_data(distance_distribution, len(r_edges), r_edges)
        return hist_DD


local_dir = Path("Italy_1000_20")
load_dir = Path.cwd().parent / "data/two_point_correlation" / local_dir
save_dir = Path.cwd() / "plot"
l_error = compute_error(load_dir) 

#each_plot_two_point_correlation(load_dir,l_error,save_dir)
plot_two_point_correlation(load_dir,l_error,save_dir)
hist_DD = shot_noise(load_dir,l_error,save_dir)
scatter_point_correlation(load_dir,l_error,save_dir)
#plot_distance_distribution(load_dir,save_dir)