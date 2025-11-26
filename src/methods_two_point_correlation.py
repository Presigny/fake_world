import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial import distance
import seaborn as sns
from scipy.spatial import cKDTree
from scipy.stats import qmc
from shapely.geometry import Point

def find_rmax_meaningful(gdf_edge,crs,initial_guess):
    gdf_border_in_crs = gdf_edge.to_crs(crs)
    circle=gdf_border_in_crs.centroid.buffer(initial_guess)
    radius = initial_guess
    while not(gdf_border_in_crs.within(circle)[0]):
        radius += 1000
        circle=gdf_border_in_crs.centroid.buffer(radius)
    f,ax = plt.subplots()
    gdf_border_in_crs.plot(ax=ax,color="red")
    circle.plot(ax=ax)
    plt.show()
    print(radius)
    return radius


def load_df_to_gdf(path,threshold):
    df = pd.read_csv(path,low_memory=False)
    df = df.loc[df["population"] >= threshold]
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lng, df.lat), crs="EPSG:4326")
    return gdf

def generate_sobol_point(gdf_edge,size,crs,check_gpd=False):
    print(size)
    sampler = qmc.Sobol(d=2, scramble=True)
    sobol_points = sampler.random(n=size)
    gdf_projected = gdf_edge.to_crs(crs)
    print(qmc.discrepancy(sobol_points))
    minx, miny, maxx, maxy = gdf_projected.total_bounds
    for i in range(len(sobol_points)):
        sobol_points[i][0] = sobol_points[i][0]*(maxx-minx)+minx
        sobol_points[i][1] = sobol_points[i][1]*(maxy-miny)+miny
    geometry = gpd.points_from_xy(sobol_points.T[0], sobol_points.T[1])
    geometry = gpd.GeoDataFrame(geometry).set_geometry(0,crs=crs)
    geometry = geometry.rename(columns ={0:'geometry'}).set_geometry('geometry',crs=crs)
    points_inside = gpd.sjoin(geometry,gdf_projected, how="inner",predicate="within")
    points_inside.plot()
    coord = points_inside.get_coordinates().to_numpy()
    print(len(coord))
    if check_gpd:
        return gdf_projected
    else:
        return coord
    
def generate_random_point(gdf_edge,size,crs,check_gpd=False):
    sample = gdf_edge.sample_points(size)
    gdf_projected = sample.to_crs(crs)
    coord = gdf_projected.get_coordinates().to_numpy()
    if check_gpd:
        return gdf_projected
    else:
        return coord

# def compute_one_RR(gdf_edge,size,crs):
#     coord_random = generate_random_point(gdf_edge, size, crs)
#     RR = np.triu(distance.cdist(coord_random,coord_random)).astype(np.int32)
#     RR = np.ravel(RR)
#     RR = RR[RR != 0]
#     return RR

def compute_DD(gdf_projected):
    coord_data = gdf_projected.get_coordinates().to_numpy()
    DD = distance.pdist(coord_data).astype(np.int32)#np.triu(distance.cdist(coord_data,coord_data)).astype(np.int32)
    #DD = np.ravel(DD)
    #DD = DD[DD != 0]
    return DD

def compute_one_DR_RR(gdf_projected,gdf_edge, size, crs):
    coord_random = generate_random_point(gdf_edge, size, crs)#generate_sobol_point(gdf_edge, size, crs)#generate_random_point(gdf_edge, size, crs)
    coord_data = gdf_projected.get_coordinates().to_numpy()
    DR = distance.cdist(coord_data,coord_random).astype(np.int32)
    DR = np.ravel(DR)
    DR = DR[DR != 0]
    RR = distance.pdist(coord_random).astype(np.int32)#np.triu(distance.cdist(coord_random,coord_random)).astype(np.int32)
    #RR = np.ravel(RR)
    #RR = RR[RR != 0]
    return DR,RR#,len(coord_random)

def compute_DD_SP(gdf_projected):
    coord_data = gdf_projected.get_coordinates().to_numpy()
    DD = distance.pdist(coord_data).astype(np.float32)#np.triu(distance.cdist(coord_data,coord_data)).astype(np.int32)
    #DD = np.ravel(DD)
    #DD = DD[DD != 0]
    return DD

def compute_one_DR_RR_SP(gdf_projected,gdf_edge, size, crs):
    coord_random = generate_random_point(gdf_edge, size, crs)#generate_sobol_point(gdf_edge, size, crs)#generate_random_point(gdf_edge, size, crs)
    coord_data = gdf_projected.get_coordinates().to_numpy()
    DR = distance.cdist(coord_data,coord_random).astype(np.float32)
    DR = np.ravel(DR)
    DR = DR[DR != 0]
    RR = distance.pdist(coord_random).astype(np.float32)#np.triu(distance.cdist(coord_random,coord_random)).astype(np.int32)
    #RR = np.ravel(RR)
    #RR = RR[RR != 0]
    return DR,RR#,len(coord_random)

def binning_data(data,nbins,r_edges):
    indices = np.digitize(data, r_edges,right=True)
    hist = np.bincount(indices)
    return hist

def normalized_count(hist,N1,DR=False,N2=None):
    var_norm = np.zeros(len(hist))
    if DR:
        for r in range(len(var_norm)):
            var_norm[r] = hist[r]*1/(N1*N2)
    else:
        for r in range(len(var_norm)):
            var_norm[r] = hist[r]*2/(N1*(N1-1))
    return var_norm

def compute_LS_correlation(DD,DR,RR):
    xi = np.zeros(len(DD))
    for r in range(len(DD)):
        xi[r] = (DD[r] - 2*DR[r] + RR[r])/RR[r]
    return xi

# def load_border(path,name):
#     gdf = gpd.read_file(path)
#     gdf = gdf.loc[gdf["NAME_ENGL"]==name]
#     return gdf

def crs_selector(name):
    if name == "France":
        return 'EPSG:2154'
    if name == "Italy":
        return "EPSG:32632"
    if name == "Belgium":
        return "EPSG:3812"
    if name == "Switzerland":
        return "EPSG:21781"
    if name == "Ukraine":
        return "EPSG:5558"
    if name == "Germany":
        return "EPSG:4839"
    if name == "Spain":
        return "EPSG:25830"
    if name == "Netherlands":
        return "EPSG:28992"
    return 0

def compute_rmax(DD,DR,RR):
    rmax = [np.max(DD),np.max(DR),np.max(DD)]
    return np.max(rmax)+1 #plus one is to ensure we have the correct number of bins and not one added on top

def compute_rmin(gdf_projected):
    coord_data = gdf_projected.get_coordinates().to_numpy()
    Tree = cKDTree(coord_data)
    second_nearest,points = Tree.query(coord_data,k=3)
    second_nearest =  second_nearest.T
    nearest = second_nearest[1]
    index_zero = np.where(nearest == 0)
    after_nearest = second_nearest[2][index_zero] #becasue some neighborhood of cities have same position
    nearest = np.concatenate((nearest,after_nearest))
    return np.mean(nearest)
    

def compute_two_point_correlation(gdf_projected,gdf_edge,crs,N_run,size,rmin,nbins=20):
    DD = compute_DD(gdf_projected)
    DR,RR = compute_one_DR_RR(gdf_projected,gdf_edge, size, crs)
    for i in range(N_run-1):
        DR_i,RR_i = compute_one_DR_RR(gdf_projected,gdf_edge, size, crs)
        RR = np.concatenate((RR,RR_i))
        DR = np.concatenate((DR,DR_i))
    rmin = compute_rmin(gdf_projected)
    rmax = compute_rmax(DD,DR,RR)
    r_edges = np.logspace(np.log10(rmin), np.log10(rmax), nbins)
   # r_edges = np.linspace(rmin,rmax,nbins)
    hist_DD = binning_data(DD,nbins,r_edges)
    DD_norm = hist_DD/len(DD) #normalized_count(hist_DD,N_D)
    del DD
    hist_DR = binning_data(DR,nbins,r_edges)
    DR_norm = hist_DR/len(DR)
    del DR
    hist_RR = binning_data(RR,nbins,r_edges)
    RR_norm = hist_RR/len(RR)
    del RR
    print(rmin,rmax)
    print(RR_norm)
    xi= compute_LS_correlation(DD_norm,DR_norm,RR_norm)
    return r_edges,xi

# def compute_two_point_correlation_average(gdf_projected,gdf_edge,crs,N_run,size,rmin,nbins=20):
#     DD = compute_DD(gdf_projected)
#     rmin = compute_rmin(gdf_projected)
#     rmax = np.max(DD)+1
#     r_edges = np.logspace(np.log10(rmin), np.log10(rmax), nbins)
#     hist_DD = binning_data(DD,nbins,r_edges)
#     DD_norm = hist_DD/len(DD) #normalized_count(hist_DD,N_D)
#     l_RR = []
#     l_DR = []
#     l_xi = []
#     for i in range(N_run):
#        # print(len(DD_norm))
#         DR_i,RR_i = compute_one_DR_RR(gdf_projected,gdf_edge, size, crs)
#         hist_DR = binning_data(DR_i,nbins,r_edges)
#         DR_norm = hist_DR/len(DR_i)
#         #print(len(DR_norm))
#         if len(DR_norm) == len(DD_norm)+1: # dont count the bin where r>r_max
#             DR_norm = DR_norm[0:(len(DR_norm)-1)]
#         l_DR.append(DR_norm)
#         del DR_i
#         hist_RR = binning_data(RR_i,nbins,r_edges)
#         RR_norm = hist_RR/len(RR_i)
#         #print(len(RR_norm))
#         if len(RR_norm) == len(DD_norm)+1:
#             RR_norm = RR_norm[0:(len(RR_norm)-1)]
#             #print("lol")
#         #if len(RR_norm) == len(DD_norm)-1:
#             #print("lol1")
#         del RR_i
#         l_RR.append(RR_norm)
#         print(hist_DD[-1],hist_RR[-2],hist_DR[-2])
#         l_xi.append(compute_LS_correlation(DD_norm,DR_norm,RR_norm))
#    # r_edges = np.linspace(rmin,rmax,nbins)
#     print(np.mean(l_xi,axis=0))
#     print(np.std(l_xi,axis=0))
#     avg_DR = np.mean(l_DR,axis=0)
#     avg_RR = np.mean(l_RR,axis=0)
#     print(rmin,rmax)
#     print(avg_RR[-1])
#     print(avg_DR[-1])
#     print(DD_norm[-1])
#     print(DD_norm)
#     xi= compute_LS_correlation(DD_norm,avg_DR,avg_RR)
#     return r_edges,xi

# def compute_two_point_correlation_with_RR(gdf_projected,gdf_edge,crs,N_run,size,rmin,nbins=20):
#     DD = compute_DD(gdf_projected)
#     DR,RR = compute_one_DR_RR(gdf_projected,gdf_edge, size, crs)
#     for i in range(N_run-1):
#         DR_i,RR_i = compute_one_DR_RR(gdf_projected,gdf_edge, size, crs)
#         RR = np.concatenate((RR,RR_i))
#         DR = np.concatenate((DR,DR_i))
#     rmin = compute_rmin(gdf_projected)
#     rmax = compute_rmax(DD,DR,RR)
#     r_edges = np.logspace(np.log10(rmin), np.log10(rmax), nbins)
#    # r_edges = np.linspace(rmin,rmax,nbins)
#     hist_DD = binning_data(DD,nbins,r_edges)
#     DD_norm = hist_DD/len(DD) #normalized_count(hist_DD,N_D)
#     del DD
#     hist_DR = binning_data(DR,nbins,r_edges)
#     DR_norm = hist_DR/len(DR)
#     del DR
#     hist_RR = binning_data(RR,nbins,r_edges)
#     RR_norm = hist_RR/len(RR)
#     xi= compute_LS_correlation(DD_norm,DR_norm,RR_norm)
#     return r_edges,xi,RR

def compute_LS_correlation_2019(DD,DR,RR,Nd,Nr_prime,Nr):
    return (Nr*(Nr_prime-1)*DD)/(Nd*(Nd-1)*RR)-(((Nr_prime-1)*DR)/(Nd*RR))+1

def compute_two_point_correlation_2019(gdf_projected,gdf_edge,crs,N_run,Nr_prime,rmin,Nd,rmax,scale,nbins=20):
    Nr = Nr_prime*N_run
    #Nr = len(gdf_projected)
    DD = compute_DD(gdf_projected)
    DR,RR = compute_one_DR_RR(gdf_projected,gdf_edge, Nr_prime, crs)
    for i in range(N_run-1):
        print(i)
        DR_i,RR_i = compute_one_DR_RR(gdf_projected,gdf_edge, Nr_prime, crs)
        RR = np.concatenate((RR,RR_i))
        DR = np.concatenate((DR,DR_i))
    if not rmin:
        rmin = compute_rmin(gdf_projected)
    if rmax:
        rmax = rmax
    else:
        rmax = np.max(DD)+1#find_rmax_meaningful(gdf_edge,crs,100000)
    if scale =="log":
        r_edges = np.logspace(np.log10(rmin), np.log10(rmax), nbins)
    elif scale == "lin":
        r_edges = np.linspace(rmin,rmax,nbins)
    hist_DD = binning_data(DD,nbins,r_edges)
    print("max DD", np.max(DD))
    #hist_DD = hist_DD[0:(len(hist_DD)-1)]
    del DD
    hist_DR = binning_data(DR,nbins,r_edges)
    del DR
    hist_RR = binning_data(RR,nbins,r_edges)
    del RR
    print(rmin,rmax)
    if len(hist_RR) == len(hist_DD)+1:
        hist_RR = hist_RR[0:(len(hist_RR)-1)]
    if len(hist_DR) == len(hist_DD)+1:
        hist_DR = hist_DR[0:(len(hist_DR)-1)]
    print(np.sum(hist_DD)*(2/(Nd*(Nd-1))))
    print(np.sum(hist_DR)/((Nd*(Nr))))
    print(np.sum(hist_RR)*(2/(Nr*(Nr_prime-1))))
    xi= compute_LS_correlation_2019(hist_DD,hist_DR,hist_RR,Nd,Nr_prime,Nr)
    return r_edges,xi

def compute_two_point_correlation_2019_SP(gdf_projected,gdf_edge,crs,N_run,Nr_prime,rmin,Nd,rmax,scale,nbins=20):
    Nr = Nr_prime*N_run
    #Nr = len(gdf_projected)
    DD = compute_DD_SP(gdf_projected)
    DR,RR = compute_one_DR_RR_SP(gdf_projected,gdf_edge, Nr_prime, crs)
    for i in range(N_run-1):
        print(i)
        DR_i,RR_i = compute_one_DR_RR_SP(gdf_projected,gdf_edge, Nr_prime, crs)
        RR = np.concatenate((RR,RR_i))
        DR = np.concatenate((DR,DR_i))
    if not rmin:
        rmin = compute_rmin(gdf_projected)
    if rmax:
        rmax = rmax
    else:
        rmax = np.max(DD)+1#find_rmax_meaningful(gdf_edge,crs,100000)
    if scale =="log":
        r_edges = np.logspace(np.log10(rmin), np.log10(rmax), nbins)
    elif scale == "lin":
        r_edges = np.linspace(rmin,rmax,nbins)
    hist_DD = binning_data(DD,nbins,r_edges)
    print("max DD", np.max(DD))
    #hist_DD = hist_DD[0:(len(hist_DD)-1)]
    del DD
    hist_DR = binning_data(DR,nbins,r_edges)
    del DR
    hist_RR = binning_data(RR,nbins,r_edges)
    del RR
    print(rmin,rmax)
    if len(hist_RR) == len(hist_DD)+1:
        hist_RR = hist_RR[0:(len(hist_RR)-1)]
    if len(hist_DR) == len(hist_DD)+1:
        hist_DR = hist_DR[0:(len(hist_DR)-1)]
    print(np.sum(hist_DD)*(2/(Nd*(Nd-1))))
    print(np.sum(hist_DR)/((Nd*(Nr))))
    print(np.sum(hist_RR)*(2/(Nr*(Nr_prime-1))))
    xi= compute_LS_correlation_2019(hist_DD,hist_DR,hist_RR,Nd,Nr_prime,Nr)
    return r_edges,xi

def compute_two_point_correlation_jack(gdf_projected,gdf_edge,crs,N_run,Nr_prime,rmin,rmax,Nd,nbins=20):
    Nr = Nr_prime*N_run
    #Nr = len(gdf_projected)
    DD = compute_DD(gdf_projected)
    DR,RR = compute_one_DR_RR(gdf_projected,gdf_edge, Nr_prime, crs)
    for i in range(N_run-1):
        print(i)
        DR_i,RR_i = compute_one_DR_RR(gdf_projected,gdf_edge, Nr_prime, crs)
        RR = np.concatenate((RR,RR_i))
        DR = np.concatenate((DR,DR_i))
    rmax = 1e5
    r_edges = np.logspace(np.log10(rmin), np.log10(rmax), nbins)
    r_edges = np.linspace(rmin,rmax,nbins)
    hist_DD = binning_data(DD,nbins,r_edges)
    hist_DD = hist_DD[0:(len(hist_DD)-1)]
    del DD
    hist_DR = binning_data(DR,nbins,r_edges)
    del DR
    hist_RR = binning_data(RR,nbins,r_edges)
    del RR
    print(rmin,rmax)
    if len(hist_RR) == len(hist_DD)+1:
        hist_RR = hist_RR[0:(len(hist_RR)-1)]
    if len(hist_DR) == len(hist_DD)+1:
        hist_DR = hist_DR[0:(len(hist_DR)-1)]
    print(np.sum(hist_DD)*(2/(Nd*(Nd-1))))
    print(np.sum(hist_DR)/((Nd*(Nr))))
    print(np.sum(hist_RR)*(2/(Nr*(Nr_prime-1))))
    xi= compute_LS_correlation_2019(hist_DD,hist_DR,hist_RR,Nd,Nr_prime,Nr)
    return r_edges,xi

def PCF_with_variance(gdf_projected,gdf_edge,crs,N_run,size,k,rmax,scale,nbins,rmin=False):
    l_xi = []
    #rmin = 0
    if len(gdf_projected) == 1:
        number_point = len(gdf_projected.explode())
    else:
        number_point = len(gdf_projected)
    for i in range(k):
        r_edges,xi = compute_two_point_correlation_2019(gdf_projected,gdf_edge,crs,N_run,size,rmin,number_point,rmax,scale,nbins=nbins)
        l_xi.append(xi)
    xi = np.mean(l_xi,axis=0)
    return r_edges,np.array(l_xi)

def PCF_with_variance_SP(gdf_projected,gdf_edge,crs,N_run,size,k,rmax,scale,nbins,rmin=False):
    l_xi = []
    #rmin = 0
    if len(gdf_projected) == 1:
        number_point = len(gdf_projected.explode())
    else:
        number_point = len(gdf_projected)
    for i in range(k):
        r_edges,xi = compute_two_point_correlation_2019_SP(gdf_projected,gdf_edge,crs,N_run,size,rmin,number_point,rmax,scale,nbins=nbins)
        l_xi.append(xi)
    xi = np.mean(l_xi,axis=0)
    return r_edges,np.array(l_xi)

# def compute_two_point_correlation_sobol(gdf_projected,gdf_edge,crs,N_run,Nr_prime,rmin,Nd,nbins=20):
#     #Nr = len(gdf_projected)
#     size = Nr_prime
#     DD = compute_DD(gdf_projected)
#     DR,RR,Nr_prime = compute_one_DR_RR(gdf_projected,gdf_edge, Nr_prime, crs)
#     for i in range(N_run-1):
#         print(i)
#         DR_i,RR_i,Nr_prime = compute_one_DR_RR(gdf_projected,gdf_edge,size, crs)
#         RR = np.concatenate((RR,RR_i))
#         DR = np.concatenate((DR,DR_i))
#     Nr = Nr_prime*N_run
#     rmin = compute_rmin(gdf_projected)
#     rmax = find_rmax_meaningful(gdf_edge,crs,100)
#     r_edges = np.logspace(np.log10(rmin), np.log10(rmax), nbins)
#    # r_edges = np.linspace(rmin,rmax,nbins)
#     hist_DD = binning_data(DD,nbins,r_edges)
#     hist_DD = hist_DD[0:(len(hist_DD)-1)]
#     del DD
#     hist_DR = binning_data(DR,nbins,r_edges)
#     del DR
#     hist_RR = binning_data(RR,nbins,r_edges)
#     del RR
#     print(rmin,rmax)
#     if len(hist_RR) == len(hist_DD)+1:
#         hist_RR = hist_RR[0:(len(hist_RR)-1)]
#     if len(hist_DR) == len(hist_DD)+1:
#         hist_DR = hist_DR[0:(len(hist_DR)-1)]
#     print(Nd)
#     print(np.sum(hist_DD)*(2/(Nd*(Nd-1))))
#     print(np.sum(hist_DR)/((Nd*(Nr))))
#     print(np.sum(hist_RR)*(2/(Nr*(Nr_prime-1))))
#     xi= compute_LS_correlation_2019(hist_DD,hist_DR,hist_RR,Nd,Nr_prime,Nr)
#     return r_edges,xi

    
        
# if __name__ == "__main__":
#     data_dir = Path("data")
#     N_run = 5
#     size = 10000
#     threshold = 10
#     name = "Italy"
#     rmin = 1000
#     nbins = 10
#     crs = crs_selector(name)
#     path_city = data_dir / "italy_cities.csv"
#     path_border = "data/map/Italy.geojson"
    
#     def func(x,x0,gamma):
#         return (x/x0)**(-gamma)
#     from scipy.stats import linregress
#     gdf_city = load_df_to_gdf(path_city,threshold)
#     gdf_edge = gpd.read_file(path_border)
#     gdf_projected = gdf_city.to_crs(crs)
    
#     r_edges,xi = compute_two_point_correlation(gdf_projected,gdf_edge,crs,N_run,size,rmin,nbins=nbins)
#     # mask = (xi>0) & (r_edges > 10000) & (r_edges < 5e5)  # for example
#     # r_fit, xi_fit = r_edges[mask], np.log10(xi[mask])
#     # slope, intercept, r_value, p_value, std_err = linregress(r_fit, xi_fit)
#     # gamma = -slope
#     # r0 = 10 ** (intercept / gamma)
#     # sns.set(style="whitegrid")
#     color = sns.color_palette("viridis", 1)[0]
#     lol = np.log10(r_edges)
#     width = np.concatenate((np.array([0.2]),np.diff(lol))) #need to find a way to encode properly the first alignement
#     plt.bar(lol,xi,width=width,align="center",edgecolor="black",color=color,alpha=0.8)
#     plt.xscale("linear")
#     plt.yscale("log")
#     plt.show()
#     # DD = compute_DD(gdf_projected)
#     # plt.hist(DD,bins=r_edges)
#     # plt.xscale("linear")
#     # plt.show()

