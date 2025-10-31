import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial import distance
import seaborn as sns


def load_df_to_gdf(path,threshold):
    df = pd.read_csv(path,low_memory=False)
    df = df.loc[df["population"] >= threshold]
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lng, df.lat), crs="EPSG:4326")
    return gdf
    
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
    DD = np.triu(distance.cdist(coord_data,coord_data)).astype(np.int32)
    DD = np.ravel(DD)
    DD = DD[DD != 0]
    return DD

def compute_one_DR_RR(gdf_projected,gdf_edge, size, crs):
    coord_random = generate_random_point(gdf_edge, size, crs)
    coord_data = gdf_projected.get_coordinates().to_numpy()
    DR = np.triu(distance.cdist(coord_data,coord_random)).astype(np.int32)
    DR = np.ravel(DR)
    DR = DR[DR != 0]
    RR = np.triu(distance.cdist(coord_random,coord_random)).astype(np.int32)
    RR = np.ravel(RR)
    RR = RR[RR != 0]
    return DR,RR

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
    return 0

def compute_rmax(DD,DR,RR):
    rmax = [np.max(DD),np.max(DR),np.max(DD)]
    return np.max(rmax)
    

def compute_two_point_correlation(gdf_projected,gdf_edge,crs,N_run,size,rmin,nbins=20):
    DD = compute_DD(gdf_projected)
    DR,RR = compute_one_DR_RR(gdf_projected,gdf_edge, size, crs)
    for i in range(N_run-1):
        DR_i,RR_i = compute_one_DR_RR(gdf_projected,gdf_edge, size, crs)
        RR = np.concatenate((RR,RR_i))
        DR = np.concatenate((DR,DR_i))
    rmax = compute_rmax(DD,DR,RR)
    r_edges = np.logspace(np.log10(rmin), np.log10(rmax), nbins)
    hist_DD = binning_data(DD,nbins,r_edges)
    DD_norm = hist_DD/len(DD) #normalized_count(hist_DD,N_D)
    del DD
    hist_DR = binning_data(DR,nbins,r_edges)
    DR_norm = hist_DR/len(DR)
    del DR
    hist_RR = binning_data(RR,nbins,r_edges)
    RR_norm = hist_RR/len(RR)
    del RR
    xi= compute_LS_correlation(DD_norm,DR_norm,RR_norm)
    return r_edges,xi
    
        
if __name__ == "__main__":
    data_dir = Path("data")
    N_run = 5
    size = 10000
    threshold = 5000
    name = "Italy"
    rmin = 1000
    nbins = 20
    crs = crs_selector(name)
    path_city = data_dir / "italy_cities.csv"
    path_border = "data/map/Italy.geojson"
    
    gdf_city = load_df_to_gdf(path_city,threshold)
    gdf_edge = gpd.read_file(path_border)
    gdf_projected = gdf_city.to_crs(crs)
    
    r_edges,xi = compute_two_point_correlation(gdf_projected,gdf_edge,crs,N_run,size,rmin,nbins=nbins)
    
    sns.set(style="whitegrid")
    color = sns.color_palette("viridis", 1)[0]
    width = np.concatenate((np.array([200]),np.diff(r_edges))) #need to find a way to encode properly the first alignement
    plt.bar(r_edges,xi,width=width,align="center",edgecolor="black",color=color,alpha=0.8)
    plt.xscale("log")
    plt.yscale("log")
    plt.show()
    
