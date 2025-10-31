import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial import distance
import seaborn as sns
data_dir = Path("data")

def load_df_to_gdf(path,threshold):
    df = pd.read_csv(path,low_memory=False)
    df = df.loc[df["population"] >= threshold]
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lng, df.lat), crs="EPSG:4326")
    return gdf
    
def generate_random_point(gdf_edge,size,crs):
    sample = gdf_edge.sample_points(size)
    gdf_projected = sample.to_crs(crs)
    coord = gdf_projected.get_coordinates().to_numpy()
    return coord

def compute_one_RR(coord_random):
    RR = np.triu(distance.cdist(coord_random,coord_random)).astype(np.int32)
    RR = np.ravel(RR)
    RR = RR[RR != 0]
    return RR

def compute_DD(coord_data):
    DD = np.triu(distance.cdist(coord_data,coord_data)).astype(np.int32)
    DD = np.ravel(DD)
    DD = DD[DD != 0]
    return DD

def compute_one_DR(coord_data,coord_random):
    DR = np.triu(distance.cdist(coord_data,coord_random)).astype(np.int32)
    DR = np.ravel(DR)
    DR = DR[DR != 0]
    return DR

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
        


N_run = 5
size = 10000
threshold = 5000
crs = 'EPSG:2154' #for france #"EPSG:32632"fro italy
path = data_dir / "france_cities.csv"


gpd_fr = gpd.read_file("/home/utente/Documenti/FAKE_WORLD/code/data/CNTR_RG_01M_2024_4326.geojson")
gpd_fr = gpd_fr.loc[gpd_fr["NAME_ENGL"]=="France"]
gdf_edge = gpd.clip(gpd_fr, (-5,40,10,52))

gdf_fr = load_df_to_gdf(path,threshold)
N_D = len(gdf_fr)
N_R = N_run*size
gdf_projected = gdf_fr.to_crs(crs)
coord_data = gdf_projected.get_coordinates().to_numpy()
DD = compute_DD(coord_data)
coord_random = generate_random_point(gdf_edge, size, crs)
RR = compute_one_RR(coord_random)
DR = compute_one_DR(coord_data, coord_random)
for i in range(N_run-1):
    coord_random = generate_random_point(gdf_edge, size, crs)
    RR = np.concatenate((RR,compute_one_RR(coord_random)))
    DR = np.concatenate((DR,compute_one_DR(coord_data,coord_random)))
    
nbins=20
rmin = 1000
rmax = np.max(DD)
print("lol")
r_edges = np.logspace(np.log10(rmin), np.log10(rmax), nbins)
hist_DD = binning_data(DD,nbins,r_edges)
DD_norm = hist_DD/len(DD) #normalized_count(hist_DD,N_D)
del DD
hist_DR = binning_data(DR,nbins,r_edges)
DR_norm = hist_DR/len(DR)
del DR
hist_RR = binning_data(RR,nbins,r_edges)
RR_norm = hist_RR/len(RR)
xi= compute_LS_correlation(DD_norm,DR_norm,RR_norm)
##PLOT Part
bins_center = np.zeros(len(xi))
for i in range(len(xi)-1):
    if i == 0:
        bins_center[i] = 0.5*(r_edges[i])
    else:
        bins_center[i] = 0.5*(r_edges[i]-r_edges[i-1])
#plt.plot(r_edges,xi)
sns.set(style="whitegrid")
color = sns.color_palette("viridis", 1)[0]
width = np.concatenate((np.array([200]),np.diff(r_edges))) #need to find a way to encode properly the first alignement
plt.bar(r_edges,xi,width=width,align="center",edgecolor="black",color=color,alpha=0.8)
plt.xscale("log")
plt.yscale("log")
plt.show()

# hist_DR = binning_data(DR,nbins,r_edges)
# hist_RR = binning_data(RR,nbins,r_edges)
# #sns.histplot(RR,stat="probability",binwidth=50000)

# xi = np.zeros(len(r_edges))
# for r in range(len(xi)):
#     DD_norm= hist_DD * 2/(N_D*(N_D-1))
#     DD_norm= hist_DD * 1/(N_R*N_D)
#     RR_norm= hist_RR * 2/(N_R*(N_R-1))
    





# sample = geo_clip.sample_points(10000)
# fig,ax = plt.subplots()
# geo_clip.plot(ax=ax)
# sample.plot(ax=ax,color="red",markersize=0.1)

# gdf_projected = sample.to_crs('EPSG:2154')
# coord = gdf_projected.get_coordinates().to_numpy()
# distance_matrix_cdist = np.triu(distance.cdist(coord,coord)).astype(np.int32)
# distance_matrix_cdist = np.ravel(distance_matrix_cdist)
# distance_matrix_cdist = distance_matrix_cdist[distance_matrix_cdist != 0]
# sns.histplot(distance_matrix_cdist,stat="probability",binwidth=20000)
