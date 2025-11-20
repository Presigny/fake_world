import sklearn.cluster as sk
import itertools
import sys
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.metrics import silhouette_score
from pathlib import Path
import geopandas as gpd
path = Path().cwd() / "src" #to add the src directory to the path regognized by Python
sys.path.append(str(path))
import methods_two_point_correlation as mtpc
import pandas as pd
import numpy as np

def compute_jackknive_polygon(coord,l_result):
    df1 = pd.DataFrame(l_result,columns=["label"])
    df2 = coord.join(df1)
    df3 = df2[df2["label"] == 0]
    l_polygon = []
    for i in range(np.max(l_result)+1):
        df3 = df2[df2["label"] == i]
        #print("len(df3)",len(df3))
        points_gdf = gpd.GeoDataFrame(df3, geometry=gpd.points_from_xy(df3['x'], df3['y']))
        # convex_hull_polygon = points_gdf.union_all()
        # polygon = convex_hull_polygon#.convex_hull
        # geoserie = gpd.GeoSeries(polygon)
        #l_polygon.append(gpd.GeoDataFrame({'geometry': geoserie, 'df':[1]},crs=crs))
        l_polygon.append(points_gdf)
    return l_polygon

def clusterize_system(number_tested,coord,display=True):
    sse = []
    for k in range(1, number_tested): #use the kmean clustering from k=1 cluster to k=11 clusters
         kmeans = sk.KMeans(n_clusters=k,tol=1e-8,max_iter=1000,n_init=100).fit(coord)
         sse.append(kmeans.inertia_) #save the sum of squared error
    # Plot the of the elbow method
    if display:
        fig, ax = plt.subplots(figsize=(10,10))
        plt.style.use("fivethirtyeight")
        plt.plot(range(1, number_tested), sse)
        plt.xticks(range(1, number_tested))
        plt.xlabel("Number of Clusters")
        plt.ylabel("SSE")
        ax.set_aspect(1. / ax.get_data_ratio(), adjustable='box')
        #plt.savefig("kmeans_elbow_score.png",format="png",transparent=True)
        plt.show()
        plt.close()
    return kmeans,sse

def optimal_cluster_elbow(sse,coord):
    kl = KneeLocator(range(1, number_tested), sse, curve="convex", direction="decreasing") #locate the knee point on the figure
    print(kl.elbow)
    kmeans = sk.KMeans(n_clusters=kl.elbow,tol=1e-8,max_iter=1000,n_init=100).fit(coord)
    l_result = kmeans.labels_ #the assignement of each empirical network
    return l_result

def plot_cluster_color(gdf_edge,gdf_city,l_res):
    l_color = ["black","darkorange",'b','r','#25fde9','g','y','brown','orange','turquoise','grey',"pink",'#25fde9','g','y','brown','orange','turquoise','grey',"pink"]
    h = iter(l_res) # associate the cluster assignement of each network as
    color_coordinates = []
    coord = gdf_city.get_coordinates()
    for i in range(len(coord)):
        color_coordinates.append(l_color[next(h)])
    fig, ax = plt.subplots()
    gdf_edge.to_crs(crs).plot(ax=ax)
    gdf_city.plot(ax=ax,markersize=0.5,color=color_coordinates)
    xmin,ymin,xmax,ymax = gdf_city.total_bounds
    plt.xlim((xmin,xmax))
    plt.ylim((ymin,ymax))
    plt.show()


path = Path().cwd()
sys.path.append(str(path))
data_dir = Path("data/cities")
map_dir = Path("data/map")
threshold = 1
name = "Netherlands"
crs = mtpc.crs_selector(name)
path_city = Path.cwd()/ data_dir / f"{name}_cities.csv"
path_border = Path.cwd() / map_dir / f"{name}.geojson"
gdf_city = mtpc.load_df_to_gdf(path_city,threshold)
gdf_edge = gpd.read_file(path_border)
gdf_projected = gdf_city.to_crs(crs)
coord = gdf_projected.get_coordinates()
number_tested = 20


kmeans,sse=clusterize_system(number_tested,coord)
l_result = optimal_cluster_elbow(sse, coord)
l_polygon = compute_jackknive_polygon(coord,l_result) #breaks the l_result into smalle piece
plot_cluster_color(gdf_edge,gdf_city,l_result)
list_of_cluster = []
list_of_cluster.append(np.max(l_result)+1)
for i in range(len(l_polygon)):
    coord_polygon = l_polygon[i].get_coordinates()
    kmeans,sse=clusterize_system(number_tested,coord_polygon)
    #print(sse)
    l_result_polygon = optimal_cluster_elbow(sse, coord_polygon)
    plot_cluster_color(gdf_edge,l_polygon[i],l_result_polygon)
    list_of_cluster.append(np.max(l_result_polygon)+1)
    l_cluster = compute_jackknive_polygon(coord_polygon,l_result_polygon)
    for j in range(len(l_cluster)):
        coord_polygon = l_cluster[j].get_coordinates()
        kmeans,sse=clusterize_system(number_tested,coord_polygon)
        #print(sse)
        l_result_polygon = optimal_cluster_elbow(sse, coord_polygon)
        plot_cluster_color(gdf_edge,l_cluster[i],l_result_polygon)
        list_of_cluster.append(np.max(l_result_polygon)+1)
        
    



kmax = 20
iteration= 1
l_silhouette = []
for i in range(1):
    print(i)
    silhouette_coefficients = []
    for k in range(2, kmax):
        kmeans = sk.KMeans(n_clusters=k,tol=1e-8,max_iter=1000,n_init=100).fit(coord)
        score = silhouette_score(coord, kmeans.labels_)
        silhouette_coefficients.append(score)
    l_silhouette.append(silhouette_coefficients)
    

fig, ax = plt.subplots(figsize=(10,10))
plt.style.use("fivethirtyeight")
plt.plot(range(2, kmax), np.mean(l_silhouette,axis=0))
plt.errorbar(range(2, kmax), np.mean(l_silhouette,axis=0), yerr=np.std(l_silhouette,axis=0), fmt="o", color="r",ms=5)
plt.xticks(range(2, kmax))
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
ax.set_aspect(1. / ax.get_data_ratio(), adjustable='box')
#plt.savefig("kmeans_silhouette_score.png",format="png",transparent=True)
plt.show()