import sklearn.cluster as sk
import itertools
import sys
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.metrics import silhouette_score
from pathlib import Path
import geopandas as gpd
import methods_two_point_correlation as mtpc
path = Path().cwd()
sys.path.append(str(path))
data_dir = Path("data")
threshold = 1000
name = "France"
crs = mtpc.crs_selector(name)
path_city = Path.cwd()/ data_dir / "france_cities.csv"
path_border = Path.cwd() / data_dir / "map/France.geojson"
gdf_city = mtpc.load_df_to_gdf(path_city,threshold)
gdf_edge = gpd.read_file(path_border)
gdf_projected = gdf_city.to_crs(crs)
coord = gdf_projected.get_coordinates()
number_tested = 100
sse = []
for k in range(1, number_tested): #use the kmean clustering from k=1 cluster to k=11 clusters
     kmeans = sk.KMeans(n_clusters=k,tol=1e-8,max_iter=1000).fit(coord)
     sse.append(kmeans.inertia_) #save the sum of squared error
# Plot the of the elbow method
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

kl = KneeLocator(range(1, number_tested), sse, curve="convex", direction="decreasing") #locate the knee point on the figure
print(kl.elbow)
l_color = ["black","darkslategrey","darkorange",'b','r','#25fde9','g','y','brown','orange','turquoise']
kmeans = sk.KMeans(n_clusters=kl.elbow,tol=1e-8,max_iter=1000).fit(coord)
l_result = kmeans.labels_ #the assignement of each empirical network
inertia = kmeans.inertia_ #variable to evaluate the quality of the cluster
h = iter(l_result) # associate the cluster assignement of each network as
color_coordinates = []
for i in range(len(coord)):
    color_coordinates.append(l_color[next(h)])
## Clustering using Agglomerative clustering algorithm,
# kmeans = sk.AgglomerativeClustering(n_clusters=n_cluster).fit(l_X)
# l_result = kmeans.labels_
# #inertia = kmeans.inertia_
# h = iter(l_result)
# for i in range(len(l_x)):
#     for j in range(len(l_x[i])):
#         label_cluster[i][j] = next(h)
# 
fig, ax = plt.subplots()
gdf_edge.plot(ax=ax)
gdf_city.plot(ax=ax,markersize=0.1,color=color_coordinates)
plt.show()
# silhouette_coefficients = []
# for k in range(2, number_tested):
#          kmeans = sk.KMeans(n_clusters=k,tol=1e-8,max_iter=1000).fit(coord)
#          score = silhouette_score(coord, kmeans.labels_)
#          silhouette_coefficients.append(score)

# fig, ax = plt.subplots(figsize=(10,10))
# plt.style.use("fivethirtyeight")
# plt.plot(range(2, number_tested), silhouette_coefficients)
# plt.xticks(range(2, number_tested))
# plt.xlabel("Number of Clusters")
# plt.ylabel("Silhouette Coefficient")
# ax.set_aspect(1. / ax.get_data_ratio(), adjustable='box')
# #plt.savefig("kmeans_silhouette_score.png",format="png",transparent=True)
# plt.show()