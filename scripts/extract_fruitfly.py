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
from shapely.geometry import Polygon
import ast

# coming form this dataset: https://github.com/bluevex/elegans-atlas/blob/main/LowResAtlasWithHighResHeadsAndTails.csv

path_neurons = Path.cwd().parent/"data/fruitfly/coordinates.csv"

df = pd.read_csv(path_neurons,header=0)

df[["x", "y", "z"]] = (
    df["position"]
      .str.strip("[]")      # Remove the brackets
      .str.split(expand=True)  # Split on any whitespace
      .astype(int)
)

gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y,df.z))

polygon_geom = Polygon(gpd.points_from_xy(df.x, df.y,df.z))
polygon = gpd.GeoDataFrame(index=[0],geometry=[polygon_geom]) 
res = polygon.convex_hull

# path_map = Path.cwd().parent/"data/worm/"
# res.to_file(path_map/"c_elegans.geojson", driver="GeoJSON")
# path_neurons = Path.cwd().parent/"data/worm/"
# gdf.to_csv(path_neurons/"neurons.csv")



import plotly.graph_objects as go
from scipy.spatial import ConvexHull

# Extract points
points = np.column_stack((
    gdf.geometry.x,
    gdf.geometry.y,
    gdf.geometry.z
))

# Compute 3D convex hull
hull = ConvexHull(points)

# Hull triangular faces
faces = hull.simplices

# Create mesh
fig = go.Figure()

# Add original points
fig.add_trace(go.Scatter3d(
    x=points[:, 0],
    y=points[:, 1],
    z=points[:, 2],
    mode="markers",
    marker=dict(size=3),
    name="Points"
))

# Add convex hull surface
fig.add_trace(go.Mesh3d(
    x=points[:, 0],
    y=points[:, 1],
    z=points[:, 2],
    i=faces[:, 0],
    j=faces[:, 1],
    k=faces[:, 2],
    opacity=0.4,
    name="Convex Hull"
))

# Keep equal scale
fig.update_layout(
    scene=dict(
        aspectmode="data",
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z"
    ),
    title="3D Convex Hull"
)

fig.show(renderer="browser")