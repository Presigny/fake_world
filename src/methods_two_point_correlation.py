"""
author: Charley Presigny
The library contains all the functions to perform the two-point correlation function analysis 
on datasets trasnform into Geopandas datasets
"""
# /// script
# dependencies = [
#  "numpy == 1.26.4",
#  "matplotlib == 3.8.4",
#  "scipy == 1.13.1",
#  "pandas == 2.2.3",
#  "geopandas == 1.0.1",
#  "
# ]
# ///
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.spatial import cKDTree
from scipy.spatial import ConvexHull
from scipy.stats import qmc
from tqdm import tqdm

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
    return radius


def load_df_to_gdf(path,threshold, threshold_variable = "population"):
    df = pd.read_csv(path,low_memory=False)
    df = df.loc[df[threshold_variable] >= threshold]
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lng, df.lat), crs="EPSG:4326")
    return gdf

def load_worm(path):
    df = pd.read_csv(path,low_memory=False)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y,df.z))
    return gdf


def generate_random_point(gdf_edge,size,crs=None,check_gpd=False):
    """Generate random points within the input borders
    Input: gdf_edge: Geopandas of the border of a country
    size: number of random points to generate
    crs: coordinate reference system (None: no projection, e.g. for the worm)
    check_gpd: if True, function will return the geopandas of the random point+border projected onto the crs
    Output: coord: numpy array pf coordinates of the random points in the crs    """

    sample = gdf_edge.sample_points(size)
    if crs is None:
        gdf_projected = sample
    else:
        gdf_projected = sample.to_crs(crs)
    coord = gdf_projected.get_coordinates().to_numpy()
    if check_gpd:
        return gdf_projected
    else:
        return coord

def compute_DD(gdf_projected, chunk_size=1000):
    """Distances between all pairs of points of gdf_projected, as int32.

    Same output as distance.pdist (same condensed order), computed point by
    point and by blocks of chunk_size to keep the memory footprint low.
    """
    coord = gdf_projected.get_coordinates().to_numpy()
    N = coord.shape[0]
    parts = []
    for i in range(N - 1):
        pi = coord[i:i+1]
        for j0 in range(i+1, N, chunk_size):
            j1 = min(N, j0 + chunk_size)
            dist_block = distance.cdist(pi, coord[j0:j1]).ravel()
            parts.append(dist_block.astype(np.int32))
    return np.concatenate(parts)


def compute_one_DR_RR(gdf_projected, gdf_edge, size, crs, chunk_size=1000):
    """Draw one random catalogue of `size` points inside gdf_edge and return
    DR (data-random distances, zeros removed) and RR (random-random
    distances, condensed order), both as 1D int32 arrays.

    Computed by blocks of chunk_size to keep the memory footprint low.
    """
    coord_random = generate_random_point(gdf_edge, size, crs)
    coord_data = gdf_projected.get_coordinates().to_numpy()
    N_data = coord_data.shape[0]
    N_rand = coord_random.shape[0]

    DR_parts = []
    for i0 in range(N_data):
        pi = coord_data[i0:i0+1]
        for j0 in range(0, N_rand, chunk_size):
            j1 = min(N_rand, j0 + chunk_size)
            d = distance.cdist(pi, coord_random[j0:j1]).ravel()
            DR_parts.append(d.astype(np.int32))
    DR = np.concatenate(DR_parts)
    DR = DR[DR != 0]

    RR_parts = []
    for i in range(N_rand - 1):
        pi = coord_random[i:i+1]
        for j0 in range(i+1, N_rand, chunk_size):
            j1 = min(N_rand, j0 + chunk_size)
            d = distance.cdist(pi, coord_random[j0:j1]).ravel()
            RR_parts.append(d.astype(np.int32))
    RR = np.concatenate(RR_parts)
    return DR, RR


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
    """Bin the data with every bins in r_edges by counting the occurence within 
    each particular bin of the data
    Output: array hist of the size of r_edges
    """
    indices = np.digitize(data, r_edges,right=True)
    hist = np.bincount(indices,minlength=nbins)
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


def crs_selector(name):
    """Give the standard coordinate reference system for each country which is important to have 
    precise distances between object in the countries
    Input - name: str of the country
    Output - str of the crs associated with the input country
    """
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
    if name == "Austria":
        return "EPSG:3416"
    if name == "Bulgaria":
        return "EPSG:9391"
    if name == "Croatia":
        return "EPSG:3766"
    if name == "Cyprus":
        return "EPSG:6312"
    if name == "Iceland":
        return "EPSG:9947"
    if name == "Greece":
        return "EPSG:2100"
    return 0

def compute_rmax(DD,DR,RR):
    rmax = [np.max(DD),np.max(DR),np.max(DD)]
    return np.max(rmax)+1 #plus one is to ensure we have the correct number of bins and not one added on top

def compute_rmin(gdf_projected):
    """Compute the average value between every points and there second nearest neighbours using KDTree
    It gives a minimum scale below which asessing the 2pcf is meaningless
    """
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
    xi= compute_LS_correlation(DD_norm,DR_norm,RR_norm)
    return r_edges,xi


def compute_LS_correlation_2019(DD,DR,RR,Nd,Nr_prime,Nr):
    """Compute the normalized Landy_Szalay estimator for the 2 point correlation function using
    the split random catalog scheme of Keihanen et al. 2019
    """
    return (Nr*(Nr_prime-1)*DD)/(Nd*(Nd-1)*RR)-(((Nr_prime-1)*DR)/(Nd*RR))+1

def compute_two_point_correlation_2019(gdf_projected,gdf_edge,crs,N_run,Nr_prime,rmin,Nd,rmax,scale,nbins=20):
    """Compute the two_point correlation function using the scheme of Keinahen et al. (2019 -https://doi.org/10.1051/0004-6361/201935828) 
    Input:
        gdf_projected: geopandas of the system of points we are interested in in the right crs
        gdf_edge: geopandas of the border of the system of points
        crs: str of coordinate reference system
        N_run: int of the number of time a random catalog is generated to compute 2pcf
        Nr_prime: Number of points in a single random catalog
        rmin: deprecated
        Nd: number of datapoints in system under investigation
        rmax: float maximal value below which the 2pcf will be computed
        scale: str linear or logscale at which the distances will be binned
        nbins: int number of bins between rmin and rmax
    Output:
        r-edges: list of distance bins with length equals nbins
        xi: array of shape (nbins) that stor for each distance bin the value of the 2pcf
    """
    Nr = Nr_prime*N_run # Number of points of the effective catalog
    #Nr = len(gdf_projected)
    DD = compute_DD(gdf_projected)
    DR,RR = compute_one_DR_RR(gdf_projected,gdf_edge, Nr_prime, crs)
    for i in range(N_run-1): #run over several random catalog of size Nr_prime
        DR_i,RR_i = compute_one_DR_RR(gdf_projected,gdf_edge, Nr_prime, crs)
        RR = np.concatenate((RR,RR_i)) #Accumulate the values in RR,DR
        DR = np.concatenate((DR,DR_i))
    if not rmin:
        rmin = compute_rmin(gdf_projected)
    if rmax:
        rmax = rmax
    else:
        rmax = np.max(DD)+1#max distance in the data
    if scale =="log": #choose a log scale for bining the distances
        r_edges = np.logspace(np.log10(rmin), np.log10(rmax), nbins)
    elif scale == "lin":
        r_edges = np.linspace(rmin,rmax,nbins)
    hist_DD = binning_data(DD,nbins,r_edges)
    #hist_DD = hist_DD[0:(len(hist_DD)-1)]
    del DD #supress the value after use to gain memory
    hist_DR = binning_data(DR,nbins,r_edges)
    del DR
    hist_RR = binning_data(RR,nbins,r_edges)
    del RR
    # Bring all histograms to the length of hist_DD (any length gap, not only
    # +-1 as before, e.g. shapes (19,) and (21,) used to crash).
    n = len(hist_DD)
    hist_RR = hist_RR[:n] if len(hist_RR) > n else np.append(hist_RR, np.zeros(n - len(hist_RR), dtype=hist_RR.dtype))
    hist_DR = hist_DR[:n] if len(hist_DR) > n else np.append(hist_DR, np.zeros(n - len(hist_DR), dtype=hist_DR.dtype))
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
    #hist_DD = hist_DD[0:(len(hist_DD)-1)]
    del DD
    hist_DR = binning_data(DR,nbins,r_edges)
    del DR
    hist_RR = binning_data(RR,nbins,r_edges)
    del RR
    if len(hist_RR) == len(hist_DD)+1:
        hist_RR = hist_RR[0:(len(hist_RR)-1)]
    if len(hist_DR) == len(hist_DD)+1:
        hist_DR = hist_DR[0:(len(hist_DR)-1)]
    xi= compute_LS_correlation_2019(hist_DD,hist_DR,hist_RR,Nd,Nr_prime,Nr)
    return r_edges,xi

def compute_two_point_correlation_jack(gdf_projected,gdf_edge,crs,N_run,Nr_prime,rmin,rmax,Nd,nbins=20):
    Nr = Nr_prime*N_run
    #Nr = len(gdf_projected)
    DD = compute_DD(gdf_projected)
    DR,RR = compute_one_DR_RR(gdf_projected,gdf_edge, Nr_prime, crs)
    for i in range(N_run-1):
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
    if len(hist_RR) == len(hist_DD)+1:
        hist_RR = hist_RR[0:(len(hist_RR)-1)]
    if len(hist_DR) == len(hist_DD)+1:
        hist_DR = hist_DR[0:(len(hist_DR)-1)]
    xi= compute_LS_correlation_2019(hist_DD,hist_DR,hist_RR,Nd,Nr_prime,Nr)
    return r_edges,xi

def PCF_with_variance(gdf_projected,gdf_edge,crs,N_run,size,k,rmax,scale,nbins,rmin=False):
    """Compute k different instances of the 2 points correlation function for gdf_projected system, 
    to build a statistical varaince on the 2 pcf. For each instance the system is compared to N_run random catalog
    with an effective size of N_run*size
    Input:
        gdf_projected: geopandas of the system of points we are interested in in the right crs
        gdf_edge: geopandas of the border of the system of points
        crs: str of coordinate reference system
        N_run: int of the number of time a random catalog is generated to compute 2pcf
        size: int size in number of points of each random catalog
        k:  int number of times to cpmpute the 2pcf to build a statistical variance
        rmax: float maximal value below which the 2pcf will be computed
        scale: str linear or logscale at which the distances will be binned
        nbins: int number of bins between rmin and rmax
    Output:
        r-edges: list of distance bins with length equals nbins
        l_xi: array of shape (nbins,k) that stor for each distance bin the value of the 2pcf
    """
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

def generate_sobol_point(gdf_edge,size,crs,check_gpd=False):
    sampler = qmc.Sobol(d=2, scramble=True)
    sobol_points = sampler.random(n=size)
    gdf_projected = gdf_edge.to_crs(crs)
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
    if check_gpd:
        return gdf_projected
    else:
        return coord



def compute_two_point_correlation_bootstrap(gdf_projected, gdf_edge, crs, N_run,
                                            Nr_prime, rmin, rmax, scale, nbins,
                                            Nsample, N_real):
    """2PCF on random subsamples, for large datasets.

    N_real times, Nsample points of gdf_projected are drawn without
    replacement and their 2PCF is computed with
    compute_two_point_correlation_2019.
    Output: r_edges, all_xi (array (N_real, nbins))
    """
    all_xi = []
    for _ in range(N_real):
        gdf_sample = gdf_projected.sample(Nsample, replace=False)
        r_edges, xi = compute_two_point_correlation_2019(
            gdf_sample, gdf_edge, crs, N_run, Nr_prime, rmin, Nsample,
            rmax, scale, nbins)
        all_xi.append(xi)
    return r_edges, np.vstack(all_xi)


def compute_two_point_correlation_density_groups(gdf_points, path_border_geojson,
                                                 name, param, Nr_prime, rmin,
                                                 rmax, Nsample,
                                                 mark="population_density"):
    """2PCF of groups of points sorted by a mark.

    Points are sorted by `mark`. Three groups of Nsample points are taken at
    the top of the ranking and three at the bottom, and the 2PCF of each of
    the 6 groups is computed (N_run, rmax, scale and nbins come from param).
    Output: r_edges, all_xi (array (6, nbins)), labels, mean mark of each group
    """
    N_run = param["N_run"]
    rmax = param["rmax"]
    scale = param["scale"]
    nbins = param["nbins"]

    crs = crs_selector(name)
    gdf_projected = gdf_points.to_crs(crs)
    gdf_edge = gpd.read_file(path_border_geojson)

    top = gdf_projected.sort_values(mark, ascending=False)
    bot = gdf_projected.sort_values(mark, ascending=True)
    groups = [top.iloc[0:Nsample], top.iloc[Nsample:2*Nsample],
              top.iloc[2*Nsample:3*Nsample], bot.iloc[2*Nsample:3*Nsample],
              bot.iloc[Nsample:2*Nsample], bot.iloc[0:Nsample]]
    labels = [f"Top {mark} (1st bin)", f"Top {mark} (2nd bin)",
              f"Top {mark} (3rd bin)", f"Low {mark} (3rd bin)",
              f"Low {mark} (2nd bin)", f"Low {mark} (1st bin)"]

    all_xi, l_av_bin_metadata = [], []
    for gdf_sample in tqdm(groups):
        r_edges, xi = compute_two_point_correlation_2019(
            gdf_sample, gdf_edge, crs, N_run, Nr_prime, rmin, len(gdf_sample),
            rmax, scale, nbins)
        all_xi.append(xi)
        l_av_bin_metadata.append(gdf_sample[mark].mean())
    return r_edges, np.vstack(all_xi), labels, l_av_bin_metadata


def compute_two_point_correlation_density_groups2(*args, **kwargs):
    """Same as compute_two_point_correlation_density_groups with mark="age"."""
    kwargs.setdefault("mark", "age")
    return compute_two_point_correlation_density_groups(*args, **kwargs)


# ===========================================================================
# 3D 2PCF (connectomes)
#
# Follows the scheme introduced for the worm (load_worm + crs=None), but
# really works in three dimensions. The 2D functions above cannot be used
# as they are, for two reasons:
#   - gdf.get_coordinates() returns (x, y) and silently drops z;
#   - gdf_edge.sample_points() draws in a planar polygon, so the random
#     catalogue does not fill the volume.
# Pairs are counted with a double KD-tree: for N ~ 2e5 points the distance
# matrix (2.8e10 pairs) does not fit in memory.
# ===========================================================================

def load_points_3d(path, columns=("x", "y", "z")):
    """CSV (or .zip holding a CSV) of positions -> GeoDataFrame of POINT Z."""
    df = pd.read_csv(path, low_memory=False)
    x, y, z = (df[c] for c in columns)
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(x, y, z))


def coords_3d(gdf):
    """Coordinates (N, 3) of a GeoDataFrame of POINT Z."""
    return gdf.get_coordinates(include_z=True).to_numpy(dtype=float)


def compute_rmin_nd(coords):
    """Mean nearest-neighbour distance, the scale below which the 2PCF is not
    meaningful. Duplicated positions (zero distance) are discarded."""
    d, _ = cKDTree(coords).query(coords, k=2, workers=-1)
    nn = d[:, 1]
    return float(nn[nn > 0].mean())


def sample_in_hull(hull, size, rng=None):
    """Uniform points in a convex hull, by rejection in the bounding box
    (hull.equations gives the half-spaces, A x + b <= 0 inside).
    Works in any dimension. It is the volume analogue of
    gdf_edge.sample_points(), which is restricted to the plane."""
    rng = np.random.default_rng() if rng is None else rng
    A, b = hull.equations[:, :-1], hull.equations[:, -1]
    lo, hi = hull.points.min(axis=0), hull.points.max(axis=0)
    accept = hull.volume / np.prod(hi - lo)
    out, n = [], 0
    while n < size:
        draw = int((size - n) / accept * 1.2) + 64
        P = rng.uniform(lo, hi, size=(draw, len(lo)))
        P = P[(P @ A.T + b <= 0).all(axis=1)]
        out.append(P)
        n += len(P)
    return np.vstack(out)[:size]


def pair_counts(A, r_edges, B=None):
    """Number of pairs per distance bin (length len(r_edges)-1).
    B=None gives the unordered internal pairs of A. Pairs below r_edges[0]
    (including self-pairs and duplicated positions) fall in the cumulative
    count removed by np.diff."""
    tree_A = cKDTree(A)
    if B is None:
        return np.diff(tree_A.count_neighbors(tree_A, r_edges)) / 2.0
    return np.diff(tree_A.count_neighbors(cKDTree(B), r_edges)).astype(float)


def compute_two_point_correlation_nd(coords, hull, r_edges, N_run=5,
                                     Nr_prime=None, rng=None):
    """Landy-Szalay 2PCF with the split random catalogues of Keihanen et al.
    (2019), the same formula as compute_LS_correlation_2019 applied to
    KD-tree pair counts. The dimension is the one of coords, so passing the
    (x, y) projection of a 3D dataset gives the projected 2PCF with exactly
    the same estimator.
    Input:
        coords: array (N, d) of positions
        hull: scipy.spatial.ConvexHull bounding the volume (role of gdf_edge)
        r_edges: edges of the distance bins
        N_run: number of random catalogues
        Nr_prime: size of one catalogue (default: as many as data points)
    Output:
        xi: array (len(r_edges)-1,)"""
    rng = np.random.default_rng() if rng is None else rng
    coords = np.asarray(coords, dtype=float)
    Nd = len(coords)
    Nr_prime = Nd if Nr_prime is None else Nr_prime
    DD = pair_counts(coords, r_edges)
    DR = np.zeros(len(r_edges) - 1)
    RR = np.zeros(len(r_edges) - 1)
    for _ in range(N_run):
        R = sample_in_hull(hull, Nr_prime, rng)
        DR += pair_counts(coords, r_edges, B=R)
        RR += pair_counts(R, r_edges)
    return compute_LS_correlation_2019(DD, DR, RR, Nd, Nr_prime, Nr_prime * N_run)


def PCF_with_variance_nd(coords, hull, r_edges, N_run=5, Nr_prime=None, k=10,
                         Nsample=None, rng=None):
    """k realisations of the 2PCF, to estimate its dispersion.
    Nsample subsamples the data at each realisation (needed beyond about
    1e5 points). The estimator is a ratio of pair densities, so it is
    invariant under random subsampling, up to noise.
    Output: array (k, len(r_edges)-1)"""
    rng = np.random.default_rng() if rng is None else rng
    coords = np.asarray(coords, dtype=float)
    l_xi = []
    for _ in range(k):
        if Nsample is None:
            sub = coords
        else:
            sub = coords[rng.choice(len(coords), Nsample, replace=False)]
        l_xi.append(compute_two_point_correlation_nd(sub, hull, r_edges,
                                                     N_run, Nr_prime, rng))
    return np.array(l_xi)
