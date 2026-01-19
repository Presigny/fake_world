import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
import geopandas as gpd
from tqdm import tqdm
import random



def uniformity_test(positions_array, Rmax, plot = False, coeff = 1):
    """
    Test the uniformity of a set of points.

    Parameters
    ----------
    positions_array : list of lists
        A list of two lists, where the first list contains the x coordinates and
        the second list contains the y coordinates of the points.
    Rmax : float
        The maximum radius of the circle.
    plot : bool, optional
        If True, plot the data points. Default is False.
    coeff : float, optional
        The coefficient of the uniformity curve in the plot. Default is 1.

    Returns
    -------
    R_list, N_list : lists
        The list of radii and the list of counts of points within the circles.

    Notes
    -----
    This function tests the uniformity of a set of points. For each point, 
    it counts the number of points within a circle of radius R < Rmax. 
    The function then plots the data points if plot is True. 
    Otherwise, it returns the list of radii and the list of counts of points within the circles.
    """
    def count_points_within_window(points_data, center_point, radius):
        # Buffer the center point to create a circle with the given radius
        circle = center_point.buffer(radius)
        # Filter points that intersect with the clipped circle
        points_within_circle = points_data.within(circle.geometry.iloc[0])
        count = sum(points_within_circle) #number of point in the above red circle
        return count

    points = gpd.GeoSeries([Point(xy) for xy in zip(positions_array[0], positions_array[1])])
    df_gdf = gpd.GeoDataFrame(geometry=points)
    R_list = []
    N_list = []
    for i in tqdm(range(1000)):
        random_selection = df_gdf.sample(n=1)
        R = random.uniform(0,Rmax)  #Select a random radius <Rmax for the circle
        R_list+=[R/Rmax]
        N_list+=[count_points_within_window(df_gdf,random_selection,R)]
    
    if plot:
        plt.plot(R_list,N_list,'x',label='data points $N(R)$')
        plt.plot(R_list, [coeff*r**2 for r in R_list],label=r'$\propto R^2$')
        plt.xlabel('$R/R_{max}$')
        plt.ylabel('$N$')
        plt.loglog()
        plt.legend()
        plt.show()
    else:
        return R_list, N_list

def hyperuniformity_test(positions_array, Rmax, plot = False, coeff = 1):
    """
    Tests the hyperuniformity of a set of points using the variance of the counts of points 
    within circles of increasing radius.

    Parameters
    ----------
    positions_array : list of 2 lists
        The positions of the points in the list of lists, where the first list contains the x-coordinates 
        and the second list contains the y-coordinates.
    Rmax : float
        The maximum radius of the circles.
    plot : bool, optional
        If True, plots the data points. Defaults to False.
    coeff : float, optional
        The coefficient of the uniformity curve. Defaults to 1.

    Returns
    -------
    R_list : list
        The list of radii of the circles.
    sigma_squared : list
        The list of variances of the counts of points within the circles.
    """
    def count_points_within_window(points_data, center_point, radius):
        # Buffer the center point to create a circle with the given radius
        circle = center_point.buffer(radius)
        # Filter points that intersect with the clipped circle
        points_within_circle = points_data.within(circle.geometry.iloc[0])
        count = sum(points_within_circle) #number of point in the above red circle
        return count

    points = gpd.GeoSeries([Point(xy) for xy in zip(positions_array[0], positions_array[1])])
    df_gdf = gpd.GeoDataFrame(geometry=points)
    
    R_list = []
    sigma_squared = []
    for r in tqdm(np.arange(Rmax/1_000, Rmax, Rmax/1_000)):
        number_in_circle =[]
        while len(number_in_circle) < 100:
            random_selection = df_gdf.sample(n=1)
            number_in_circle += [count_points_within_window(df_gdf,random_selection,r)]
        mean_number = np.mean(number_in_circle)
        N_variance = []
        for n in number_in_circle:
            N_variance+=[(mean_number-n)**2]
        
        R_list+=[r/Rmax]
        sigma_squared+=[np.mean(N_variance)]


    if plot:
        plt.plot(R_list,sigma_squared,'x', label='data points $N(R)$')
        plt.plot(R_list, [coeff*r**2 for r in R_list], label=r'$\propto R^2$')
        plt.xlabel('$R/R_{max}$')
        plt.ylabel('$\sigma^2$')
        plt.loglog()
        plt.legend()
        plt.show()
    else:
        return R_list, sigma_squared
    

def _compute_C_and_D2_single(coords, r_values):
    """
    Fonction interne pour calculer C(r) et D2 sur un échantillon de coordonnées.
    """
    from scipy.spatial.distance import pdist

    distances = pdist(coords)
    n_pairs = len(distances)

    C_values = np.array([np.sum(distances < r) / n_pairs for r in r_values])

    # Fit log-log
    mask = (C_values > 0) & (r_values > 0)
    if np.sum(mask) < 3:
        return C_values, np.nan, np.nan

    log_r = np.log(r_values[mask])
    log_C = np.log(C_values[mask])

    D2, intercept = np.polyfit(log_r, log_C, 1)

    return C_values, D2, intercept


def correlation_dimension(name, data_type, path_points_csv, path_border_geojson,
                          threshold=1, threshold_variable="population_density",
                          n_scales=30, r_range=None,
                          bootstrap_size=10_000, n_bootstrap=10, plot=True):
    """
    Calcule la dimension de corrélation D2 d'une distribution spatiale de points
    en utilisant l'intégrale de corrélation C(r) avec bootstrap.

    Pour les datasets > bootstrap_size points, la fonction effectue n_bootstrap
    tirages aléatoires de bootstrap_size points et moyenne les résultats.

    Parameters
    ----------
    name : str
        Nom du pays (ex: "France", "Germany", "Switzerland")
    data_type : str
        Type de données (ex: "Gares", "Commune", "Routes")
    path_points_csv : Path
        Chemin vers le fichier CSV contenant les points
    path_border_geojson : Path
        Chemin vers le fichier GeoJSON contenant les frontières
    threshold : float, optional
        Seuil minimal pour la variable de filtrage. Default is 1.
    threshold_variable : str, optional
        Nom de la variable pour le filtrage. Default is "population_density".
    n_scales : int, optional
        Nombre d'échelles r testées. Default is 30.
    r_range : tuple(float, float), optional
        (r_min, r_max) en mètres. Si None, calculé automatiquement.
    bootstrap_size : int, optional
        Taille de chaque échantillon bootstrap. Default is 10_000.
    n_bootstrap : int, optional
        Nombre de répétitions bootstrap. Default is 10.
    plot : bool, optional
        Si True, trace C(r) vs r et le fit. Default is True.

    Returns
    -------
    r_values : np.ndarray
        Valeurs des rayons testés
    C_mean : np.ndarray
        Valeurs moyennes de l'intégrale de corrélation C(r)
    C_std : np.ndarray
        Écart-type de C(r) sur les bootstraps
    D2_mean : float
        Estimation moyenne de la dimension de corrélation
    D2_std : float
        Écart-type de D2 sur les bootstraps
    """
    from scipy.spatial.distance import pdist
    from pathlib import Path
    import sys
    path_src = Path().cwd().parent / "src"
    sys.path.append(str(path_src))
    import methods_two_point_correlation as mtpc

    # Chargement et projection des données
    crs = mtpc.crs_selector(name)
    gdf_points = mtpc.load_df_to_gdf(path_points_csv, threshold, threshold_variable)
    gdf_projected = gdf_points.to_crs(crs)

    N_total = len(gdf_projected)
    print(f"Dataset: {N_total} points")

    # Déterminer r_range sur un premier échantillon si non fourni
    if r_range is None:
        sample_for_range = gdf_projected.sample(min(bootstrap_size, N_total), replace=False)
        coords_sample = sample_for_range.get_coordinates().values
        distances_sample = pdist(coords_sample)
        r_min = np.percentile(distances_sample, 0.1)
        r_max = np.percentile(distances_sample, 90)
        del distances_sample
    else:
        r_min, r_max = r_range

    r_values = np.logspace(np.log10(r_min), np.log10(r_max), n_scales)

    # Si dataset petit, pas de bootstrap
    if N_total <= bootstrap_size:
        print(f"Dataset petit, calcul direct sur {N_total} points...")
        coords = gdf_projected.get_coordinates().values
        C_values, D2, intercept = _compute_C_and_D2_single(coords, r_values)

        # Estimation erreur par résidus
        mask = (C_values > 0) & (r_values > 0)
        log_r = np.log(r_values[mask])
        log_C = np.log(C_values[mask])
        residuals = log_C - (D2 * log_r + intercept)
        D2_std = np.std(residuals) / np.sqrt(len(log_r))

        print(f"Dimension de corrélation D2 = {D2:.3f} ± {D2_std:.3f}")

        if plot:
            _plot_correlation_dimension(name, data_type, r_values, C_values, None,
                                        D2, D2_std, intercept)

        return r_values, C_values, None, D2, D2_std

    # Bootstrap pour grands datasets
    print(f"Bootstrap: {n_bootstrap} échantillons de {bootstrap_size} points...")

    all_C = []
    all_D2 = []
    all_intercept = []

    for i in tqdm(range(n_bootstrap), desc="Bootstrap"):
        gdf_sample = gdf_projected.sample(bootstrap_size, replace=False)
        coords = gdf_sample.get_coordinates().values

        C_values, D2, intercept = _compute_C_and_D2_single(coords, r_values)

        if not np.isnan(D2):
            all_C.append(C_values)
            all_D2.append(D2)
            all_intercept.append(intercept)

    # Moyennes et écarts-types
    all_C = np.array(all_C)
    C_mean = np.mean(all_C, axis=0)
    C_std = np.std(all_C, axis=0)

    D2_mean = np.mean(all_D2)
    D2_std = np.std(all_D2)
    intercept_mean = np.mean(all_intercept)

    print(f"Dimension de corrélation D2 = {D2_mean:.3f} ± {D2_std:.3f}")

    if plot:
        _plot_correlation_dimension(name, data_type, r_values, C_mean, C_std,
                                    D2_mean, D2_std, intercept_mean)

    return r_values, C_mean, C_std, D2_mean, D2_std


def _plot_correlation_dimension(name, data_type, r_values, C_values, C_std,
                                 D2, D2_std, intercept):
    """Fonction interne pour tracer les résultats."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: C(r) vs r en log-log
    ax1 = axes[0]
    ax1.loglog(r_values, C_values, 'o-', label='C(r)', markersize=4)
    if C_std is not None:
        ax1.fill_between(r_values,
                         np.maximum(C_values - C_std, 1e-10),
                         C_values + C_std,
                         alpha=0.3, label='±1σ')
    ax1.set_xlabel('r (mètres)')
    ax1.set_ylabel('C(r)')
    ax1.set_title(f'{name} - {data_type}\nIntégrale de corrélation')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Fit linéaire en log-log
    ax2 = axes[1]
    mask = (C_values > 0) & (r_values > 0)
    log_r = np.log(r_values[mask])
    log_C = np.log(C_values[mask])

    ax2.plot(log_r, log_C, 'o', label='Données', markersize=4)
    ax2.plot(log_r, D2 * log_r + intercept, 'r--',
             label=f'Fit: D2 = {D2:.3f} ± {D2_std:.3f}')
    ax2.set_xlabel('ln(r)')
    ax2.set_ylabel('ln(C(r))')
    ax2.set_title('Régression log-log')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()


def correlation_dimension_from_gdf(name, data_type, gdf_points, path_border_geojson,
                                    n_scales=30, r_range=None,
                                    bootstrap_size=10_000, n_bootstrap=10, plot=True):
    """
    Version de correlation_dimension qui prend directement un GeoDataFrame en entrée.
    Utile pour les données générées (ex: modèle Soneira-Peebles).

    Parameters
    ----------
    name : str
        Nom du pays (pour le CRS)
    data_type : str
        Type de données (pour le titre du plot)
    gdf_points : GeoDataFrame
        GeoDataFrame contenant les points
    path_border_geojson : Path
        Chemin vers le fichier GeoJSON des frontières
    n_scales : int, optional
        Nombre d'échelles r testées. Default is 30.
    r_range : tuple(float, float), optional
        (r_min, r_max) en mètres. Si None, calculé automatiquement.
    bootstrap_size : int, optional
        Taille de chaque échantillon bootstrap. Default is 10_000.
    n_bootstrap : int, optional
        Nombre de répétitions bootstrap. Default is 10.
    plot : bool, optional
        Si True, trace les graphiques. Default is True.

    Returns
    -------
    r_values, C_mean, C_std, D2_mean, D2_std
    """
    from scipy.spatial.distance import pdist
    from pathlib import Path
    import sys
    path_src = Path().cwd().parent / "src"
    sys.path.append(str(path_src))
    import methods_two_point_correlation as mtpc

    crs = mtpc.crs_selector(name)
    gdf_projected = gdf_points.to_crs(crs)

    N_total = len(gdf_projected)
    print(f"Dataset: {N_total} points")

    # Déterminer r_range sur un premier échantillon si non fourni
    if r_range is None:
        sample_for_range = gdf_projected.sample(min(bootstrap_size, N_total), replace=False)
        coords_sample = sample_for_range.get_coordinates().values
        distances_sample = pdist(coords_sample)
        r_min = np.percentile(distances_sample, 0.1)
        r_max = np.percentile(distances_sample, 90)
        del distances_sample
    else:
        r_min, r_max = r_range

    r_values = np.logspace(np.log10(r_min), np.log10(r_max), n_scales)

    # Si dataset petit, pas de bootstrap
    if N_total <= bootstrap_size:
        print(f"Dataset petit, calcul direct sur {N_total} points...")
        coords = gdf_projected.get_coordinates().values
        C_values, D2, intercept = _compute_C_and_D2_single(coords, r_values)

        mask = (C_values > 0) & (r_values > 0)
        log_r = np.log(r_values[mask])
        log_C = np.log(C_values[mask])
        residuals = log_C - (D2 * log_r + intercept)
        D2_std = np.std(residuals) / np.sqrt(len(log_r))

        print(f"Dimension de corrélation D2 = {D2:.3f} ± {D2_std:.3f}")

        if plot:
            _plot_correlation_dimension(name, data_type, r_values, C_values, None,
                                        D2, D2_std, intercept)

        return r_values, C_values, None, D2, D2_std

    # Bootstrap pour grands datasets
    print(f"Bootstrap: {n_bootstrap} échantillons de {bootstrap_size} points...")

    all_C = []
    all_D2 = []
    all_intercept = []

    for i in tqdm(range(n_bootstrap), desc="Bootstrap"):
        gdf_sample = gdf_projected.sample(bootstrap_size, replace=False)
        coords = gdf_sample.get_coordinates().values

        C_values, D2, intercept = _compute_C_and_D2_single(coords, r_values)

        if not np.isnan(D2):
            all_C.append(C_values)
            all_D2.append(D2)
            all_intercept.append(intercept)

    all_C = np.array(all_C)
    C_mean = np.mean(all_C, axis=0)
    C_std = np.std(all_C, axis=0)

    D2_mean = np.mean(all_D2)
    D2_std = np.std(all_D2)
    intercept_mean = np.mean(all_intercept)

    print(f"Dimension de corrélation D2 = {D2_mean:.3f} ± {D2_std:.3f}")

    if plot:
        _plot_correlation_dimension(name, data_type, r_values, C_mean, C_std,
                                    D2_mean, D2_std, intercept_mean)

    return r_values, C_mean, C_std, D2_mean, D2_std


def strogatz_box_ratio(points, n_scales=20, eps_range=(0.5, 1e-3), plot=True):
    """
    Calcule et trace le ratio R(ε) = ln N(ε) / ln(1/ε)
    selon la méthode de Strogatz.
    
    Parameters
    ----------
    points : np.ndarray
        Tableau (2, N) contenant les coordonnées (x, y)
    n_scales : int
        Nombre d’échelles testées
    eps_range : tuple(float, float)
        (eps_max, eps_min)
    plot : bool
        Si True, trace R(ε) vs ε

    Returns
    -------
    epsilons : np.ndarray
        Valeurs des tailles de boîte
    ratios : np.ndarray
        Valeurs du ratio ln N(ε) / ln(1/ε)
    D_est : float
        Estimation moyenne de la dimension (pour petites ε)
    """
    assert points.shape[0] == 2, "Les points doivent être de forme (2, N)"
    # Normalisation dans [0,1]^2
    mins = points.min(axis=1, keepdims=True)
    ranges = np.ptp(points, axis=1, keepdims=True)
    pts = (points - mins) / ranges

    # Espacement logarithmique entre eps_max et eps_min
    eps_max, eps_min = eps_range
    epsilons = np.logspace(np.log10(eps_max), np.log10(eps_min), n_scales)
    N_boxes = []
    for eps in epsilons:
        idx = np.floor(pts / eps).astype(int)
        unique_boxes = np.unique(idx, axis=1)
        N_boxes.append(unique_boxes.shape[1])
    
    N_boxes = np.array(N_boxes)
    ratios = np.log(N_boxes) / np.log(1/epsilons)

    # Estimation : moyenne du ratio sur les petites échelles (3 dernières valeurs)
    D_est = np.mean(ratios[-3:])

    if plot:
        plt.figure(figsize=(6,4))
        plt.plot(epsilons, ratios, 'o-', label='R(ε) = ln N(ε)/ln(1/ε)')
        plt.axhline(D_est, color='r', linestyle='--', label=f'Mean small-scale = {D_est:.3f}')
        plt.xscale('log')
        plt.gca().invert_xaxis()
        plt.xlabel('ε (taille des boîtes)')
        plt.ylabel('R(ε)')
        plt.title("Convergence de la dimension de boîte (méthode de Strogatz)")
        plt.legend()
        plt.grid(True)
        plt.show()

    return epsilons, ratios, D_est