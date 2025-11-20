import geopandas as gpd
import sys
from pathlib import Path
path = Path().cwd() / "src" #to add the src directory to the path regognized by Python
sys.path.append(str(path))
import methods_two_point_correlation as mtpc
import save_load_pickle as slp
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.spatial import distance

def soneira_peebles_model(lamb,eta,L,R,erase_nodes=False):
    position = [(0,0)]
    final_position = []
    rng = np.random.default_rng()
    for i in range(L):
        l_pos = []
        for x,y in position:
            l_random = (R/(lamb**i))*rng.random((eta,2))
            for x_rand,y_rand in l_random:
                l_pos.append((x_rand-x,y_rand-y))
        position = np.copy(l_pos)
        final_position = final_position + l_pos
    if type(erase_nodes) == int and erase_nodes >=0:
        position = list(position)
        while erase_nodes != 0:
            index = rng.choice(range(len(position)))
            position.pop(index)
            erase_nodes -= 1
    position = np.array(position)
    print(len(position))        
    return position

country = "Ukraine"
out_dir = Path.cwd() /Path(f"out/2pcf/{country}")
file = out_dir / "1_5_7000_3__100000.0_lin_20"
dico_results = slp.load_results(file)

eta = 9 #density, number of circles by steps
l = 6 #fraction of space covered by the circles
L = 4  #size of circles, number of steps
gamma = 2 - np.log(eta)/np.log(l)
print("gamma", gamma)
print("fractal dimension",np.log(eta)/np.log(l))
print("l = ", l)
R = 1322000 # radius max

min_rad = R/(l**(L-1))
nbins=20
rmin=0
N_run= 10000
size=5
print("mimimum radius",min_rad )
position = soneira_peebles_model(l,eta,L,R)
distance_distribution = distance.pdist(position)
sns.histplot(distance_distribution)

##generate a Soenira