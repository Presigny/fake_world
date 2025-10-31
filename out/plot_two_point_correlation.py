import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import save_load_pickle as slp

local_dir = Path("France_1000_20")
load_dir = Path.cwd().parent / "data/two_point_correlation" / local_dir
i=1
fig, ax = plt.subplots()
sns.set(style="whitegrid")
l_xi = []
for item in load_dir.iterdir():
    d = slp.load_results(item)
    r_edges, xi = d["r_edges"],d["xi"]
    l_xi.append(xi)
l_xi = np.array(l_xi).transpose()
l_error = []
for i in range(len(l_xi)):
    l_error.append(3*np.std(l_xi[i]))
for item in load_dir.iterdir():
    d = slp.load_results(item)
    r_edges, xi = d["r_edges"],d["xi"]
    color = sns.color_palette("viridis", i)[0]
    i+=1
    width = np.concatenate((np.array([200]),np.diff(r_edges))) #need to find a way to encode properly the first alignement
    ax.bar(r_edges,xi,width=width,align="center",edgecolor="black",color=color,alpha=0.5)
plt.errorbar(r_edges, xi, yerr=l_error, fmt="o", color="r",ms=1)
plt.xscale("log")
plt.yscale("linear")
plt.show()
