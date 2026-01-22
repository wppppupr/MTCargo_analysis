import matplotlib.pyplot as plt
import pandas as pd
import cal_vel
import numpy as np
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import MT_order as order

path_MT = "/Volumes/data/Sasaki/MTsingleBeads/20260121/beads_trans_crop_crop/MTs.zarr"
path_track = "/Users/sasakinozomu/code/MTCargo_analysis/experiment/20260121/beads_trans_crop_crop/beads_tracks.csv"
output_name = "/Volumes/data/Sasaki/MTsingleBeads/20260121/beads_trans_crop_crop"

scale = 0.11
interval = 4

MTs = np.load(path_MT)
# trajectory_beads
tracks = cal_vel.cal(pd.read_csv(path_track), scale=scale ,frame_interval=interval)


# AFT parameters
window_size_um = 2
frame = 100

#### required parameters ####
window_size = int(window_size_um/scale)
overlap = 0.5
neighborhood_radius = 2

from joblib import Parallel, delayed
from tqdm import tqdm

d = 5

particles = sorted(set(tracks['particle']))

def _compute_order(particle):
    try:
        return order.get_order(
            tracks,
            particle,
            MTs,
            d,
            scale,
            window_size,
            overlap,
            neighborhood_radius,
            eccentricity_thresh=0.2
        )
    except Exception as e:
        print(f"Error processing particle {particle}: {e}")
        return []

# Run in parallel across available cores with a progress bar
results = Parallel(n_jobs=-1, backend='loky')(
    delayed(_compute_order)(p) for p in tqdm(particles, desc='particles')
)

# flatten results
all_orders = [item for sublist in results for item in sublist]

all_orders_array = np.array(all_orders)
np.save(f"{output_name}/order", all_orders_array)

from scipy.optimize import curve_fit
import fit_model

def plot_hist(data, bins, density = bool, xscale = 'linear', yscale = 'linear', xlim = (0.0, 1.0), ylim = (1e-1, 5.0), fitting = bool):
    counts, bin_edges = np.histogram(data, bins, density = density)
    
    fig, ax = plt.subplots()
    # ヒストグラムの棒グラフを描画
    ax.hist(data, bins=bin_edges, density=True)
    if fitting == True:
            indices = np.where((counts>0) & (bin_edges[:-1]>0) & (bin_edges[:-1]<300))
            popt, pcov = curve_fit(fit_model.gaussian, bin_edges[indices], counts[indices])
            ax.plot(bin_edges[:-1], fit_model.gaussian(bin_edges[:-1], popt[0], popt[1]))
            print(popt)

    ax.set(
            xscale = xscale,
            yscale = yscale,
            xlim = xlim,
            ylim = ylim,
            xlabel = 'Order Parameter $S$',
            ylabel = '$P(S)$'
            )
    ax.legend()

    plt.show()

plot_hist(all_orders_array, np.linspace(0.0, 1.0, 20), yscale = 'log')
