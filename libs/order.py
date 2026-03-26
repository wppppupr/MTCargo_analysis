import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from scipy.optimize import curve_fit
from joblib import Parallel, delayed
from tqdm import tqdm

import cal_vel
import MT_order as order
import fit_model

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate and plot order parameter for MTs and beads.")
    
    # Data paths
    parser.add_argument('--path_mt', type=str, required=True, 
                        help="Path to the MTs data (e.g., .zarr or .npy file)")
    parser.add_argument('--path_track', type=str, required=True, 
                        help="Path to the beads tracks CSV file")
    parser.add_argument('--out_dir', type=str, required=True, 
                        help="Output directory to save results")
    
    # Optional measurement parameters
    parser.add_argument('--scale', type=float, default=0.11, 
                        help="Scale (um/pixel). Default: 0.11")
    parser.add_argument('--interval', type=float, default=4.0, 
                        help="Frame interval. Default: 4.0")
    
    # Order calculation parameters
    parser.add_argument('--window_size_um', type=float, default=2.0, 
                        help="Window size in um. Default: 2.0")
    parser.add_argument('--overlap', type=float, default=0.5, 
                        help="Overlap fraction. Default: 0.5")
    parser.add_argument('--neighborhood_radius', type=int, default=2, 
                        help="Neighborhood radius for order calculation. Default: 2")
    parser.add_argument('--d', type=int, default=5, 
                        help="Parameter d for order calculation. Default: 5")
    parser.add_argument('--eccentricity_thresh', type=float, default=0.2, 
                        help="Eccentricity threshold. Default: 0.2")
    
    # System parameters
    parser.add_argument('--n_jobs', type=int, default=-1, 
                        help="Number of parallel jobs to run. Default: -1 (all cores)")
    
    return parser.parse_args()


def plot_hist(data, bins, density=True, xscale='linear', yscale='linear', 
              xlim=(0.0, 1.0), ylim=(1e-1, 5.0), fitting=False):
    """
    Plots the histogram of the computed order parameter data.
    """
    counts, bin_edges = np.histogram(data, bins, density=density)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Draw histogram
    ax.hist(data, bins=bin_edges, density=True, alpha=0.7, color='C0', edgecolor='black')
    
    if fitting:
        # Avoid zero values in log scale, and limit the fitting range
        indices = np.where((counts > 0) & (bin_edges[:-1] > 0) & (bin_edges[:-1] < 300))
        if len(indices[0]) > 0:
            try:
                popt, pcov = curve_fit(fit_model.gaussian, bin_edges[indices], counts[indices])
                ax.plot(bin_edges[:-1], fit_model.gaussian(bin_edges[:-1], *popt), 
                        color='red', linewidth=2, label='Gaussian Fit')
                print(f"Fitted parameters (mu, sigma): {popt}")
            except Exception as e:
                print(f"Curve fitting failed: {e}")
        else:
            print("Not enough points for fitting.")

    # Configure axes
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('Order Parameter $S$')
    ax.set_ylabel('$P(S)$')
    
    if fitting:
        ax.legend()

    plt.tight_layout()
    plt.show()


def compute_order_for_particle(particle, tracks, MTs, d, scale, window_size, overlap, 
                               neighborhood_radius, eccentricity_thresh):
    """
    Wrapper function to compute the order parameter for a single particle execution thread.
    """
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
            eccentricity_thresh=eccentricity_thresh
        )
    except Exception as e:
        print(f"Error processing particle {particle}: {e}")
        return []


def main():
    args = parse_args()
    
    # Prepare output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load MTs data (support both numpy and zarr if applicable)
    print(f"Loading MTs data from: {args.path_mt}")
    if args.path_mt.endswith('.zarr'):
        import zarr
        MTs = zarr.open(args.path_mt, mode='r')
    else:
        MTs = np.load(args.path_mt)
        
    # Load and process tracks
    print(f"Loading and processing tracks from: {args.path_track}")
    df_tracks = pd.read_csv(args.path_track)
    tracks = cal_vel.cal(df_tracks, scale=args.scale, frame_interval=args.interval)
    
    # Pre-calculate required tracking parameters
    window_size = int(args.window_size_um / args.scale)
    particles = sorted(set(tracks['particle']))
    
    print(f"Processing {len(particles)} particles across {args.n_jobs if args.n_jobs > 0 else 'all'} cores...")
    
    # Run in parallel across available cores with a progress bar
    results = Parallel(n_jobs=args.n_jobs, backend='loky')(
        delayed(compute_order_for_particle)(
            p, tracks, MTs, args.d, args.scale, window_size, 
            args.overlap, args.neighborhood_radius, args.eccentricity_thresh
        ) for p in tqdm(particles, desc='Calculating Order')
    )
    
    # Flatten multidimensional results
    all_orders = [item for sublist in results for item in sublist]
    all_orders_array = np.array(all_orders)
    
    # Save the aggregated order array
    output_path = os.path.join(args.out_dir, "order.npy")
    print(f"Saving order parameters to: {output_path}")
    np.save(output_path, all_orders_array)
    
    # Plot output histogram
    print("Plotting histogram...")
    bins = np.linspace(0.0, 1.0, 20)
    plot_hist(all_orders_array, bins=bins, yscale='log', fitting=False)


if __name__ == "__main__":
    main()
