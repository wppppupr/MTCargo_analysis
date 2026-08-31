import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import zarr
from scipy.optimize import curve_fit
import os
import sys
import glob
from pathlib import Path

# Add workspace directory to path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from libs import fit_model as fm

# Try to use project style if available
try:
    plt.style.use('libs/my_style.mplstyle')
except Exception:
    pass

def exp_decay(r, xi, a=1.0, c=0.0):
    """Exponential decay: a * exp(-r / xi) + c"""
    return a * np.exp(-r / xi) + c

def load_condition_correlations(root_path, folder_name, particle_zarr="angular_correlation_w.zarr", bg_zarr="angular_correlation_bg.zarr"):
    """
    Load particle and background angular correlation zarr datasets across all experiments in a folder.
    
    Parameters:
    -----------
    root_path : str or Path
        Root directory path.
    folder_name : str
        Condition folder name (e.g. 'beads1um').
    particle_zarr : str
        Particle zarr filename.
    bg_zarr : str
        Background zarr filename.
        
    Returns:
    --------
    particles : list of xr.DataArray
        Mean angular correlation for particles (averaged over frame and particle).
    bgs : list of xr.DataArray
        Mean angular correlation for background (averaged over frame).
    beads : list of xr.DataArray
        Mean bead-flow angular correlation (averaged over frame and particle).
    """
    root_path = Path(root_path)
    particles = []
    bgs = []
    beads = []

    pattern = str(root_path / folder_name / "*" / "*")
    for input_dir in sorted(glob.glob(pattern)):
        input_dir = Path(input_dir)
        if os.path.isfile(input_dir):
            continue

        p_path = input_dir / particle_zarr
        bg_path = input_dir / bg_zarr

        if p_path.exists():
            ds_p = xr.open_zarr(str(p_path), consolidated=False)
            if 'angular_correlation' in ds_p:
                p_mean = ds_p.angular_correlation.mean(dim=['frame', 'particle'])
                particles.append(p_mean)
            if 'bead_correlation' in ds_p:
                bead_mean = ds_p.bead_correlation.mean(dim=['frame', 'particle'])
                beads.append(bead_mean)

        if bg_path.exists():
            ds_bg = xr.open_zarr(str(bg_path), consolidated=False)
            if 'angular_correlation' in ds_bg:
                bg_mean = ds_bg.angular_correlation.mean(dim=['frame'])
                bgs.append(bg_mean)

    return particles, bgs, beads

def make_correlation_df(data_list, var_name='correlation'):
    """
    Convert list of xr.DataArray into a consolidated pandas DataFrame.
    """
    df_list = []
    for exp_id, da in enumerate(data_list):
        dist_coord = 'distance' if 'distance' in da.coords else list(da.coords.keys())[0]
        df_exp = pd.DataFrame({
            'distance': np.array(da[dist_coord]),
            var_name: np.array(da),
            'exp': exp_id
        })
        df_list.append(df_exp)

    if not df_list:
        return pd.DataFrame(columns=['distance', var_name, 'exp'])
    return pd.concat(df_list, ignore_index=True)

def plot_angular_correlation(df, var_name='correlation', ax=None, label=None, marker='o', linestyle='-', scale=0.11, color=None):
    """
    Plot angular correlation vs distance with SEM error bars.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing 'distance' and var_name.
    var_name : str
        Target column name (e.g. 'correlation' or 'bead_correlation').
    ax : matplotlib.axes.Axes, optional
        Matplotlib axis.
    label : str, optional
        Label for legend.
    scale : float
        Spatial scaling factor (um/pixel, default: 0.11).
        
    Returns:
    --------
    distances_um : np.ndarray
    mean_corr : pd.Series
    sem_corr : pd.Series
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    distances_um = np.sort(df['distance'].unique()) * scale
    grouped = df.groupby('distance')
    mean_corr = grouped[var_name].mean()
    sem_corr = grouped[var_name].sem()

    ax.errorbar(distances_um, mean_corr, yerr=sem_corr, label=label, marker=marker, linestyle=linestyle, color=color, capsize=3)
    ax.set_xlabel(r'Distance $r$ [$\mu$m]')
    ax.set_ylabel(r'Angular Correlation $C(r) = \langle \hat{\mathbf{u}}(0) \cdot \hat{\mathbf{u}}(r) \rangle$')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

    return distances_um, mean_corr, sem_corr

def fit_correlation_length(distances_um, mean_corr, max_fit_dist=None):
    """
    Fit exponential decay C(r) = exp(-r / xi) to extract correlation length xi.
    """
    mask = ~np.isnan(mean_corr)
    if max_fit_dist is not None:
        mask = mask & (distances_um <= max_fit_dist)

    x = np.array(distances_um)[mask]
    y = np.array(mean_corr)[mask]

    if len(x) < 3:
        return None, None

    try:
        popt, pcov = curve_fit(exp_decay, x, y, p0=[20.0, 1.0, 0.0], bounds=([0.1, 0.0, -1.0], [500.0, 2.0, 1.0]))
        xi = popt[0]
        return xi, popt
    except Exception as e:
        print(f"Fit failed: {e}")
        return None, None
