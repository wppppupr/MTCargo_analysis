import argparse
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
    style_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
except Exception:
    pass

def exp_decay(r, xi, a=1.0):
    """Exponential decay: a * exp(-r / xi) with zero offset (c=0)."""
    return a * np.exp(-r / xi)

def fit_correlation_length(distances_um, mean_corr, max_fit_dist=None):
    """
    Fit exponential decay C(r) = a * exp(-r / xi) (with c=0 fixed) to extract correlation length xi.
    Excludes NaN values.
    """
    mask = ~np.isnan(mean_corr)
    if max_fit_dist is not None:
        mask = mask & (distances_um <= max_fit_dist)

    x = np.array(distances_um)[mask]
    y = np.array(mean_corr)[mask]

    if len(x) < 3:
        return None, None

    # Estimate initial amplitude from first valid point (capped at 1.0)
    a_init = float(np.clip(y[0], 0.01, 1.0))
    p0 = [10.0, a_init]
    bounds = ([0.1, 0.0], [200.0, 1.5])

    try:
        popt, pcov = curve_fit(exp_decay, x, y, p0=p0, bounds=bounds)
        xi = popt[0]
        return xi, popt
    except Exception as e:
        return None, None

def load_condition_correlations(root_path, folder_name, particle_zarr="angular_correlation_w.zarr", bg_zarr="angular_correlation_bg.zarr"):
    """
    Load particle and background angular correlation zarr datasets across all experiments in a folder.
    Supports total, 1st principal component (parallel), and 2nd principal component (perpendicular).
    
    Returns:
    --------
    data_dict : dict of lists of xr.DataArray
    """
    root_path = Path(root_path)
    
    data = {
        'flow_total': [],
        'flow_par': [],
        'flow_perp': [],
        'bead_total': [],
        'bead_par': [],
        'bead_perp': [],
        'bg_total': [],
        'bg_par': [],
        'bg_perp': []
    }

    candidate_dirs = []
    base = root_path / folder_name
    if base.exists():
        for p in base.glob("*/*"):
            if p.is_dir() and ((p / particle_zarr).exists() or (p / bg_zarr).exists()):
                candidate_dirs.append(p)
        if not candidate_dirs:
            for p in base.glob("*"):
                if p.is_dir() and ((p / particle_zarr).exists() or (p / bg_zarr).exists()):
                    candidate_dirs.append(p)

    for input_dir in sorted(candidate_dirs):
        p_path = input_dir / particle_zarr
        bg_path = input_dir / bg_zarr

        if p_path.exists():
            ds_p = xr.open_zarr(str(p_path), consolidated=False)
            
            # Flow around particle
            if 'angular_correlation' in ds_p:
                data['flow_total'].append(ds_p.angular_correlation.mean(dim=['frame', 'particle']))
            if 'angular_correlation_parallel' in ds_p:
                data['flow_par'].append(ds_p.angular_correlation_parallel.mean(dim=['frame', 'particle']))
            if 'angular_correlation_perpendicular' in ds_p:
                data['flow_perp'].append(ds_p.angular_correlation_perpendicular.mean(dim=['frame', 'particle']))

            # Bead velocity vs surrounding flow
            if 'bead_correlation' in ds_p:
                data['bead_total'].append(ds_p.bead_correlation.mean(dim=['frame', 'particle']))
            if 'bead_correlation_parallel' in ds_p:
                data['bead_par'].append(ds_p.bead_correlation_parallel.mean(dim=['frame', 'particle']))
            if 'bead_correlation_perpendicular' in ds_p:
                data['bead_perp'].append(ds_p.bead_correlation_perpendicular.mean(dim=['frame', 'particle']))

        if bg_path.exists():
            ds_bg = xr.open_zarr(str(bg_path), consolidated=False)
            if 'angular_correlation' in ds_bg:
                data['bg_total'].append(ds_bg.angular_correlation.mean(dim=['frame']))
            if 'angular_correlation_parallel' in ds_bg:
                data['bg_par'].append(ds_bg.angular_correlation_parallel.mean(dim=['frame']))
            if 'angular_correlation_perpendicular' in ds_bg:
                data['bg_perp'].append(ds_bg.angular_correlation_perpendicular.mean(dim=['frame']))

    return data

def make_correlation_df(data_list, var_name='correlation', bead_radius_um=None, scale=0.11):
    """
    Convert list of xr.DataArray into a consolidated pandas DataFrame.
    Replace 0 values resulting from particle masking at small r with NaN.
    Zeros at large r (after values have risen/outside mask) are kept as 0.
    """
    df_list = []
    for exp_id, da in enumerate(data_list):
        dist_coord = 'distance' if 'distance' in da.coords else list(da.coords.keys())[0]
        distances = np.array(da[dist_coord])
        distances_um = distances * scale
        vals = np.array(da, dtype=float).copy()

        # Mask only leading near-zeros at small r (within bead mask or before first non-zero signal)
        for i in range(len(vals)):
            r_um = distances_um[i]
            is_zero = np.abs(vals[i]) < 1e-4 or np.isnan(vals[i])
            if bead_radius_um is not None:
                # If bead radius is specified, mask when r <= radius (or if still leading near-zero)
                if r_um <= (bead_radius_um + 0.1) and is_zero:
                    vals[i] = np.nan
                elif r_um > bead_radius_um and not is_zero:
                    # Exited mask and encountered real data -> stop masking
                    break
            else:
                # If radius not explicitly given, mask leading zeros from r=0 until first non-zero value
                if is_zero:
                    vals[i] = np.nan
                else:
                    break

        df_exp = pd.DataFrame({
            'distance': distances,
            var_name: vals,
            'exp': exp_id
        })
        df_list.append(df_exp)

    if not df_list:
        return pd.DataFrame(columns=['distance', var_name, 'exp'])
    return pd.concat(df_list, ignore_index=True)

def plot_angular_correlation(df, var_name='correlation', ax=None, label=None, marker='o', linestyle='-', scale=0.11, color=None, ylabel=None):
    """
    Plot angular correlation vs distance with SEM error bars (ignoring NaN values).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    distances_um = np.sort(df['distance'].unique()) * scale
    grouped = df.groupby('distance')
    # Use nan-ignoring statistics
    mean_corr = grouped[var_name].apply(lambda x: np.nan if x.dropna().empty else x.mean())
    sem_corr = grouped[var_name].apply(lambda x: np.nan if len(x.dropna()) <= 1 else x.sem())

    # Mask valid (non-NaN) points for plotting
    valid = ~mean_corr.isna()
    x_plot = distances_um[valid]
    y_plot = mean_corr[valid]
    yerr_plot = sem_corr[valid]

    ax.errorbar(x_plot, y_plot, yerr=yerr_plot, label=label, marker=marker, linestyle=linestyle, color=color, capsize=3)
    ax.set_xlabel(r'Distance $r$ [$\mu$m]')
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel(r'Angular Correlation $C(r)$')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

    return distances_um, mean_corr, sem_corr



def main():
    parser = argparse.ArgumentParser(description="Analyze and plot angular spatial correlation (Total, 1st PC / Parallel, 2nd PC / Perpendicular) across bead conditions.")
    parser.add_argument('--root_dir', type=str, default='/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads', help='Root directory')
    parser.add_argument('--conditions', type=str, nargs='+', default=['beads06um', 'beads1um', 'beads3um', 'beads5um', 'beads7um', 'beads20um'],
                        help='Bead conditions to plot')
    parser.add_argument('--marker_list', type=str, nargs='+', default=['^', 'o', 'd', 10,  11, 's'])
    #parser.add_argument('--color_list', type=str, nargs='+', default=[style_colors[0], style_colors[1], style_colors[2], style_colors[3], style_colors[4], style_colors[5], "#333333"])
    parser.add_argument('--save_fig', type=str, default='angular_correlation_summary.svg', help='Save figure path')
    args = parser.parse_args()

    root_path = Path(args.root_dir)
    if not root_path.exists():
        root_path = Path('/Volumes/data/Sasaki/MTsingleBeads')

    fig, axes = plt.subplots(3, 3, figsize=(18, 14), sharex=True)
    rows_titles = [
        ("Total Correlation", "total", r"$C(r)$"),
        ("1st Principal Component (Parallel to Nematic Axis)", "par", r"$C_\parallel(r)$"),
        ("2nd Principal Component (Perpendicular to Nematic Axis)", "perp", r"$C_\perp(r)$")
    ]

    # Helper to infer bead radius (um) from condition name (e.g., beads06um -> 0.3um, beads20um -> 10.0um)
    def get_bead_radius(cond_str):
        import re
        cond = str(cond_str).strip()
        radius_map = {
            'beads06um': 0.3,
            'beads1um': 0.5,
            'beads3um': 1.5,
            'beads5um': 2.5,
            'beads7um': 3.5,
            'beads20um': 10.0,
        }
        if cond in radius_map:
            return radius_map[cond]
        m = re.search(r'beads(\d+)(?:_?(\d+))?um', cond)
        if m:
            val_str = m.group(1)
            if val_str.startswith('0') and len(val_str) > 1:
                val = float(f"0.{val_str[1:]}")
            else:
                val = float(val_str)
            return val / 2.0
        return None

    results = []

    for condition, marker in zip(args.conditions, args.marker_list):
        data = load_condition_correlations(root_path, condition)
        has_data = any(len(v) > 0 for v in data.values())
        if not has_data:
            print(f"Skipping {condition}: no correlation data found.")
            continue

        n_p = len(data['flow_total'])
        n_bg = len(data['bg_total'])
        print(f"\nCondition: {condition} (particles: {n_p}, bg: {n_bg})")

        r_bead = get_bead_radius(condition)

        for row_idx, (row_label, comp_key, y_lbl) in enumerate(rows_titles):
            ax_flow = axes[row_idx, 0]
            ax_bead = axes[row_idx, 1]
            ax_bg = axes[row_idx, 2]

            # 1. Flow around particle (mask within bead radius)
            flow_list = data[f'flow_{comp_key}']
            if flow_list:
                df_f = make_correlation_df(flow_list, var_name='flow_corr', bead_radius_um=r_bead)
                dist_um, mean_c, _ = plot_angular_correlation(df_f, var_name='flow_corr', ax=ax_flow, label=condition, ylabel=y_lbl, marker=marker)
                xi_f, _ = fit_correlation_length(dist_um, mean_c, max_fit_dist=30.0)
                if xi_f is not None:
                    print(f"  -> [{row_label}] Flow Correlation Length xi = {xi_f:.2f} um")
                    results.append({'condition': condition, 'component': comp_key, 'type': 'flow_particle', 'xi_um': xi_f})

            # 2. Bead-flow correlation (mask within bead radius)
            bead_list = data[f'bead_{comp_key}']
            if bead_list:
                df_b = make_correlation_df(bead_list, var_name='bead_corr', bead_radius_um=r_bead)
                dist_um, mean_c, _ = plot_angular_correlation(df_b, var_name='bead_corr', ax=ax_bead, label=condition, ylabel=y_lbl, marker=marker)
                xi_b, _ = fit_correlation_length(dist_um, mean_c, max_fit_dist=30.0)
                if xi_b is not None:
                    print(f"  -> [{row_label}] Bead-Flow Correlation Length xi = {xi_b:.2f} um")
                    results.append({'condition': condition, 'component': comp_key, 'type': 'bead_flow', 'xi_um': xi_b})

            # 3. Background flow correlation (no bead mask)
            bg_list = data[f'bg_{comp_key}']
            if bg_list:
                df_bg = make_correlation_df(bg_list, var_name='bg_corr', bead_radius_um=None)
                dist_um, mean_c, _ = plot_angular_correlation(df_bg, var_name='bg_corr', ax=ax_bg, label=condition, ylabel=y_lbl, marker=marker)
                xi_bg, _ = fit_correlation_length(dist_um, mean_c, max_fit_dist=30.0)
                if xi_bg is not None:
                    print(f"  -> [{row_label}] BG Correlation Length xi = {xi_bg:.2f} um")
                    results.append({'condition': condition, 'component': comp_key, 'type': 'background', 'xi_um': xi_bg})

    # Titles and formatting
    axes[0, 0].set_title(r"$\bf{Flow\ Direction\ Correlation}$" + "\n" + r"$\langle \hat{\mathbf{u}}_{\mathrm{flow}}(0) \cdot \hat{\mathbf{u}}_{\mathrm{flow}}(r) \rangle$")
    axes[0, 1].set_title(r"$\bf{Bead\ Velocity\ Correlation}$" + "\n" + r"$\langle \hat{\mathbf{u}}_{\mathrm{bead}} \cdot \hat{\mathbf{u}}_{\mathrm{flow}}(r) \rangle$")
    axes[0, 2].set_title(r"$\bf{Background\ Flow\ Correlation}$" + "\n" + r"$\langle \hat{\mathbf{u}}_{\mathrm{bg}}(0) \cdot \hat{\mathbf{u}}_{\mathrm{bg}}(r) \rangle$")

    for r in range(3):
        for c in range(3):
            handles, labels = axes[r, c].get_legend_handles_labels()
            if handles:
                axes[r, c].legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    out_fig = root_path/"figure"/args.save_fig
    plt.savefig(out_fig, dpi=300)
    print(f"\n[SUCCESS] Summary plot saved to {out_fig.resolve()}")

    if results:
        df_res = pd.DataFrame(results)
        print("\n--- Correlation Length Summary ---")
        print(df_res.to_string(index=False))
        df_res.to_csv(root_path/"xi.csv", index=False)
        print("[SUCCESS] xi summary saved to xi.csv")

if __name__ == "__main__":
    main()
