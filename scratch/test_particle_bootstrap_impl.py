import pandas as pd
import numpy as np
from pathlib import Path

df_fp = pd.read_pickle('scratch/footprint_heterogeneity_cache.pkl')
df_ext = pd.read_pickle('scratch/cargo_flow_profile_extended_cache.pkl')
ext_map = {Path(p).name: float(u) for p, u in zip(df_ext['exp_dir'], df_ext['u_inf'])}
df_fp['u_inf'] = df_fp['exp_dir'].map(ext_map)
df_fp['vpar_norm'] = df_fp['v_parallel'] / df_fp['u_inf']
df_fp['vvirt_norm'] = df_fp['u_fp'] / df_fp['u_inf']
df_fp['one_minus_phi'] = np.clip(1.0 - df_fp['phi_t'], 1e-5, 1.0)

# Quantile bins across all data
n_bins = 8
df_fp['qbin'], bin_edges = pd.qcut(df_fp['one_minus_phi'], q=n_bins, retbins=True)

# Particle bootstrap function
def bootstrap_particle_stats(sub_bin: pd.DataFrame, n_boot: int = 1000):
    """
    Bootstrap sampling over unique particles (clusters)
    """
    unique_pids = sub_bin['unique_pid'].unique()
    n_p = len(unique_pids)
    if n_p == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    
    # Pre-aggregate by particle to allow fast re-sampling
    part_means = sub_bin.groupby('unique_pid')[['vpar_norm', 'vvirt_norm', 'one_minus_phi']].mean()
    part_weights = sub_bin.groupby('unique_pid').size()
    
    vpar_vals = part_means['vpar_norm'].values
    vvirt_vals = part_means['vvirt_norm'].values
    w_vals = part_weights.values
    
    # Weighted mean
    orig_vpar = np.average(vpar_vals, weights=w_vals)
    orig_vvirt = np.average(vvirt_vals, weights=w_vals)
    orig_x = np.average(part_means['one_minus_phi'].values, weights=w_vals)
    
    if n_p == 1:
        return orig_x, orig_vpar, 0.0, orig_vvirt, 0.0, 1
    
    boot_vpar = []
    boot_vvirt = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        sample_idx = rng.choice(n_p, size=n_p, replace=True)
        w_sample = w_vals[sample_idx]
        if np.sum(w_sample) == 0:
            continue
        boot_vpar.append(np.average(vpar_vals[sample_idx], weights=w_sample))
        boot_vvirt.append(np.average(vvirt_vals[sample_idx], weights=w_sample))
        
    vpar_err = float(np.std(boot_vpar))
    vvirt_err = float(np.std(boot_vvirt))
    
    return orig_x, orig_vpar, vpar_err, orig_vvirt, vvirt_err, n_p

print("Testing bootstrap on beads1um:")
sub1 = df_fp[df_fp['bead_name'] == 'beads1um']
for qb, grp in sub1.groupby('qbin', observed=False):
    x_m, vp_m, vp_e, vv_m, vv_e, n_p = bootstrap_particle_stats(grp)
    print(f"Bin {qb}: n_p={n_p}, 1-Phi={x_m:.4f}, vpar={vp_m:.3f} +- {vp_e:.3f}, vvirt={vv_m:.3f} +- {vv_e:.3f}")
