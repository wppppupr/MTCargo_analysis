import pandas as pd
import numpy as np

def bootstrap_eta_stats(sub_bin: pd.DataFrame, n_boot: int = 2000, seed: int = 42):
    unique_pids = sub_bin['unique_pid'].unique()
    n_p = len(unique_pids)
    if n_p == 0:
        return np.nan, np.nan, 0

    part_means = sub_bin.groupby('unique_pid')[['vpar_norm', 'vvirt_norm', 'one_minus_phi']].mean()
    part_weights = sub_bin.groupby('unique_pid').size().values

    vpar_vals = part_means['vpar_norm'].values
    vvirt_vals = part_means['vvirt_norm'].values
    x_vals = part_means['one_minus_phi'].values

    orig_vpar = float(np.average(vpar_vals, weights=part_weights))
    orig_vvirt = float(np.average(vvirt_vals, weights=part_weights))
    orig_x = float(np.average(x_vals, weights=part_weights))
    orig_eta = orig_vpar / orig_vvirt if orig_vvirt > 0 else np.nan

    if n_p == 1:
        return orig_x, orig_eta, 0.0, 1

    rng = np.random.default_rng(seed)
    boot_eta = []
    for _ in range(n_boot):
        idx = rng.choice(n_p, size=n_p, replace=True)
        w_sample = part_weights[idx]
        w_sum = np.sum(w_sample)
        if w_sum == 0:
            continue
        v_p = np.average(vpar_vals[idx], weights=w_sample)
        v_v = np.average(vvirt_vals[idx], weights=w_sample)
        if v_v > 0:
            boot_eta.append(v_p / v_v)

    eta_err = float(np.std(boot_eta, ddof=1)) if len(boot_eta) > 10 else 0.0
    return orig_x, orig_eta, eta_err, n_p

df_fp = pd.read_pickle('scratch/footprint_heterogeneity_cache.pkl')
from pathlib import Path
df_ext = pd.read_pickle('scratch/cargo_flow_profile_extended_cache.pkl')
ext_map = {Path(p).name: float(u) for p, u in zip(df_ext['exp_dir'], df_ext['u_inf'])}
df_fp['u_inf'] = df_fp['exp_dir'].map(ext_map)
df_fp['vpar_norm'] = df_fp['v_parallel'] / df_fp['u_inf']
df_fp['vvirt_norm'] = df_fp['u_fp'] / df_fp['u_inf']
df_fp['one_minus_phi'] = np.clip(1.0 - df_fp['phi_t'], 1e-5, 1.0)

print("Direct bootstrap eta on beads3um:")
sub3 = df_fp[df_fp['bead_name'] == 'beads3um']
sub3['qbin'] = pd.qcut(sub3['one_minus_phi'], q=8)
for qb, grp in sub3.groupby('qbin', observed=False):
    x_m, eta_m, eta_e, n_p = bootstrap_eta_stats(grp)
    print(f"Bin {qb}: n_p={n_p}, 1-Phi={x_m:.4f}, eta={eta_m:.3f} +- {eta_e:.3f}")
