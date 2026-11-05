import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

df_fp = pd.read_pickle('scratch/footprint_heterogeneity_cache.pkl')
df_ext = pd.read_pickle('scratch/cargo_flow_profile_extended_cache.pkl')
ext_map = {Path(p).name: u for p, u in zip(df_ext['exp_dir'], df_ext['u_inf'])}

df_fp['u_inf'] = df_fp['exp_dir'].map(ext_map)
df_fp['vpar_norm'] = df_fp['v_parallel'] / df_fp['u_inf']
df_fp['v_virt_norm'] = df_fp['u_fp'] / df_fp['u_inf']
df_fp['one_minus_phi'] = np.clip(1.0 - df_fp['phi_t'], 1e-5, 1.0)

# Quantile bins across all beads
n_bins = 8
df_fp['qbin'], bin_edges = pd.qcut(df_fp['one_minus_phi'], q=n_bins, retbins=True)

print("Bin edges (1 - Phi):", bin_edges)

for bname in ['beads06um', 'beads1um', 'beads3um', 'beads5um', 'beads7um', 'beads20um']:
    sub = df_fp[df_fp['bead_name'] == bname]
    binned = sub.groupby('qbin', observed=False).agg({
        'one_minus_phi': 'mean',
        'phi_t': 'mean',
        'vpar_norm': ['mean', 'sem', 'count'],
        'v_virt_norm': ['mean', 'sem']
    })
    print(f"=== {bname} ===")
    print(binned)
