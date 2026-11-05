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

# Check beads20um:
sub20 = df_fp[df_fp['bead_name'] == 'beads20um']
print(f"Total points in beads20um: {len(sub20)}")
print(f"Total particles in beads20um: {sub20['unique_pid'].nunique()}")
print(sub20.groupby('unique_pid').size())

# If min_points per bin is e.g. 30:
n_bins = 8
sub20['qbin'] = pd.qcut(sub20['one_minus_phi'], q=n_bins)
print("beads20um counts per bin:")
print(sub20.groupby('qbin', observed=False).size())
