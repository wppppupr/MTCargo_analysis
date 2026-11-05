import pandas as pd
import numpy as np

df_fp = pd.read_pickle('scratch/footprint_heterogeneity_cache.pkl')
df_ext = pd.read_pickle('scratch/cargo_flow_profile_extended_cache.pkl')
from pathlib import Path
ext_map = {Path(p).name: u for p, u in zip(df_ext['exp_dir'], df_ext['u_inf'])}
df_fp['u_inf'] = df_fp['exp_dir'].map(ext_map)
df_fp['vc_norm'] = df_fp['v_cargo'] / df_fp['u_inf']

# Test binning by Phi for beads1um
sub = df_fp[df_fp['bead_name'] == 'beads1um']
bins = np.linspace(0.0, 1.0, 11)
sub['phi_bin'] = pd.cut(sub['phi_t'], bins)
binned = sub.groupby('phi_bin', observed=False).agg(
    phi_mean=('phi_t', 'mean'),
    vc_mean=('vc_norm', 'mean'),
    vc_median=('vc_norm', 'median'),
    count=('vc_norm', 'count')
)
print("beads1um binned by phi:")
print(binned)
