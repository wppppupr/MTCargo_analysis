import pandas as pd
import numpy as np

df_fp = pd.read_pickle('scratch/footprint_heterogeneity_cache.pkl')
df_ext = pd.read_pickle('scratch/cargo_flow_profile_extended_cache.pkl')
from pathlib import Path
ext_map = {Path(p).name: u for p, u in zip(df_ext['exp_dir'], df_ext['u_inf'])}
df_fp['u_inf'] = df_fp['exp_dir'].map(ext_map)
df_fp['vc_norm'] = df_fp['v_cargo'] / df_fp['u_inf']

for b in ['beads06um', 'beads1um', 'beads3um', 'beads5um', 'beads7um', 'beads20um']:
    sub = df_fp[df_fp['bead_name'] == b]
    print(f"=== {b} ===")
    print("Phi quantiles:", np.quantile(sub['phi_t'], [0.01, 0.05, 0.5, 0.95, 0.99]))
    print("vc/u_inf quantiles:", np.quantile(sub['vc_norm'], [0.01, 0.05, 0.5, 0.95, 0.99]))
