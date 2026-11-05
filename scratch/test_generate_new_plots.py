import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CURRENT_DIR = Path(__file__).resolve().parent.parent

df_fp = pd.read_pickle(CURRENT_DIR / 'scratch' / 'footprint_heterogeneity_cache.pkl')
df_ext = pd.read_pickle(CURRENT_DIR / 'scratch' / 'cargo_flow_profile_extended_cache.pkl')
ext_map = {Path(p).name: float(u) for p, u in zip(df_ext['exp_dir'], df_ext['u_inf'])}

df_fp['u_inf'] = df_fp['exp_dir'].map(ext_map)
df_fp['vpar_norm'] = df_fp['v_parallel'] / df_fp['u_inf']
df_fp['v_virt_norm'] = df_fp['u_fp'] / df_fp['u_inf']
df_fp['one_minus_phi'] = np.clip(1.0 - df_fp['phi_t'], 1e-5, 1.0)

# Quantile bins across all beads
n_bins = 10
df_fp['qbin'], bin_edges = pd.qcut(df_fp['one_minus_phi'], q=n_bins, retbins=True)
print("10 quantile bin edges:", bin_edges)

