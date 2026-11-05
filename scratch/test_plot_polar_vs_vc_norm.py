import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

CURRENT_DIR = Path(__file__).resolve().parent.parent

# Load data
df_fp = pd.read_pickle(CURRENT_DIR / 'scratch' / 'footprint_heterogeneity_cache.pkl')
df_ext = pd.read_pickle(CURRENT_DIR / 'scratch' / 'cargo_flow_profile_extended_cache.pkl')
ext_map = {Path(p).name: u for p, u in zip(df_ext['exp_dir'], df_ext['u_inf'])}

df_fp['u_inf'] = df_fp['exp_dir'].map(ext_map)
df_fp['vc_norm'] = df_fp['v_cargo'] / df_fp['u_inf']
df_fp['vpar_norm'] = df_fp['v_parallel'] / df_fp['u_inf']

# Bead ordering
bead_order = [
    {"name": "beads06um", "diameter_um": 0.63, "color": "#882255", "marker": "^", "label": r"$0.63\,\mu\mathrm{m}$"},
    {"name": "beads1um",  "diameter_um": 1.18, "color": "#CC6677", "marker": "o", "label": r"$1.18\,\mu\mathrm{m}$"},
    {"name": "beads3um",  "diameter_um": 3.37, "color": "#DDCC77", "marker": "d", "label": r"$3.37\,\mu\mathrm{m}$"},
    {"name": "beads5um",  "diameter_um": 5.00, "color": "#999933", "marker": "p", "label": r"$5.00\,\mu\mathrm{m}$"},
    {"name": "beads7um",  "diameter_um": 7.24, "color": "#117733", "marker": "h", "label": r"$7.24\,\mu\mathrm{m}$"},
    {"name": "beads20um", "diameter_um": 20.0, "color": "#44AA99", "marker": "s", "label": r"$20.0\,\mu\mathrm{m}$"},
]

print("Ready to generate plots.")
