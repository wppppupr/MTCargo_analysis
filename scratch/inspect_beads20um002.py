import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_fp = pd.read_pickle('scratch/footprint_heterogeneity_cache.pkl')
sub20 = df_fp[df_fp['exp_dir'] == 'beads20um002'].sort_values('frame')
print(f"beads20um002 total frames: {len(sub20)}")
print(f"frame range: {sub20['frame'].min()} to {sub20['frame'].max()}")
print(f"phi_t stats: min={sub20['phi_t'].min():.3f}, median={sub20['phi_t'].median():.3f}, max={sub20['phi_t'].max():.3f}")
print(f"u_fp stats: min={sub20['u_fp'].min():.3f}, median={sub20['u_fp'].median():.3f}, max={sub20['u_fp'].max():.3f}")
print(f"v_cargo stats: min={sub20['v_cargo'].min():.3f}, median={sub20['v_cargo'].median():.3f}, max={sub20['v_cargo'].max():.3f}")

# Check correlation over time
print("\nFirst 10 frames:")
print(sub20[['frame', 'phi_t', 'u_fp', 'v_parallel', 'v_cargo']].head(10))

# Time segments: early (0-200), mid (200-400), late (400-end)
for f_min, f_max in [(0, 150), (150, 300), (300, 550), (550, 1000), (1000, 2000)]:
    seg = sub20[(sub20['frame'] >= f_min) & (sub20['frame'] < f_max)]
    if len(seg) > 0:
        print(f"Frames {f_min}-{f_max} (N={len(seg)}): phi={seg['phi_t'].mean():.3f}, u_fp={seg['u_fp'].mean():.3f}, v_cargo={seg['v_cargo'].mean():.3f}")
