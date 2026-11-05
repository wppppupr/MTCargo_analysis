import pandas as pd
import numpy as np

df_fp = pd.read_pickle('scratch/footprint_heterogeneity_cache.pkl')
df_fp['one_minus_phi'] = np.clip(1.0 - df_fp['phi_t'], 1e-5, 1.0)
n_bins = 8
df_fp['qbin'], bin_edges = pd.qcut(df_fp['one_minus_phi'], q=n_bins, retbins=True)

for bname in ['beads06um', 'beads1um', 'beads3um', 'beads5um', 'beads7um', 'beads20um']:
    sub = df_fp[df_fp['bead_name'] == bname]
    counts = sub.groupby('qbin', observed=False).size()
    print(f"\n{bname} (total {len(sub)} points):")
    for b_idx, (cat, cnt) in enumerate(counts.items()):
        print(f"  Bin {b_idx} [{cat}]: {cnt} points")
