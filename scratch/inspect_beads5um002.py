import pandas as pd
import numpy as np

df_fp = pd.read_pickle('scratch/footprint_heterogeneity_cache.pkl')
sub5 = df_fp[df_fp['exp_dir'] == 'beads5um002'].sort_values('frame')
print(f"beads5um002 total frames: {len(sub5)}")
print(f"frame range: {sub5['frame'].min()} to {sub5['frame'].max()}")
print(f"particles: {sub5['particle'].unique()}")

for pid, grp in sub5.groupby('particle'):
    print(f"\nParticle {pid}: N={len(grp)}, frames={grp['frame'].min()}-{grp['frame'].max()}")
    print(f"   phi mean={grp['phi_t'].mean():.3f}, u_fp mean={grp['u_fp'].mean():.3f}, v_cargo mean={grp['v_cargo'].mean():.3f}")
    # High phi subset
    hp = grp[grp['phi_t'] >= 0.99]
    print(f"   high phi (>=0.99): N={len(hp)}, frame min={hp['frame'].min() if len(hp)>0 else None}, max={hp['frame'].max() if len(hp)>0 else None}, u_fp mean={hp['u_fp'].mean() if len(hp)>0 else None:.3f}")

