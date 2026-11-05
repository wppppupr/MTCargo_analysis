import pandas as pd
import numpy as np
from pathlib import Path

df_fp = pd.read_pickle('scratch/footprint_heterogeneity_cache.pkl')
df_ext = pd.read_pickle('scratch/cargo_flow_profile_extended_cache.pkl')
ext_map = {Path(p).name: float(u) for p, u in zip(df_ext['exp_dir'], df_ext['u_inf'])}
df_fp['u_inf'] = df_fp['exp_dir'].map(ext_map)
df_fp['vvirt_norm'] = df_fp['u_fp'] / df_fp['u_inf']
df_fp['one_minus_phi'] = np.clip(1.0 - df_fp['phi_t'], 1e-5, 1.0)

# Check beads5um and beads20um for phi > 0.99
for bname in ['beads5um', 'beads20um']:
    sub = df_fp[df_fp['bead_name'] == bname]
    high_phi = sub[sub['phi_t'] >= 0.99]
    print(f"\n==================== {bname} (phi >= 0.99, N={len(high_phi)}/{len(sub)}) ====================")
    for exp, grp in high_phi.groupby('exp_dir'):
        frames = grp['frame'].values
        pids = grp['particle'].unique()
        v_virt = grp['vvirt_norm'].values
        u_fp = grp['u_fp'].values
        u_inf = grp['u_inf'].iloc[0]
        print(f"  [{exp}] N_pts={len(grp)}, particles={pids}, u_inf={u_inf:.4f}")
        print(f"       frame min={frames.min()}, max={frames.max()}, median={np.median(frames)}, unique_frames={len(np.unique(frames))}")
        print(f"       u_fp mean={u_fp.mean():.4f}, max={u_fp.max():.4f}, v_virt_norm mean={v_virt.mean():.3f}, max={v_virt.max():.3f}")
        # Show frame distribution
        if len(frames) <= 20:
            print(f"       frames list: {sorted(list(frames))}")
        else:
            print(f"       frame percentiles: {np.percentile(frames, [0, 25, 50, 75, 100])}")
