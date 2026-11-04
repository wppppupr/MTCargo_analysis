#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(CURRENT_DIR))
import verify_cos_decomposition as vcd

ROOT_DIR = vcd.ROOT_DIR
SCALE = 0.11
DT = 4.0
V_UNIT = SCALE / DT

# Load contrast_inst_cache
df_inst = pd.read_pickle(CURRENT_DIR / "scratch" / "contrast_inst_cache.pkl")
df_inst['unique_pid'] = df_inst['exp_dir'] + '_' + df_inst['particle'].astype(str)

# Load velocities_mean.csv
vm_list = []
for b_item in vcd.BEADS_INFO:
    bname = b_item['name']
    exp_dirs = vcd.find_experiment_dirs(ROOT_DIR, bname)
    for ed in exp_dirs:
        vm_file = ed / "velocities_mean.csv"
        if vm_file.exists():
            df_vm = pd.read_csv(vm_file)
            df_vm['exp_dir'] = ed.name
            vm_list.append(df_vm[['exp_dir', 'frame', 'mean_velocity']])

df_vm_all = pd.concat(vm_list, ignore_index=True)
df_merged = pd.merge(df_inst, df_vm_all, on=['exp_dir', 'frame'], how='inner')

flow_norm_px = np.hypot(df_merged['flow_u'], df_merged['flow_v'])
valid = flow_norm_px > 0
df_merged = df_merged[valid].copy()

u_hat_x = df_merged['flow_u'] / flow_norm_px
u_hat_y = df_merged['flow_v'] / flow_norm_px

# v_parallel in um/s
df_merged['v_parallel'] = (df_merged['cargo_dx'] * u_hat_x + df_merged['cargo_dy'] * u_hat_y) * V_UNIT
# u_fp in um/s
df_merged['u_fp'] = flow_norm_px * V_UNIT
# u_px in um/s
df_merged['u_px'] = df_merged['mean_velocity']

np.random.seed(42)
N_BOOT = 2000

print("=== METHOD 1: Particle-level time means -> Bootstrap over particles ===")
for b_item in vcd.BEADS_INFO:
    bname = b_item['name']
    d_um = b_item['diameter_um']
    sub = df_merged[df_merged['bead_name'] == bname]
    rc_xi = sub['rc_over_xi'].mean()
    
    # Group by particle to get mean per particle
    p_means = sub.groupby('unique_pid')[['v_parallel', 'u_fp', 'u_px']].mean().reset_index()
    n_p = len(p_means)
    
    # Point estimate from particle-level unweighted mean
    X_mean = p_means['v_parallel'].mean()
    Y_mean = p_means['u_fp'].mean()
    Z_mean = p_means['u_px'].mean()
    
    eta_pt = X_mean / Y_mean
    A_pt = Y_mean / Z_mean
    tot_pt = X_mean / Z_mean
    
    # Bootstrap
    boot_eta = []
    boot_A = []
    boot_tot = []
    p_indices = np.arange(n_p)
    for _ in range(N_BOOT):
        idx = np.random.choice(p_indices, size=n_p, replace=True)
        sample = p_means.iloc[idx]
        xm = sample['v_parallel'].mean()
        ym = sample['u_fp'].mean()
        zm = sample['u_px'].mean()
        boot_eta.append(xm / ym)
        boot_A.append(ym / zm)
        boot_tot.append(xm / zm)
        
    eta_sem = np.std(boot_eta)
    eta_ci = (np.percentile(boot_eta, 2.5), np.percentile(boot_eta, 97.5))
    A_sem = np.std(boot_A)
    A_ci = (np.percentile(boot_A, 2.5), np.percentile(boot_A, 97.5))
    tot_sem = np.std(boot_tot)
    tot_ci = (np.percentile(boot_tot, 2.5), np.percentile(boot_tot, 97.5))
    
    print(f"\n{d_um:.2f} um (Rc/xi={rc_xi:.3f}, N_p={n_p}):")
    print(f"  A   = {A_pt:.4f} +/- {A_sem:.4f}  [95% CI: {A_ci[0]:.4f}, {A_ci[1]:.4f}]")
    print(f"  eta = {eta_pt:.4f} +/- {eta_sem:.4f}  [95% CI: {eta_ci[0]:.4f}, {eta_ci[1]:.4f}]")
    print(f"  tot = {tot_pt:.4f} +/- {tot_sem:.4f}  [95% CI: {tot_ci[0]:.4f}, {tot_ci[1]:.4f}]")
    print(f"  Verification: eta * A = {eta_pt * A_pt:.4f} vs tot = {tot_pt:.4f}")

print("\n=== METHOD 2: Particle resampling with frame aggregation (frame weighted) ===")
for b_item in vcd.BEADS_INFO:
    bname = b_item['name']
    d_um = b_item['diameter_um']
    sub = df_merged[df_merged['bead_name'] == bname]
    rc_xi = sub['rc_over_xi'].mean()
    p_ids = sub['unique_pid'].unique()
    n_p = len(p_ids)
    
    # Point estimate from all frames pooled
    X_mean = sub['v_parallel'].mean()
    Y_mean = sub['u_fp'].mean()
    Z_mean = sub['u_px'].mean()
    eta_pt = X_mean / Y_mean
    A_pt = Y_mean / Z_mean
    tot_pt = X_mean / Z_mean
    
    # Pre-group frames by particle for fast bootstrap
    p_dict = {pid: grp[['v_parallel', 'u_fp', 'u_px']].values for pid, grp in sub.groupby('unique_pid')}
    
    boot_eta = []
    boot_A = []
    boot_tot = []
    for _ in range(N_BOOT):
        sampled_pids = np.random.choice(p_ids, size=n_p, replace=True)
        sampled_arrs = [p_dict[pid] for pid in sampled_pids]
        cat = np.concatenate(sampled_arrs, axis=0)
        xm = np.mean(cat[:, 0])
        ym = np.mean(cat[:, 1])
        zm = np.mean(cat[:, 2])
        boot_eta.append(xm / ym)
        boot_A.append(ym / zm)
        boot_tot.append(xm / zm)
        
    eta_sem = np.std(boot_eta)
    eta_ci = (np.percentile(boot_eta, 2.5), np.percentile(boot_eta, 97.5))
    A_sem = np.std(boot_A)
    A_ci = (np.percentile(boot_A, 2.5), np.percentile(boot_A, 97.5))
    tot_sem = np.std(boot_tot)
    tot_ci = (np.percentile(boot_tot, 2.5), np.percentile(boot_tot, 97.5))
    
    print(f"\n{d_um:.2f} um (Rc/xi={rc_xi:.3f}, N_p={n_p}):")
    print(f"  A   = {A_pt:.4f} +/- {A_sem:.4f}  [95% CI: {A_ci[0]:.4f}, {A_ci[1]:.4f}]")
    print(f"  eta = {eta_pt:.4f} +/- {eta_sem:.4f}  [95% CI: {eta_ci[0]:.4f}, {eta_ci[1]:.4f}]")
    print(f"  tot = {tot_pt:.4f} +/- {tot_sem:.4f}  [95% CI: {tot_ci[0]:.4f}, {tot_ci[1]:.4f}]")
    print(f"  Verification: eta * A = {eta_pt * A_pt:.4f} vs tot = {tot_pt:.4f}")
