#!/usr/bin/env python3
"""
scratch/plot_cargo_vs_virtual_flow_profile.py

実測貨物 vs 仮想点トレーサーの動径流速プロファイル比較プロット
|u|(r/xi)/u_inf を各サイズおよび重ね描きで比較し、
「受動粒子が低速域に集まるのか」「貨物が流れを乱しているのか」を決定づける。
"""

import sys
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.interpolate import interp1d

# パス設定
CURRENT_DIR = Path("/home/sasaki/MTCargo_analysis")
OUT_DIR = CURRENT_DIR / "figure" / "flow_profile_around_cargo"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# スタイル
style_path = CURRENT_DIR / 'libs' / 'my_style.mplstyle'
if style_path.exists():
    plt.style.use(str(style_path))

BEADS_ORDER = [
    ("beads06um", 0.63, 0.315, "#882255", "^", r"$0.63\,\mu\mathrm{m}$"),
    ("beads1um",  1.18, 0.590, "#CC6677", "o", r"$1.18\,\mu\mathrm{m}$"),
    ("beads3um",  3.37, 1.685, "#DDCC77", "d", r"$3.37\,\mu\mathrm{m}$"),
    ("beads5um",  5.00, 2.500, "#999933", "p", r"$5.00\,\mu\mathrm{m}$"),
    ("beads7um",  7.24, 3.620, "#117733", "h", r"$7.24\,\mu\mathrm{m}$"),
    ("beads20um", 20.0, 10.00, "#44AA99", "s", r"$20.0\,\mu\mathrm{m}$"),
]

def load_xi_dict():
    p = CURRENT_DIR / "figure" / "bg_angular_correlation" / "bg_angular_correlation_length_per_experiment.csv"
    xi_map = {}
    if p.exists():
        df = pd.read_csv(p)
        for _, r in df.iterrows():
            xi_map[str(r['exp_dir']).strip()] = float(r['xi_um'])
    return xi_map

def main():
    real_cache = CURRENT_DIR / "scratch" / "cargo_flow_profile_extended_cache.pkl"
    virt_cache = CURRENT_DIR / "scratch" / "virtual_tracer_flow_profile_cache.pkl"
    
    if not real_cache.exists() or not virt_cache.exists():
        print("Waiting for caches to be ready...")
        return
        
    with open(real_cache, "rb") as f:
        df_real = pickle.load(f)
    with open(virt_cache, "rb") as f:
        df_virt = pickle.load(f)
        
    xi_map = load_xi_dict()
    
    # 共通の r/xi グリッド
    r_xi_grid = np.linspace(0.0, 4.0, 100)
    
    summary_rows = []
    
    # 6パネルプロット作成: 各サイズについて Real vs Virtual
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=False)
    axes = axes.flatten()
    
    for idx, (b_name, diam, rad, col, marker, label) in enumerate(BEADS_ORDER):
        ax = axes[idx]
        
        sub_real = df_real[df_real['bead_name'] == b_name]
        sub_virt = df_virt[df_virt['bead_name'] == b_name]
        
        # Real curves
        real_curves = []
        for _, row in sub_real.iterrows():
            exp_dir = str(row['exp_dir']).strip()
            xi = xi_map.get(exp_dir, 8.5)
            r_um = np.array(row['r_edge_um'])
            norm_spd = np.array(row['norm_speed'])
            r_xi = r_um / xi
            f_interp = interp1d(r_xi, norm_spd, bounds_error=False, fill_value="extrapolate")
            real_curves.append(f_interp(r_xi_grid))
            
        real_curves = np.array(real_curves)
        real_mean = np.nanmean(real_curves, axis=0)
        real_sem = np.nanstd(real_curves, axis=0) / np.sqrt(len(real_curves)) if len(real_curves) > 1 else np.zeros_like(real_mean)
        
        # Virtual curves
        virt_curves = []
        for _, row in sub_virt.iterrows():
            exp_dir = str(row['exp_dir']).strip()
            xi = xi_map.get(exp_dir, 8.5)
            r_um = np.array(row['r_edge_um'])
            norm_spd = np.array(row['norm_speed'])
            r_xi = r_um / xi
            f_interp = interp1d(r_xi, norm_spd, bounds_error=False, fill_value="extrapolate")
            virt_curves.append(f_interp(r_xi_grid))
            
        virt_curves = np.array(virt_curves)
        virt_mean = np.nanmean(virt_curves, axis=0)
        virt_sem = np.nanstd(virt_curves, axis=0) / np.sqrt(len(virt_curves)) if len(virt_curves) > 1 else np.zeros_like(virt_mean)
        
        # Plot Real
        ax.plot(r_xi_grid, real_mean, color=col, lw=2.5, label=f"Real Cargo ({label})")
        ax.fill_between(r_xi_grid, real_mean - real_sem, real_mean + real_sem, color=col, alpha=0.25)
        
        # Plot Virtual
        ax.plot(r_xi_grid, virt_mean, color="black", ls="--", lw=2.0, label="Virtual Tracer (R_c mask)")
        ax.fill_between(r_xi_grid, virt_mean - virt_sem, virt_mean + virt_sem, color="gray", alpha=0.25)
        
        ax.axhline(1.0, color='gray', ls=':', lw=1.2, alpha=0.7)
        ax.set_title(f"$d = {diam}\,\mu\mathrm{{m}}$", fontsize=14, fontweight='bold')
        ax.set_xlim(0, 3.5)
        if diam < 15:
            ax.set_ylim(0.4, 1.2)
        else:
            ax.set_ylim(0.4, 2.0)
            
        ax.set_xlabel(r"Distance from edge $r/\xi$")
        ax.set_ylabel(r"Normalized speed $|u|(r)/u_\infty$")
        ax.legend(loc="lower right", frameon=True, fontsize=10)
        ax.grid(True, ls='--', alpha=0.3)
        
        # Record summary
        summary_rows.append({
            'bead_name': b_name,
            'diameter_um': diam,
            'real_edge_norm': real_mean[0],
            'real_edge_sem': real_sem[0],
            'virt_edge_norm': virt_mean[0],
            'virt_edge_sem': virt_sem[0],
            'diff_real_vs_virt': real_mean[0] - virt_mean[0]
        })
        
    plt.tight_layout()
    out_fig = OUT_DIR / "real_vs_virtual_flow_profile_6panels.png"
    plt.savefig(out_fig, dpi=300)
    plt.savefig(OUT_DIR / "real_vs_virtual_flow_profile_6panels.svg")
    plt.close()
    print(f"Saved 6-panel comparison figure to {out_fig}")
    
    # オーバーレイ図（全サイズ比較）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    
    for b_name, diam, rad, col, marker, label in BEADS_ORDER:
        sub_real = df_real[df_real['bead_name'] == b_name]
        sub_virt = df_virt[df_virt['bead_name'] == b_name]
        
        r_curves = []
        for _, row in sub_real.iterrows():
            xi = xi_map.get(str(row['exp_dir']).strip(), 8.5)
            r_curves.append(interp1d(np.array(row['r_edge_um']) / xi, row['norm_speed'], bounds_error=False, fill_value="extrapolate")(r_xi_grid))
        r_mean = np.nanmean(r_curves, axis=0)
        ax1.plot(r_xi_grid, r_mean, color=col, lw=2.2, label=label)
        
        v_curves = []
        for _, row in sub_virt.iterrows():
            xi = xi_map.get(str(row['exp_dir']).strip(), 8.5)
            v_curves.append(interp1d(np.array(row['r_edge_um']) / xi, row['norm_speed'], bounds_error=False, fill_value="extrapolate")(r_xi_grid))
        v_mean = np.nanmean(v_curves, axis=0)
        ax2.plot(r_xi_grid, v_mean, color=col, lw=2.2, label=label)
        
    for ax, title in [(ax1, "Real Cargo"), (ax2, "Lagrangian Virtual Tracer")]:
        ax.axhline(1.0, color='gray', ls=':', lw=1.2)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlim(0, 3.5)
        ax.set_ylim(0.4, 1.4)
        ax.set_xlabel(r"Distance from edge $r/\xi$")
        ax.set_ylabel(r"Normalized speed $|u|(r)/u_\infty$")
        ax.legend(loc="lower right", frameon=True, fontsize=10)
        ax.grid(True, ls='--', alpha=0.3)
        
    plt.tight_layout()
    out_overlay = OUT_DIR / "real_vs_virtual_flow_profile_overlay.png"
    plt.savefig(out_overlay, dpi=300)
    plt.savefig(OUT_DIR / "real_vs_virtual_flow_profile_overlay.svg")
    plt.close()
    print(f"Saved overlay figure to {out_overlay}")
    
    # サマリー保存
    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(OUT_DIR / "real_vs_virtual_flow_profile_summary.csv", index=False)
    print("Summary table:")
    print(df_sum.to_string())

if __name__ == "__main__":
    main()

