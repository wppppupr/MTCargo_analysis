#!/usr/bin/env python3
"""
plot_heterogeneity_conditioned_eta.py

サイズ固定・局所不均一性条件づけ伝達効率解析:
サイズ間の比較だけでは排除できない単純な流体力学的粘性抵抗 (Stokes抵抗 gamma \propto R) の
影響を排し、同一サイズ内において足印内の局所流動の不均一さ H_t で条件づけたときの
伝達効率 eta = \sum v_\parallel / \sum |<u_Rc>| を比較する。

不均一さの指標 H_t (|u| を含まない方向ベースの量):
1. 局所極性秩序 Phi_t = || <u / |u|>_{Rc} || (方向の揃い具合: 0〜1)
2. 逆向きレーン占有率 f_opp = fraction( u . <u_Rc> < 0 ) (対向流の混ざり度: 0〜0.5)

誤差評価:
各ビーズサイズ・各ビンにおいて、粒子単位のクラスタ・ブートストラップ (N_boot=2000) で
比のサンプリング分布を生成し、SEM および 95% 信頼区間を付与。
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = Path("/mnt/NAS-Ebanaru/sasaki/MTsingleBeads")
if not ROOT_DIR.exists():
    ROOT_DIR = CURRENT_DIR

sys.path.append(str(CURRENT_DIR))
import verify_cos_decomposition as vcd

OUT_DIRS = [
    CURRENT_DIR / "figure" / "contrast_analysis",
    ROOT_DIR / "figure" / "contrast_analysis"
]

N_BOOT = 2000
SEED = 42


def setup_matplotlib():
    plt.rcParams.update({
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.family': 'sans-serif',
        'mathtext.fontset': 'dejavusans',
        'font.size': 12,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10.5,
        'figure.titlesize': 15,
        'axes.linewidth': 1.2,
        'lines.linewidth': 2.0,
        'grid.linewidth': 0.6,
        'grid.alpha': 0.35,
        'grid.linestyle': '--',
    })


def save_fig(fig, base_name):
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{base_name}.png", dpi=300, bbox_inches='tight')
        fig.savefig(out_dir / f"{base_name}.svg", format='svg', bbox_inches='tight')
    print(f"Saved figure: {base_name}.png and .svg")


def compute_binned_eta_stats(df_sub: pd.DataFrame, bin_col: str, bins: List[float], labels: List[str]):
    """
    同一サイズグループにおいて、bin_col に基づくビン分割を行い、
    粒子単位のブートストラップ (N_boot=2000) により平均の比 eta = sum(v_par) / sum(u_fp) を算出する。
    """
    df_sub = df_sub.copy()
    df_sub['bin'] = pd.cut(df_sub[bin_col], bins=bins, labels=labels, right=False)

    pids = df_sub['unique_pid'].unique()
    n_p = len(pids)

    # Precompute particle-level sums for each bin
    p_v = df_sub.groupby(['unique_pid', 'bin'], observed=False)['v_parallel'].sum().unstack(fill_value=0.0).values
    p_u = df_sub.groupby(['unique_pid', 'bin'], observed=False)['u_fp'].sum().unstack(fill_value=0.0).values

    # Bin-level total point estimate
    tot_v = np.sum(p_v, axis=0)
    tot_u = np.sum(p_u, axis=0)
    eta_pt = np.where(tot_u > 0, tot_v / tot_u, np.nan)

    # Bin sample count & bin center (mean of bin_col within bin)
    bin_counts = df_sub.groupby('bin', observed=False).size().values
    bin_centers = df_sub.groupby('bin', observed=False)[bin_col].mean().values

    # Fast particle bootstrap
    np.random.seed(SEED)
    boot_eta = []
    for _ in range(N_BOOT):
        idx = np.random.choice(n_p, size=n_p, replace=True)
        bv = np.sum(p_v[idx], axis=0)
        bu = np.sum(p_u[idx], axis=0)
        boot_eta.append(np.where(bu > 0, bv / bu, np.nan))

    boot_eta = np.array(boot_eta)
    eta_sem = np.nanstd(boot_eta, axis=0)
    ci_low = np.nanpercentile(boot_eta, 2.5, axis=0)
    ci_high = np.nanpercentile(boot_eta, 97.5, axis=0)

    rows = []
    for i, l in enumerate(labels):
        if bin_counts[i] >= 10:  # require at least 10 samples
            rows.append({
                'bin_label': l,
                'bin_center': bin_centers[i],
                'count': bin_counts[i],
                'eta': eta_pt[i],
                'eta_sem': eta_sem[i],
                'ci_low': ci_low[i],
                'ci_high': ci_high[i]
            })

    return pd.DataFrame(rows)


def main():
    setup_matplotlib()

    cache_file = CURRENT_DIR / "scratch" / "footprint_heterogeneity_cache.pkl"
    if not cache_file.exists():
        print(f"Error: {cache_file} not found. Run extract_footprint_heterogeneity.py first.")
        sys.exit(1)

    df = pd.read_pickle(cache_file)
    print(f"Loaded {len(df)} records from {cache_file.name}")

    # Bins definition
    # 1. Local polar order Phi_t
    phi_bins = [0.0, 0.60, 0.80, 0.95, 1.0001]
    phi_labels = ['Phi < 0.60 (Disordered)', '0.60 <= Phi < 0.80', '0.80 <= Phi < 0.95', 'Phi >= 0.95 (Uniform)']

    # 2. Opposite flow fraction f_opp
    fopp_bins = [-0.001, 0.02, 0.12, 0.25, 0.60]
    fopp_labels = ['f < 0.02 (Pure Forward)', '0.02 <= f < 0.12 (Weak Counter)', '0.12 <= f < 0.25 (Moderate)', 'f >= 0.25 (Strong Counter)']

    phi_results = {}
    fopp_results = {}

    print("\n=== CONDITIONING ON LOCAL POLAR ORDER Phi_t ===")
    for b_item in vcd.BEADS_INFO:
        bname = b_item['name']
        sub = df[df['bead_name'] == bname]
        if len(sub) == 0:
            continue
        res_phi = compute_binned_eta_stats(sub, 'phi_t', phi_bins, phi_labels)
        res_fopp = compute_binned_eta_stats(sub, 'f_opp', fopp_bins, fopp_labels)
        phi_results[bname] = res_phi
        fopp_results[bname] = res_fopp

        d_um = b_item['diameter_um']
        print(f"\n--- {b_item['label']} (d={d_um} um, N_p={sub['unique_pid'].nunique()}, N_pts={len(sub)}) ---")
        print("  [Phi_t conditioning]")
        for _, r in res_phi.iterrows():
            print(f"    {r['bin_label']:26s} (center={r['bin_center']:.3f}, N={r['count']:4d}): eta = {r['eta']:.4f} +/- {r['eta_sem']:.4f}")
        print("  [f_opp conditioning]")
        for _, r in res_fopp.iterrows():
            print(f"    {r['bin_label']:26s} (center={r['bin_center']:.3f}, N={r['count']:4d}): eta = {r['eta']:.4f} +/- {r['eta_sem']:.4f}")

    # =========================================================================
    # Figure 10: eta vs Local Polar Order Phi_t (Fixed-size comparison)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    for b_item in vcd.BEADS_INFO:
        bname = b_item['name']
        if bname not in phi_results or len(phi_results[bname]) == 0:
            continue
        res = phi_results[bname]
        is_20um = (b_item['diameter_um'] >= 15.0)

        # Plot points and lines
        fmt = f"{b_item['marker']}-"
        mfc = 'white' if is_20um else b_item['color']
        ax.errorbar(
            res['bin_center'], res['eta'],
            yerr=[res['eta'] - res['ci_low'], res['ci_high'] - res['eta']],
            fmt=fmt, color=b_item['color'], ecolor=b_item['color'],
            mfc=mfc, mec=b_item['color'], mew=2.0 if is_20um else 1.0,
            markersize=8, linewidth=2.0, capsize=4, capthick=1.2,
            label=f"{b_item['label']} ($R_c/\\xi = {df[df['bead_name']==bname]['rc_over_xi'].mean():.2f}$)",
            zorder=3 if is_20um else 5
        )

    ax.axhline(0.0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=1.0, alpha=0.7, label='Ideal Perfect Coupling ($y=1$)')
    ax.set_xlabel(r'Local Footprint Polar Order $\Phi_t = \|\langle\hat{\mathbf{u}}\rangle_{R_c}\|$ (Order of Flow Direction)', fontsize=13, labelpad=8)
    ax.set_ylabel(r'Dynamical Transfer Efficiency $\eta = \frac{\sum v_\parallel}{\sum |\langle\mathbf{u}\rangle_{R_c}|}$', fontsize=13, labelpad=8)
    ax.set_title('Decoupling Within the Same Size Group:\n'
                 r'Transfer Efficiency $\eta$ Conditioned on Footprint Polar Order $\Phi_t$',
                 fontsize=14, fontweight='bold', pad=12)
    ax.set_xlim(0.30, 1.02)
    ax.set_ylim(-0.15, 0.70)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='upper left', frameon=True, framealpha=0.92, fontsize=10.0)

    # Annotation text explaining the key physical finding
    ax.text(
        0.98, 0.05,
        'Key Physics:\n'
        r'$\bullet$ Size fixed $\rightarrow$ Stokes drag $\gamma$ is identical' + '\n'
        r'$\bullet$ When local flow is disordered ($\Phi_t < 0.6$), $\eta \to 0$' + '\n'
        r'$\bullet$ When local flow aligns ($\Phi_t \to 1$), $\eta$ is preserved',
        transform=ax.transAxes, ha='right', va='bottom',
        fontsize=9.5, bbox=dict(boxstyle='round,pad=0.5', fc='#F8FAFC', ec='#475569', alpha=0.95, lw=1.2)
    )

    plt.tight_layout()
    save_fig(fig, "contrast_fig10_eta_vs_local_polar_order")
    plt.close(fig)

    # =========================================================================
    # Figure 11: eta vs Opposite Flow Fraction f_opp (Fixed-size comparison)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    for b_item in vcd.BEADS_INFO:
        bname = b_item['name']
        if bname not in fopp_results or len(fopp_results[bname]) == 0:
            continue
        res = fopp_results[bname]
        is_20um = (b_item['diameter_um'] >= 15.0)

        fmt = f"{b_item['marker']}-"
        mfc = 'white' if is_20um else b_item['color']
        ax.errorbar(
            res['bin_center'], res['eta'],
            yerr=[res['eta'] - res['ci_low'], res['ci_high'] - res['eta']],
            fmt=fmt, color=b_item['color'], ecolor=b_item['color'],
            mfc=mfc, mec=b_item['color'], mew=2.0 if is_20um else 1.0,
            markersize=8, linewidth=2.0, capsize=4, capthick=1.2,
            label=f"{b_item['label']} ($R_c/\\xi = {df[df['bead_name']==bname]['rc_over_xi'].mean():.2f}$)",
            zorder=3 if is_20um else 5
        )

    ax.axhline(0.0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=1.0, alpha=0.7, label='Ideal Perfect Coupling ($y=1$)')
    ax.set_xlabel(r'Opposite-Flow Fraction $f_{\mathrm{opp}} = \mathrm{fraction}(\mathbf{u}\cdot\langle\mathbf{u}\rangle < 0)$ (Counter-Lane Occupation)', fontsize=13, labelpad=8)
    ax.set_ylabel(r'Dynamical Transfer Efficiency $\eta = \frac{\sum v_\parallel}{\sum |\langle\mathbf{u}\rangle_{R_c}|}$', fontsize=13, labelpad=8)
    ax.set_title('Decoupling Induced by Counter-Flow Lanes:\n'
                 r'Transfer Efficiency $\eta$ Conditioned on Opposite-Flow Fraction $f_{\mathrm{opp}}$',
                 fontsize=14, fontweight='bold', pad=12)
    ax.set_xlim(-0.02, 0.45)
    ax.set_ylim(-0.15, 0.70)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='upper right', frameon=True, framealpha=0.92, fontsize=10.0)

    ax.text(
        0.04, 0.05,
        'Key Physics:\n'
        r'$\bullet$ $f_{\mathrm{opp}} = 0$: Pure forward stream $\rightarrow$ High $\eta$' + '\n'
        r'$\bullet$ $f_{\mathrm{opp}} \geq 0.25$: Counter-running lanes invade' + '\n'
        r'  $\rightarrow$ $\eta$ collapses across all sizes ($0.63 \sim 7\,\mu\mathrm{m}$)',
        transform=ax.transAxes, ha='left', va='bottom',
        fontsize=9.5, bbox=dict(boxstyle='round,pad=0.5', fc='#F8FAFC', ec='#475569', alpha=0.95, lw=1.2)
    )

    plt.tight_layout()
    save_fig(fig, "contrast_fig11_eta_vs_opposite_lane_fraction")
    plt.close(fig)

    # =========================================================================
    # Figure 10ab: 2-Panel Master Edition (Phi_t and f_opp side-by-side)
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 5.8), sharey=True)

    # Panel A: Phi_t
    for b_item in vcd.BEADS_INFO:
        bname = b_item['name']
        if bname not in phi_results or len(phi_results[bname]) == 0:
            continue
        res = phi_results[bname]
        is_20um = (b_item['diameter_um'] >= 15.0)
        fmt = f"{b_item['marker']}-"
        mfc = 'white' if is_20um else b_item['color']
        ax1.errorbar(
            res['bin_center'], res['eta'],
            yerr=[res['eta'] - res['ci_low'], res['ci_high'] - res['eta']],
            fmt=fmt, color=b_item['color'], ecolor=b_item['color'],
            mfc=mfc, mec=b_item['color'], mew=2.0 if is_20um else 1.0,
            markersize=7.5, linewidth=1.8, capsize=3.5, capthick=1.0,
            label=f"{b_item['label']}",
            zorder=3 if is_20um else 5
        )

    ax1.axhline(0.0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax1.set_xlabel(r'Local Footprint Polar Order $\Phi_t = \|\langle\hat{\mathbf{u}}\rangle_{R_c}\|$', fontsize=12)
    ax1.set_ylabel(r'Dynamical Transfer Efficiency $\eta$', fontsize=12)
    ax1.set_title(r'(a) Conditioned on Local Polar Order $\Phi_t$', fontsize=13, fontweight='bold')
    ax1.set_xlim(0.30, 1.02)
    ax1.set_ylim(-0.15, 0.70)
    ax1.grid(True, linestyle='--', alpha=0.35)
    ax1.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=9.5)

    # Panel B: f_opp
    for b_item in vcd.BEADS_INFO:
        bname = b_item['name']
        if bname not in fopp_results or len(fopp_results[bname]) == 0:
            continue
        res = fopp_results[bname]
        is_20um = (b_item['diameter_um'] >= 15.0)
        fmt = f"{b_item['marker']}-"
        mfc = 'white' if is_20um else b_item['color']
        ax2.errorbar(
            res['bin_center'], res['eta'],
            yerr=[res['eta'] - res['ci_low'], res['ci_high'] - res['eta']],
            fmt=fmt, color=b_item['color'], ecolor=b_item['color'],
            mfc=mfc, mec=b_item['color'], mew=2.0 if is_20um else 1.0,
            markersize=7.5, linewidth=1.8, capsize=3.5, capthick=1.0,
            label=f"{b_item['label']}",
            zorder=3 if is_20um else 5
        )

    ax2.axhline(0.0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax2.set_xlabel(r'Opposite-Flow Fraction $f_{\mathrm{opp}}$ (Counter-Lane Occupation)', fontsize=12)
    ax2.set_title(r'(b) Conditioned on Opposite-Flow Fraction $f_{\mathrm{opp}}$', fontsize=13, fontweight='bold')
    ax2.set_xlim(-0.02, 0.45)
    ax2.grid(True, linestyle='--', alpha=0.35)
    ax2.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=9.5)

    plt.tight_layout()
    save_fig(fig, "contrast_fig10ab_eta_conditioned_on_heterogeneity_2panels")
    plt.close(fig)

    # Save summary tables to CSV
    summary_rows = []
    for bname in phi_results:
        for _, r in phi_results[bname].iterrows():
            summary_rows.append({
                'bead_name': bname,
                'metric': 'phi_t',
                **r.to_dict()
            })
    for bname in fopp_results:
        for _, r in fopp_results[bname].iterrows():
            summary_rows.append({
                'bead_name': bname,
                'metric': 'f_opp',
                **r.to_dict()
            })
    df_sum = pd.DataFrame(summary_rows)
    for out_dir in OUT_DIRS:
        df_sum.to_csv(out_dir / "eta_conditioned_on_heterogeneity_summary.csv", index=False)
    print("All tasks completed successfully!")


if __name__ == '__main__':
    main()
