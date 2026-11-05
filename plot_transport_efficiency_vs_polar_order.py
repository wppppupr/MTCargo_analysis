#!/usr/bin/env python3
"""
plot_transport_efficiency_vs_polar_order.py
===========================================

輸送効率 eta = (実測貨物速度 / 足印の流れ) = (v_parallel / |<u_fp>|) vs (1 - Phi) の解析・描画スクリプト。

【主要な仕様】
- 縦軸: 局所輸送効率 eta = v_parallel / |<u_fp>| = (v_parallel / u_inf) / (|<u_fp>| / u_inf)。
        完全追従（Ideal Tracer）なら eta = 1.0、完全停止・スタックなら eta = 0.0。
- 横軸: 1 - Phi（対数軸、左が「完全に極性的 Phi ~ 1」、右が「打ち消し合い Phi ~ 0」）。
        各ビンの点は、ビン内の 1 - Phi の平均位置に配置。
- ビン: 全サイズ共通の分位ビン（10分割）をベースにし、サンプル数が少ないビン（< 50点）は
        隣接するビンと統合（マージ）して十分な統計量を確保。
- 誤差棒: 粒子単位のクラスター・ブートストラップ（Cluster Bootstrap over Particles, N=2000）により、
  比の統計量 eta の標準誤差を頑健に算出。
- 基準線:
  - eta = 1.0 (Tracer Limit: 流体完全追従)
  - eta = 0.0 (Stall Boundary: 完全停止)
- 出力図:
  1. ビーズ径別 6パネル個別図 (eta_vs_one_minus_phi_6panels.png / .svg)
  2. 全ビーズ重ね描き図 (eta_vs_one_minus_phi_overlay.png / .svg)
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = Path("/mnt/NAS-Ebanaru/sasaki/MTsingleBeads")
if not ROOT_DIR.exists():
    ROOT_DIR = CURRENT_DIR

OUT_DIRS = [
    CURRENT_DIR / "figure" / "polar_vs_normalized_velocity",
    ROOT_DIR / "figure" / "polar_vs_normalized_velocity"
]

# スタイルの適用
style_path = CURRENT_DIR / 'libs' / 'my_style.mplstyle'
if style_path.exists():
    try:
        plt.style.use(str(style_path))
        style_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    except Exception:
        style_colors = ['#882255', '#CC6677', '#DDCC77', '#999933', '#117733', '#44AA99']
else:
    style_colors = ['#882255', '#CC6677', '#DDCC77', '#999933', '#117733', '#44AA99']

BEADS_INFO = [
    {"name": "beads06um", "diameter_um": 0.63, "radius_um": 0.315, "marker": "^", "color": style_colors[0], "label": r"$0.63\,\mu\mathrm{m}$"},
    {"name": "beads1um",  "diameter_um": 1.18, "radius_um": 0.590, "marker": "o", "color": style_colors[1], "label": r"$1.18\,\mu\mathrm{m}$"},
    {"name": "beads3um",  "diameter_um": 3.37, "radius_um": 1.685, "marker": "d", "color": style_colors[2], "label": r"$3.37\,\mu\mathrm{m}$"},
    {"name": "beads5um",  "diameter_um": 5.00, "radius_um": 2.500, "marker": "p", "color": style_colors[3], "label": r"$5.00\,\mu\mathrm{m}$"},
    {"name": "beads7um",  "diameter_um": 7.24, "radius_um": 3.620, "marker": "h", "color": style_colors[4], "label": r"$7.24\,\mu\mathrm{m}$"},
    {"name": "beads20um", "diameter_um": 20.0, "radius_um": 10.00, "marker": "s", "color": style_colors[5], "label": r"$20.0\,\mu\mathrm{m}$"},
]

def load_data() -> pd.DataFrame:
    """
    足元アニュラス（貨物真下マスク）の局所ポーラーオーダーと無次元化速度データをロード・結合する。
    """
    cache_fp = CURRENT_DIR / 'scratch' / 'footprint_heterogeneity_cache.pkl'
    cache_ext = CURRENT_DIR / 'scratch' / 'cargo_flow_profile_extended_cache.pkl'

    if not cache_fp.exists():
        raise FileNotFoundError(f"Missing required cache: {cache_fp}")
    if not cache_ext.exists():
        raise FileNotFoundError(f"Missing required cache: {cache_ext}")

    df_fp = pd.read_pickle(cache_fp)
    df_ext = pd.read_pickle(cache_ext)

    ext_map = {Path(p).name: float(u) for p, u in zip(df_ext['exp_dir'], df_ext['u_inf'])}
    df_fp['u_inf'] = df_fp['exp_dir'].map(ext_map)
    df_fp['vpar_norm'] = df_fp['v_parallel'] / df_fp['u_inf']
    df_fp['vvirt_norm'] = df_fp['u_fp'] / df_fp['u_inf']
    
    # 1 - Phi（対数軸用、下限 1e-5 クリップ）
    df_fp['one_minus_phi'] = np.clip(1.0 - df_fp['phi_t'], 1e-5, 1.0)

    return df_fp

def save_fig_all(fig: plt.Figure, base_name: str):
    """PNG & SVG で保存"""
    for out_dir in OUT_DIRS:
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_dir / f"{base_name}.png", dpi=300, bbox_inches='tight')
            fig.savefig(out_dir / f"{base_name}.svg", bbox_inches='tight')
            print(f"Saved: {out_dir / base_name}.png / .svg")
        except Exception as e:
            print(f"Warning: Failed to save {base_name} to {out_dir}: {e}")

def particle_cluster_bootstrap_eta(sub_bin: pd.DataFrame, n_boot: int = 2000, seed: int = 42) -> Tuple[float, float, float, int]:
    """
    粒子単位のクラスター・ブートストラップによる eta = v_parallel / |<u_fp>| の平均と標準誤差の算出。
    """
    unique_pids = sub_bin['unique_pid'].unique()
    n_p = len(unique_pids)
    if n_p == 0:
        return np.nan, np.nan, np.nan, 0

    part_means = sub_bin.groupby('unique_pid')[['vpar_norm', 'vvirt_norm', 'one_minus_phi']].mean()
    part_weights = sub_bin.groupby('unique_pid').size().values

    vpar_vals = part_means['vpar_norm'].values
    vvirt_vals = part_means['vvirt_norm'].values
    x_vals = part_means['one_minus_phi'].values

    orig_vpar = float(np.average(vpar_vals, weights=part_weights))
    orig_vvirt = float(np.average(vvirt_vals, weights=part_weights))
    orig_x = float(np.average(x_vals, weights=part_weights))
    orig_eta = orig_vpar / orig_vvirt if orig_vvirt > 0 else np.nan

    if n_p == 1:
        return orig_x, orig_eta, 0.0, 1

    rng = np.random.default_rng(seed)
    boot_eta = []
    for _ in range(n_boot):
        idx = rng.choice(n_p, size=n_p, replace=True)
        w_sample = part_weights[idx]
        w_sum = np.sum(w_sample)
        if w_sum == 0:
            continue
        v_p = np.average(vpar_vals[idx], weights=w_sample)
        v_v = np.average(vvirt_vals[idx], weights=w_sample)
        if v_v > 0:
            boot_eta.append(v_p / v_v)

    eta_err = float(np.std(boot_eta, ddof=1)) if len(boot_eta) > 10 else 0.0

    return orig_x, orig_eta, eta_err, n_p

def compute_eta_binned_stats(df: pd.DataFrame, n_bins: int = 10, min_points: int = 50) -> pd.DataFrame:
    """
    全サイズ共通の分位ビン（10分割）を基準とし、データ数が min_points 未満のビンは
    隣接ビンと統合して eta の粒子ブートストラップ統計量を算出する。
    """
    df_work = df.copy()
    _, bin_edges = pd.qcut(df_work['one_minus_phi'], q=n_bins, retbins=True)
    n_orig_bins = len(bin_edges) - 1

    binned_rows = []

    for item in BEADS_INFO:
        bname = item['name']
        sub = df_work[df_work['bead_name'] == bname].copy()
        if sub.empty:
            continue

        b_indices = np.digitize(sub['one_minus_phi'].values, bin_edges) - 1
        b_indices = np.clip(b_indices, 0, n_orig_bins - 1)
        counts = [int(np.sum(b_indices == i)) for i in range(n_orig_bins)]

        # ビン統合グループの決定（左から走査し、min_points未満は統合）
        merged_groups = []
        curr_group = []
        curr_count = 0
        for i in range(n_orig_bins):
            curr_group.append(i)
            curr_count += counts[i]
            if curr_count >= min_points or i == n_orig_bins - 1:
                merged_groups.append(curr_group)
                curr_group = []
                curr_count = 0

        # 末尾のグループが min_points 未満で残った場合は直前と統合
        if len(merged_groups) > 1:
            last_cnt = sum(counts[idx] for idx in merged_groups[-1])
            if last_cnt < min_points:
                last_g = merged_groups.pop()
                merged_groups[-1].extend(last_g)

        # 各統合ビンについて eta 統計量を計算
        for g_idx, g_bins in enumerate(merged_groups):
            mask = np.isin(b_indices, g_bins)
            sub_bin = sub[mask]
            n_pts = len(sub_bin)
            if n_pts == 0:
                continue

            orig_x, eta_m, eta_e, n_p = particle_cluster_bootstrap_eta(sub_bin)

            b_left = bin_edges[g_bins[0]]
            b_right = bin_edges[g_bins[-1] + 1]

            binned_rows.append({
                'bead_name': bname,
                'diameter_um': item['diameter_um'],
                'label': item['label'],
                'group_idx': g_idx,
                'merged_bins': str(g_bins),
                'bin_range': f"[{b_left:.5f}, {b_right:.5f}]",
                'n_points': n_pts,
                'n_particles': n_p,
                'x_mean_one_minus_phi': orig_x,
                'phi_mean': float(sub_bin['phi_t'].mean()),
                'eta_mean': eta_m,
                'eta_sem': eta_e,
                'vpar_mean': float(sub_bin['vpar_norm'].mean()),
                'vvirt_mean': float(sub_bin['vvirt_norm'].mean()),
            })

    return pd.DataFrame(binned_rows)

def plot_eta_6panels(df_binned: pd.DataFrame):
    """
    ビーズ径別 6パネル個別図:
    縦軸: eta = v_parallel / |<u_fp>|
    横軸: 1 - Phi (対数軸)
    """
    fig, axes = plt.subplots(2, 3, figsize=(15.6, 9.8), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for idx, item in enumerate(BEADS_INFO):
        ax = axes_flat[idx]
        bname = item['name']
        color = item['color']
        marker = item['marker']
        lbl = item['label']
        d_um = item['diameter_um']

        sub = df_binned[df_binned['bead_name'] == bname].sort_values('x_mean_one_minus_phi')

        # 基準線: eta = 1.0 (Tracer Limit: 完全追従), eta = 0.0 (Stall: 停止)
        ax.axhline(1.0, color='#222222', linestyle='--', linewidth=1.5, alpha=0.8, zorder=1)
        ax.axhline(0.0, color='#888888', linestyle='-', linewidth=0.9, alpha=0.8, zorder=1)

        if sub.empty:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(f"{lbl}", fontsize=15, fontweight='bold')
            continue

        # 実測輸送効率 eta の曲線
        ax.errorbar(
            sub['x_mean_one_minus_phi'], sub['eta_mean'], yerr=sub['eta_sem'],
            color=color, linestyle='-', linewidth=2.6,
            marker=marker, markersize=8.5, markerfacecolor=color, markeredgecolor='#111111',
            markeredgewidth=1.2, capsize=4.0, capthick=1.5,
            label=f'Transport Efficiency $\\eta$', zorder=5
        )

        ax.set_xscale('log')
        ax.set_xlim(5e-4, 1.2)
        ax.set_ylim(-0.25, 1.15)

        total_pts = sub['n_points'].sum()
        max_p = sub['n_particles'].max()
        ax.text(
            0.05, 0.94,
            f"{lbl} ($2R_c = {d_um:.2f}\\,\\mu\\mathrm{{m}}$)\n$N = {total_pts:,}$ (particles: {max_p})",
            transform=ax.transAxes,
            fontsize=12.0,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white', alpha=0.90, edgecolor='#cccccc')
        )

        ax.set_title(f"{lbl} ($2R_c = {d_um:.2f}\\,\\mu\\mathrm{{m}}$)", fontsize=15, fontweight='bold')

        if idx >= 3:
            ax.set_xlabel(r"$1 - \Phi$ (Aligned $\leftarrow$ Polar Order $\rightarrow$ Anti-aligned)", fontsize=13)
        if idx % 3 == 0:
            ax.set_ylabel(r"Transport Efficiency $\eta = v_\parallel / |\langle\mathbf{u}\rangle_{\mathrm{fp}}|$", fontsize=13.5)

        if idx == 0:
            ax.text(0.05, 0.72, r"Tracer Limit ($\eta = 1.0$)", transform=ax.transAxes, fontsize=11.0, color='#333333', fontweight='bold')

    fig.suptitle(
        r"Transport Efficiency $\eta = v_\parallel / |\langle\mathbf{u}\rangle_{\mathrm{fp}}|$ across Microtubule Polar Order $1 - \Phi$",
        fontsize=16.5, fontweight='bold', y=0.995
    )
    fig.tight_layout()
    save_fig_all(fig, "eta_vs_one_minus_phi_6panels")
    plt.close(fig)

def plot_eta_overlay(df_binned: pd.DataFrame):
    """
    全ビーズ重ね描き図:
    各サイズの輸送効率曲線 eta(1 - Phi) を同一軸上に比較描画。
    """
    fig, ax = plt.subplots(figsize=(10.0, 7.2))

    # 基準線
    ax.axhline(1.0, color='#222222', linestyle='--', linewidth=1.5, alpha=0.85, zorder=1)
    ax.axhline(0.0, color='#888888', linestyle='-', linewidth=0.9, alpha=0.8, zorder=1)
    ax.text(8e-4, 1.03, r"Tracer Limit ($\eta = 1.0$)", fontsize=12.0, color='#222222', fontweight='bold')
    ax.text(8e-4, -0.06, r"Stall Boundary ($\eta = 0.0$)", fontsize=11.0, color='#666666')

    for item in BEADS_INFO:
        bname = item['name']
        color = item['color']
        marker = item['marker']
        lbl = item['label']

        sub = df_binned[df_binned['bead_name'] == bname].sort_values('x_mean_one_minus_phi')
        if sub.empty:
            continue

        total_pts = sub['n_points'].sum()

        ax.errorbar(
            sub['x_mean_one_minus_phi'], sub['eta_mean'], yerr=sub['eta_sem'],
            color=color, linestyle='-', linewidth=2.4,
            marker=marker, markersize=8.5, markerfacecolor=color, markeredgecolor='#111111',
            markeredgewidth=1.1, capsize=4.0, capthick=1.3,
            label=f"{lbl} ($N={total_pts:,}$)", zorder=4
        )

    ax.set_xscale('log')
    ax.set_xlim(5e-4, 1.2)
    ax.set_ylim(-0.25, 1.15)

    ax.set_xlabel(r"$1 - \Phi$ (Highly Polar $\leftarrow$ Microtubule Order $\rightarrow$ Opposing / Cancelled)", fontsize=15)
    ax.set_ylabel(r"Transport Efficiency $\eta = v_\parallel / |\langle\mathbf{u}\rangle_{\mathrm{fp}}|$", fontsize=15)
    ax.set_title(r"Cargo Transport Efficiency $\eta = v_\parallel / |\langle\mathbf{u}\rangle_{\mathrm{fp}}|$ vs $1 - \Phi$", fontsize=16, fontweight='bold', pad=12)

    ax.legend(
        fontsize=11.0,
        loc='upper right',
        framealpha=0.92,
        borderpad=0.45,
        labelspacing=0.35,
        handletextpad=0.4
    )

    fig.tight_layout()
    save_fig_all(fig, "eta_vs_one_minus_phi_overlay")
    plt.close(fig)

def main():
    print("Loading footprint heterogeneity data with masked cargo region...")
    df = load_data()
    print(f"Loaded {len(df):,} instantaneous observations.")

    print("\nComputing common quantile bins with adaptive merging (min 50 points per bin)...")
    print("Evaluating particle-level cluster bootstrap for eta = v_parallel / |<u_fp>| (N=2000)...")
    df_binned = compute_eta_binned_stats(df, n_bins=10, min_points=50)

    # ビン統計テーブルの保存
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        df_binned.to_csv(out_dir / "eta_vs_one_minus_phi_binned.csv", index=False)
    print(f"Saved binned CSV to {OUT_DIRS[0] / 'eta_vs_one_minus_phi_binned.csv'}")

    print("\nGenerating eta figures...")
    plot_eta_6panels(df_binned)
    plot_eta_overlay(df_binned)
    print("\n[Complete] All eta plots generated successfully.")

if __name__ == '__main__':
    main()
