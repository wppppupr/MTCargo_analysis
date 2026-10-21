#!/usr/bin/env python3
"""
plot_okubo_weiss_analysis.py

微小管の流れ場から Okubo-Weiss パラメータ Q_OW を計算し、
以下の2つの主要プロットおよびサマリー解析を出力するスクリプト。

1. ラグランジュ的サンプリング分布の偏り（滞在バイアス）:
   背景流場全体における Q_OW の存在確率分布 P_Euler(Q) に対し、
   貨物が実際に通過した位置での Q_OW のサンプリング分布 P_cargo(Q) を比較するヒストグラム/密度分布。
   Q_cargo(t) ≡ Q_OW(r_c(t), t)

2. 軌跡と Q_OW 重ね合わせ図の設計:
   Q_OW を標準偏差 sigma_Q で規格化し、発散型カラーマップ (RdBu_r または coolwarm) を適用。
   - Q_OW < 0 (渦コア・回転支配) : 青 (Blue)
   - Q_OW ≈ 0 (中立境界) : 白または薄いグレー
   - Q_OW > 0 (流線レーン・剪断支配) : 赤 (Red)
   表示範囲: [-3 sigma_Q, +3 sigma_Q] に対称クリップ。
   背景にある時刻 t_0 の Q_OW スナップショットを表示し、その上に粒子のトラッキング軌道を描画。

さらに、全ビーズ径 (0.63, 1.18, 3.37, 5.00, 7.24, 20.00 um) の比較解析もサポート。
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from scipy import stats

# 自作モジュール
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from libs.okubo_weiss import (
    compute_okubo_weiss,
    sample_field_at_coordinates,
    analyze_experiment_okubo_weiss
)

# スタイル設定
style_path = CURRENT_DIR / 'libs' / 'my_style.mplstyle'
if style_path.exists():
    plt.style.use(str(style_path))
    style_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
else:
    style_colors = ['#882255', '#CC6677', '#DDCC77', '#117733', '#44AA99', '#332288']

POSSIBLE_ROOTS = [
    Path('/Volumes/data-1/Sasaki/MTsingleBeads'),
    Path('/Volumes/data-1/sasaki/MTsingleBeads'),
    Path('/Volumes/data/Sasaki/MTsingleBeads'),
    Path('/Volumes/data/sasaki/MTsingleBeads'),
    Path('/mnt/NAS-Ebanaru/sasaki/MTsingleBeads'),
    Path('/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads'),
]

BEADS_INFO = [
    {"name": "beads06um", "diameter_um": 0.63, "marker": "^", "color": style_colors[0]},
    {"name": "beads1um",  "diameter_um": 1.18, "marker": "o", "color": style_colors[1]},
    {"name": "beads3um",  "diameter_um": 3.37, "marker": "d", "color": style_colors[2]},
    {"name": "beads5um",  "diameter_um": 5.00, "marker": "p", "color": style_colors[3]},
    {"name": "beads7um",  "diameter_um": 7.24, "marker": "h", "color": style_colors[4]},
    {"name": "beads20um", "diameter_um": 20.0, "marker": "s", "color": style_colors[5]},
]


def resolve_dataset_path(path_in: Union[str, Path]) -> Optional[Path]:
    """大文字小文字の違い (sasaki / Sasaki など) を解決して存在する Path を返す"""
    p = Path(path_in)
    if p.exists():
        return p
    # sasaki <-> Sasaki の置き換えを試行
    p_str = str(p)
    candidates = [
        p_str.replace('/sasaki/', '/Sasaki/'),
        p_str.replace('/Sasaki/', '/sasaki/'),
        p_str.replace('/Volumes/data/', '/Volumes/data-1/'),
        p_str.replace('/Volumes/data-1/', '/Volumes/data/'),
    ]
    for c in candidates:
        cp = Path(c)
        if cp.exists():
            return cp
    return None


def find_default_dataset(root: Optional[Path] = None, bead_name: str = 'beads3um') -> Optional[Path]:
    """デフォルトの解析対象実験ディレクトリを探索する"""
    roots_to_try = [root] if root else POSSIBLE_ROOTS
    for r in roots_to_try:
        if r is not None and r.exists():
            b_dir = r / bead_name
            if b_dir.exists():
                for d1 in b_dir.iterdir():
                    if d1.is_dir() and not d1.name.startswith('.'):
                        for d2 in d1.iterdir():
                            if d2.is_dir() and not d2.name.startswith('.'):
                                flow = d2 / 'GFP_flows.h5'
                                tracks = d2 / 'beads_tracks.csv'
                                if flow.exists() and tracks.exists():
                                    return d2
    return None


def plot_lagrangian_sampling_bias(
    res: Dict,
    output_path: Path,
    title_suffix: str = "",
    xlim: Tuple[float, float] = (-4.0, 4.0),
    show_stats_box: bool = True
) -> None:
    """
    プロット1: ラグランジュ的サンプリング分布の偏り（滞在バイアス）
    背景流場全体 P_Euler(Q) と貨物軌道上 P_cargo(Q) のヒストグラム/確率密度分布を比較。
    主プロット（線形軸）＋インセット（対数軸）による最高品質の論文仕様。
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    bin_centers, p_euler = res['hist_euler']
    _, p_cargo = res['hist_cargo']
    e_stats = res['euler_stats']
    c_stats = res['cargo_stats']
    sigma_Q = res['sigma_Q']

    fig, ax = plt.subplots(figsize=(10, 7))

    # 1. 領域シェーディング (Q < 0: 青, Q > 0: 赤)
    ax.axvspan(xlim[0], 0, color='#3B82F6', alpha=0.06, zorder=1)
    ax.axvspan(0, xlim[1], color='#EF4444', alpha=0.06, zorder=1)

    # 2. 中立境界線 Q = 0
    ax.axvline(0, color='#4B5563', linestyle='--', linewidth=1.8, alpha=0.85, zorder=2, label=r'Neutral boundary ($Q=0$)')

    # 3. 背景オイラー分布 P_Euler(Q)
    widths = np.diff(res['norm_bins'])
    ax.bar(
        bin_centers,
        p_euler,
        width=widths,
        color='#9CA3AF',
        alpha=0.45,
        edgecolor='#4B5563',
        linewidth=1.2,
        label=r'Eulerian Background $P_{\mathrm{Euler}}(Q)$',
        zorder=3
    )

    # 4. 貨物ラグランジュ分布 P_cargo(Q)
    ax.step(
        np.append(res['norm_bins'][0], bin_centers),
        np.append(p_cargo[0], p_cargo),
        where='mid',
        color='#DC2626',
        linewidth=3.0,
        label=r'Cargo Lagrangian $P_{\mathrm{cargo}}(Q)$',
        zorder=4
    )
    ax.fill_between(
        bin_centers,
        0,
        p_cargo,
        step='mid',
        color='#DC2626',
        alpha=0.25,
        zorder=3
    )

    # 平均値の垂直ライン
    ax.axvline(
        e_stats['mean_norm'],
        color='#4B5563',
        linestyle=':',
        linewidth=2.0,
        alpha=0.9,
        label=rf'$\langle Q \rangle_{{\mathrm{{Euler}}}} = {e_stats["mean_norm"]:+.2f}\sigma_Q$'
    )
    ax.axvline(
        c_stats['mean_norm'],
        color='#DC2626',
        linestyle='-.',
        linewidth=2.2,
        alpha=0.95,
        label=rf'$\langle Q \rangle_{{\mathrm{{cargo}}}} = {c_stats["mean_norm"]:+.2f}\sigma_Q$'
    )

    # ピーク高さから y 軸上限を設定 (余白を広く確保)
    mask_view = (bin_centers >= xlim[0]) & (bin_centers <= xlim[1])
    peak_y = max(np.max(p_euler[mask_view]), np.max(p_cargo[mask_view]))
    y_max = peak_y * 1.45

    # 領域ラベルの追加 (グラフ上部)
    ax.text(
        -0.5 * abs(xlim[0]),
        y_max * 0.94,
        'Rotation Dominated\n(Vortex Cores: $Q < 0$)',
        ha='center',
        va='top',
        fontsize=12,
        fontweight='bold',
        color='#1D4ED8',
        bbox=dict(boxstyle='round,pad=0.35', facecolor='#EFF6FF', edgecolor='#93C5FD', alpha=0.95, lw=1)
    )
    ax.text(
        0.5 * abs(xlim[1]),
        y_max * 0.94,
        'Strain Dominated\n(Shear Lanes: $Q > 0$)',
        ha='center',
        va='top',
        fontsize=12,
        fontweight='bold',
        color='#B91C1C',
        bbox=dict(boxstyle='round,pad=0.35', facecolor='#FEF2F2', edgecolor='#FCA5A5', alpha=0.95, lw=1)
    )

    # 軸・ラベル設定
    ax.set_xlabel(r'Normalized Okubo-Weiss Parameter $Q_{\mathrm{OW}} / \sigma_Q$', fontsize=16)
    ax.set_ylabel(r'Probability Density $P(Q / \sigma_Q)$', fontsize=16)
    ax.set_xlim(xlim)
    ax.set_ylim(0, y_max)
    ax.tick_params(axis='both', which='major', labelsize=14)

    # 統計ボックス (左上)
    if show_stats_box:
        ks_p = c_stats['ks_pvalue']
        ks_str = f"{ks_p:.2e}" if ks_p < 0.001 else f"{ks_p:.3f}"
        stats_text = (
            r"$\bf{Lagrangian\ Residence\ Bias:}$" + "\n"
            rf"$\sigma_Q = {sigma_Q:.3e}\ \mathrm{{px}}^{{-2}}$" + "\n"
            rf"Euler:  Vortex {e_stats['vortex_fraction']*100:.1f}%, Strain {e_stats['strain_fraction']*100:.1f}%" + "\n"
            rf"Cargo: Vortex {c_stats['vortex_fraction']*100:.1f}%, Strain {c_stats['strain_fraction']*100:.1f}%" + "\n"
            rf"$\bf{{\Delta f_{{\mathrm{{strain}}}} = {(c_stats['strain_fraction'] - e_stats['strain_fraction'])*100:+.1f}\%}}$ (Shear Lane Bias)" + "\n"
            rf"KS-test: $D = {c_stats['ks_statistic']:.3f},\ p = {ks_str}$"
        )
        ax.text(
            0.03,
            0.82,
            stats_text,
            transform=ax.transAxes,
            fontsize=11.5,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFFFFF', edgecolor='#CBD5E1', alpha=0.95, lw=1.2)
        )

    # 凡例 (右上)
    ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.85), fontsize=11, framealpha=0.95, edgecolor='#CBD5E1')

    # インセット: 対数軸での裾野 (Tails) の比較 (右下寄りに配置)
    ax_ins = inset_axes(ax, width="32%", height="28%", loc='center right', bbox_to_anchor=(0.0, -0.16, 0.98, 1.0), bbox_transform=ax.transAxes)
    ax_ins.set_yscale('log')
    ax_ins.axvline(0, color='#4B5563', linestyle='--', linewidth=1.0, alpha=0.7)
    ax_ins.step(bin_centers, p_euler, color='#4B5563', linewidth=1.5, where='mid', label='Euler')
    ax_ins.step(bin_centers, p_cargo, color='#DC2626', linewidth=2.0, where='mid', label='Cargo')
    ax_ins.set_xlim(-5, 5)
    ax_ins.set_ylim(1e-4, max(peak_y * 1.5, 5))
    ax_ins.set_title('Log-scale Tails', fontsize=10, fontweight='bold', pad=4)
    ax_ins.tick_params(axis='both', which='major', labelsize=9)
    ax_ins.grid(True, linestyle=':', alpha=0.6)

    title_str = "Lagrangian Sampling Bias of Okubo-Weiss Parameter"
    if title_suffix:
        title_str += f" ({title_suffix})"
    ax.set_title(title_str, fontsize=16, fontweight='bold', pad=14)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    fig.savefig(output_path.with_suffix('.svg'))
    fig.savefig(output_path.with_suffix('.pdf'))
    plt.close(fig)
    print(f"Saved Lagrangian sampling bias plot to:\n  - {output_path}\n  - {output_path.with_suffix('.svg')}")


def plot_trajectory_q_ow_overlay(
    res: Dict,
    output_path: Path,
    title_suffix: str = "",
    clip_sigmas: float = 3.0,
    cmap_name: str = 'RdBu_r',
    smooth_sigma: float = 1.2
) -> None:
    """
    プロット2: 軌跡と Q_OW 重ね合わせ図の設計
    - 背景: ある時刻 t_0 の Q_OW スナップショット (正規化 Q / sigma_Q)
    - カラーマップ: Diverging Colormap (RdBu_r または coolwarm)
      Q < 0 (回転支配): 青 (Blue)
      Q ≈ 0 (中立境界): 白または薄いグレー
      Q > 0 (剪断支配): 赤 (Red)
    - 表示範囲: [-clip_sigmas, +clip_sigmas] ([-3 sigma_Q, +3 sigma_Q]) に対称クリップ
    - トラッキング軌道のオーバーレイ
    """
    from scipy.ndimage import gaussian_filter

    snapshot = res['snapshot']
    if snapshot is None:
        print("Warning: No snapshot available in results.")
        return

    df_cargo = res['cargo_df']
    scale = res['scale']  # um/pixel
    frame_idx = snapshot['frame']
    Q_norm = snapshot['Q_norm']
    H, W = Q_norm.shape

    # 光学顕微鏡の物理分解能（PSF ≈ 2.3 px）に合わせた微小平滑化で微分ノイズを除去
    if smooth_sigma > 0:
        Q_norm_plot = gaussian_filter(Q_norm.astype(np.float32), sigma=smooth_sigma)
    else:
        Q_norm_plot = Q_norm

    # 1枚の図に「全体図 (Full Field)」と「貨物軌跡ズーム (Zoom-in)」の2パネル構成 (1行2列)
    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(18, 9.2), gridspec_kw={'width_ratios': [1.15, 1.0]})

    # カラー正規化: [-clip_sigmas, +clip_sigmas] に対称クリップ
    norm = mcolors.Normalize(vmin=-clip_sigmas, vmax=clip_sigmas)

    # 物理座標への変換 (um)
    extent_um = [0, W * scale, H * scale, 0]  # [left, right, bottom, top]

    # --- 左パネル: 全体視野 (Full FOV) ---
    im_full = ax_full.imshow(
        Q_norm_plot,
        cmap=cmap_name,
        norm=norm,
        extent=extent_um,
        interpolation='bicubic'
    )

    # 軌跡の描画 (全粒子)
    unique_particles = df_cargo['particle'].unique()
    traj_colors = plt.cm.spring(np.linspace(0.1, 0.9, max(len(unique_particles), 1)))

    for i, p in enumerate(unique_particles):
        p_df = df_cargo[df_cargo['particle'] == p].sort_values('frame')
        px_um = p_df['x'].values * scale
        py_um = p_df['y'].values * scale

        # 軌道の軌跡線 (視認性を高める黒の外枠)
        ax_full.plot(px_um, py_um, color='black', linewidth=3.6, alpha=0.9, zorder=5)
        ax_full.plot(px_um, py_um, color=traj_colors[i], linewidth=2.2, alpha=1.0, zorder=6, label=f'Particle {p}' if len(unique_particles) <= 6 else None)

        # 始点と終点
        ax_full.scatter(px_um[0], py_um[0], color='#10B981', edgecolor='black', s=80, marker='o', zorder=7)
        ax_full.scatter(px_um[-1], py_um[-1], color='#F59E0B', edgecolor='black', s=110, marker='s', zorder=7)

        # 現在フレーム位置
        curr_row = p_df[p_df['frame'] == frame_idx]
        if not curr_row.empty:
            cx_um = curr_row['x'].values[0] * scale
            cy_um = curr_row['y'].values[0] * scale
            ax_full.scatter(cx_um, cy_um, color='#FFFFFF', edgecolor='#DC2626', s=140, linewidth=2.5, marker='*', zorder=8)

    # スケールバー (100 um)
    scalebar_len_um = 100
    scalebar = AnchoredSizeBar(
        ax_full.transData,
        scalebar_len_um,
        f'{scalebar_len_um} μm',
        'lower left',
        pad=0.6,
        color='black',
        frameon=True,
        size_vertical=3.0,
        fontproperties=fm.FontProperties(size=12, weight='bold')
    )
    scalebar.patch.set_boxstyle("round,pad=0.3")
    scalebar.patch.set_facecolor("white")
    scalebar.patch.set_edgecolor("#9CA3AF")
    scalebar.patch.set_alpha(0.9)
    ax_full.add_artist(scalebar)

    ax_full.set_title(f"Full Field Snapshot ($t_0 = {frame_idx * res['frame_interval']:.0f}$ s, Frame {frame_idx})", fontsize=15, fontweight='bold', pad=10)
    ax_full.set_xlabel(r'$x\ [\mu\mathrm{m}]$', fontsize=14, labelpad=8)
    ax_full.set_ylabel(r'$y\ [\mu\mathrm{m}]$', fontsize=14)
    ax_full.tick_params(axis='both', which='major', labelsize=12)

    # --- 右パネル: 代表軌跡の局所ズーム (Zoom-in) ---
    best_particle = unique_particles[0]
    max_span = 0
    for p in unique_particles:
        p_df = df_cargo[df_cargo['particle'] == p]
        span = np.hypot(p_df['x'].max() - p_df['x'].min(), p_df['y'].max() - p_df['y'].min())
        if span > max_span:
            max_span = span
            best_particle = p

    p_best_df = df_cargo[df_cargo['particle'] == best_particle].sort_values('frame')
    bx_um = p_best_df['x'].values * scale
    by_um = p_best_df['y'].values * scale

    margin_um = 25.0
    x_min_z = max(0, np.min(bx_um) - margin_um)
    x_max_z = min(W * scale, np.max(bx_um) + margin_um)
    y_min_z = max(0, np.min(by_um) - margin_um)
    y_max_z = min(H * scale, np.max(by_um) + margin_um)

    width_z = x_max_z - x_min_z
    height_z = y_max_z - y_min_z
    max_dim = max(width_z, height_z, 40.0)
    cx_mid = 0.5 * (x_min_z + x_max_z)
    cy_mid = 0.5 * (y_min_z + y_max_z)
    x_min_z = max(0, cx_mid - max_dim * 0.55)
    x_max_z = min(W * scale, cx_mid + max_dim * 0.55)
    y_min_z = max(0, cy_mid - max_dim * 0.55)
    y_max_z = min(H * scale, cy_mid + max_dim * 0.55)

    im_zoom = ax_zoom.imshow(
        Q_norm_plot,
        cmap=cmap_name,
        norm=norm,
        extent=extent_um,
        interpolation='bicubic'
    )

    # ズームパネルに軌跡を時間グラデーションで描画
    points = np.array([bx_um, by_um]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    frames_norm = (p_best_df['frame'].values[:-1] - p_best_df['frame'].min()) / max(1, p_best_df['frame'].max() - p_best_df['frame'].min())

    from matplotlib.collections import LineCollection
    lc_bg = LineCollection(segments, colors='black', linewidths=5.0, zorder=5)
    ax_zoom.add_collection(lc_bg)
    lc = LineCollection(segments, cmap='autumn_r', norm=plt.Normalize(0, 1), linewidths=3.2, zorder=6)
    lc.set_array(frames_norm)
    ax_zoom.add_collection(lc)

    # 始点・終点
    ax_zoom.scatter(bx_um[0], by_um[0], color='#10B981', edgecolor='black', s=130, marker='o', zorder=7, label='Start')
    ax_zoom.scatter(bx_um[-1], by_um[-1], color='#F59E0B', edgecolor='black', s=160, marker='s', zorder=7, label='End')

    # 現在位置
    curr_best = p_best_df[p_best_df['frame'] == frame_idx]
    if not curr_best.empty:
        cbx_um = curr_best['x'].values[0] * scale
        cby_um = curr_best['y'].values[0] * scale
        ax_zoom.scatter(cbx_um, cby_um, color='#FFFFFF', edgecolor='#DC2626', s=200, linewidth=2.8, marker='*', zorder=8, label=r'Current ($t_0$)')

    ax_zoom.set_xlim(x_min_z, x_max_z)
    ax_zoom.set_ylim(y_max_z, y_min_z)

    # 左パネルにズーム枠を示す矩形を描画
    rect = Rectangle(
        (x_min_z, y_min_z),
        x_max_z - x_min_z,
        y_max_z - y_min_z,
        linewidth=2.2,
        edgecolor='#DC2626',
        facecolor='none',
        linestyle='--',
        zorder=10
    )
    ax_full.add_patch(rect)

    # ズーム用スケールバー (20 um)
    scalebar_zoom_len = 20
    scalebar_z = AnchoredSizeBar(
        ax_zoom.transData,
        scalebar_zoom_len,
        f'{scalebar_zoom_len} μm',
        'lower left',
        pad=0.5,
        color='black',
        frameon=True,
        size_vertical=1.5,
        fontproperties=fm.FontProperties(size=12, weight='bold')
    )
    scalebar_z.patch.set_boxstyle("round,pad=0.3")
    scalebar_z.patch.set_facecolor("white")
    scalebar_z.patch.set_edgecolor("#9CA3AF")
    scalebar_z.patch.set_alpha(0.9)
    ax_zoom.add_artist(scalebar_z)

    ax_zoom.set_title(f"Zoom-in on Particle {best_particle} Trajectory", fontsize=15, fontweight='bold', pad=10)
    ax_zoom.set_xlabel(r'$x\ [\mu\mathrm{m}]$', fontsize=14, labelpad=8)
    ax_zoom.set_ylabel(r'$y\ [\mu\mathrm{m}]$', fontsize=14)
    ax_zoom.tick_params(axis='both', which='major', labelsize=12)
    ax_zoom.legend(loc='lower right', fontsize=11, framealpha=0.92, edgecolor='#CBD5E1')

    # 共通カラーバーの配置 (右側に縦型配置し、洗練された目盛りラベルを付与)
    cbar_ax = fig.add_axes([0.918, 0.14, 0.016, 0.74])
    cbar = fig.colorbar(im_full, cax=cbar_ax, orientation='vertical', extend='both')
    cbar.set_ticks([-3, -2, -1, 0, 1, 2, 3])
    cbar.set_ticklabels([
        r'$-3$ (Vortex)',
        r'$-2$',
        r'$-1$',
        r'$0$ (Neutral)',
        r'$+1$',
        r'$+2$',
        r'$+3$ (Shear)'
    ])
    cbar.set_label(
        r'Normalized Okubo-Weiss Parameter $Q_{\mathrm{OW}} / \sigma_Q$',
        fontsize=13.5,
        fontweight='bold',
        labelpad=14
    )
    cbar.ax.tick_params(labelsize=11)

    # 全体タイトルの調整
    suptitle_str = "Microtubule Flow Okubo-Weiss Parameter Field and Cargo Trajectories"
    if title_suffix:
        suptitle_str += f" ({title_suffix})"
    fig.suptitle(suptitle_str, fontsize=17, fontweight='bold', y=0.98)

    # レイアウト調整 (右側カラーバーと下部x軸ラベルの十分なマージン)
    plt.subplots_adjust(top=0.91, bottom=0.12, left=0.06, right=0.88, wspace=0.18)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    fig.savefig(output_path.with_suffix('.svg'))
    fig.savefig(output_path.with_suffix('.pdf'))
    plt.close(fig)
    print(f"Saved Trajectory & Q_OW overlay plot to:\n  - {output_path}\n  - {output_path.with_suffix('.svg')}")


def run_single_dataset_analysis(
    dataset_dir: Path,
    output_dir: Path,
    max_frames: Optional[int] = None,
    frame_stride: int = 1,
    pixel_stride: int = 2,
    clip_sigmas: float = 3.0,
    cmap_name: str = 'RdBu_r'
) -> Dict:
    """単一データセットに対する完全解析と描画の実行"""
    h5_path = dataset_dir / 'GFP_flows.h5'
    csv_path = dataset_dir / 'beads_tracks.csv'

    print(f"\n=======================================================")
    print(f"Analyzing Dataset: {dataset_dir.name}")
    print(f"H5: {h5_path}")
    print(f"CSV: {csv_path}")
    print(f"=======================================================")

    res = analyze_experiment_okubo_weiss(
        flows_h5_path=h5_path,
        tracks_csv_path=csv_path,
        scale=0.11,
        frame_interval=4.0,
        max_frames=max_frames,
        frame_stride=frame_stride,
        pixel_stride=pixel_stride,
        verbose=True
    )

    # 出力ファイル名の設定
    dataset_tag = dataset_dir.name
    # ビーズ径などの情報を親ディレクトリから取得
    bead_tag = dataset_dir.parent.parent.name if 'beads' in dataset_dir.parent.parent.name else dataset_tag

    # 1. ラグランジュ的サンプリング分布の偏りプロット
    hist_out = output_dir / f"lagrangian_sampling_bias_{bead_tag}_{dataset_tag}.png"
    plot_lagrangian_sampling_bias(
        res,
        hist_out,
        title_suffix=f"{bead_tag} / {dataset_tag}"
    )

    # 2. 軌跡と Q_OW 重ね合わせ図
    map_out = output_dir / f"trajectory_q_ow_overlay_{bead_tag}_{dataset_tag}.png"
    plot_trajectory_q_ow_overlay(
        res,
        map_out,
        title_suffix=f"{bead_tag} / {dataset_tag}",
        clip_sigmas=clip_sigmas,
        cmap_name=cmap_name
    )

    # 3. 貨物サンプリングデータの保存
    cargo_csv_out = output_dir / f"cargo_q_ow_samples_{bead_tag}_{dataset_tag}.csv"
    res['cargo_df'].to_csv(cargo_csv_out, index=False)
    print(f"Saved cargo Q_OW samples to: {cargo_csv_out}")

    return res


def plot_all_beads_comparison(
    all_results: Dict[str, Dict],
    output_path: Path
) -> None:
    """
    全ビーズ径 (0.6um〜20um) におけるラグランジュ滞在バイアスの比較プロット (2パネル)
    Panel 1: 各ビーズ径の P_cargo(Q) 分布と背景 P_Euler(Q) の比較
    Panel 2: ビーズ径 vs 渦滞在割合 f_vortex / 剪断滞在割合 f_strain
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # パネル1: 確率密度分布比較
    # 代表的な背景分布 (1つの代表データから取得)
    first_res = next(iter(all_results.values()))
    bin_centers, p_euler = first_res['hist_euler']

    ax1.axvspan(-4, 0, color='#3B82F6', alpha=0.06)
    ax1.axvspan(0, 4, color='#EF4444', alpha=0.06)
    ax1.axvline(0, color='#6B7280', linestyle='--', linewidth=1.5, alpha=0.8)

    ax1.fill_between(
        bin_centers,
        0,
        p_euler,
        color='#9CA3AF',
        alpha=0.35,
        label=r'Eulerian Background $P_{\mathrm{Euler}}(Q)$'
    )
    ax1.plot(bin_centers, p_euler, color='#4B5563', linewidth=1.5, linestyle=':')

    summary_rows = []
    for b_info in BEADS_INFO:
        b_name = b_info['name']
        if b_name in all_results:
            b_res = all_results[b_name]
            bc, p_c = b_res['hist_cargo']
            diam = b_info['diameter_um']
            color = b_info['color']
            marker = b_info['marker']

            ax1.plot(
                bc,
                p_c,
                label=rf'{diam:.2f} $\mu\mathrm{{m}}$ ({b_name})',
                color=color,
                linewidth=2.5
            )

            c_stats = b_res['cargo_stats']
            e_stats = b_res['euler_stats']
            summary_rows.append({
                'bead_name': b_name,
                'diameter_um': diam,
                'cargo_count': c_stats['count'],
                'sigma_Q': b_res['sigma_Q'],
                'euler_vortex_frac': e_stats['vortex_fraction'],
                'euler_strain_frac': e_stats['strain_fraction'],
                'cargo_vortex_frac': c_stats['vortex_fraction'],
                'cargo_strain_frac': c_stats['strain_fraction'],
                'strain_bias': c_stats['strain_fraction'] - e_stats['strain_fraction'],
                'cargo_mean_norm': c_stats['mean_norm'],
                'cargo_median_norm': c_stats['median_norm'],
                'ks_stat': c_stats['ks_statistic'],
                'ks_pvalue': c_stats['ks_pvalue'],
            })

    ax1.set_xlabel(r'Normalized Okubo-Weiss Parameter $Q_{\mathrm{OW}} / \sigma_Q$', fontsize=14)
    ax1.set_ylabel(r'Probability Density $P(Q / \sigma_Q)$', fontsize=14)
    ax1.set_xlim(-3.5, 3.5)
    ax1.set_title(r'(a) Lagrangian Sampling Distribution $P_{\mathrm{cargo}}(Q)$ vs Bead Size', fontsize=15, fontweight='bold', pad=10)
    ax1.legend(loc='upper right', fontsize=10.5, framealpha=0.9)

    # パネル2: ビーズ径 vs 渦/剪断滞在割合
    if summary_rows:
        df_sum = pd.DataFrame(summary_rows).sort_values('diameter_um')
        diams = df_sum['diameter_um'].values

        # 背景平均値 (基準水平線)
        bg_strain = np.mean(df_sum['euler_strain_frac'])
        bg_vortex = np.mean(df_sum['euler_vortex_frac'])

        ax2.axhline(bg_strain * 100, color='#DC2626', linestyle=':', linewidth=2, alpha=0.8, label=f'Eulerian Strain fraction ({bg_strain*100:.1f}%)')
        ax2.axhline(bg_vortex * 100, color='#2563EB', linestyle=':', linewidth=2, alpha=0.8, label=f'Eulerian Vortex fraction ({bg_vortex*100:.1f}%)')

        ax2.plot(
            diams,
            df_sum['cargo_strain_frac'] * 100,
            marker='o',
            color='#DC2626',
            linewidth=2.8,
            markersize=9,
            label='Cargo in Strain lanes ($Q > 0$)'
        )
        ax2.plot(
            diams,
            df_sum['cargo_vortex_frac'] * 100,
            marker='s',
            color='#2563EB',
            linewidth=2.8,
            markersize=9,
            label='Cargo in Vortex cores ($Q < 0$)'
        )

        ax2.set_xscale('log')
        ax2.set_xlabel(r'Cargo Diameter $2R_c\ [\mu\mathrm{m}]$', fontsize=14)
        ax2.set_ylabel(r'Residence Fraction $[\%]$', fontsize=14)
        ax2.set_title(r'(b) Residence Fraction in Vortex vs Strain Lanes', fontsize=15, fontweight='bold', pad=10)
        ax2.legend(loc='best', fontsize=11, framealpha=0.9)
        ax2.set_ylim(0, 100)

        # サマリー CSV の出力
        csv_path = output_path.with_name('okubo_weiss_residence_bias_summary.csv')
        df_sum.to_csv(csv_path, index=False)
        print(f"Saved residence bias summary CSV to: {csv_path}")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    fig.savefig(output_path.with_suffix('.svg'))
    fig.savefig(output_path.with_suffix('.pdf'))
    plt.close(fig)
    print(f"Saved All-Beads comparison plot to:\n  - {output_path}\n  - {output_path.with_suffix('.svg')}")


def main():
    parser = argparse.ArgumentParser(
        description='Calculate Okubo-Weiss parameter from microtubule flows and plot Lagrangian residence bias & trajectory overlays.'
    )
    parser.add_argument(
        '--dataset_dir',
        type=str,
        default=None,
        help='Path to specific experiment directory containing GFP_flows.h5 and beads_tracks.csv'
    )
    parser.add_argument(
        '--root_dir',
        type=str,
        default=None,
        help='Root directory for MTsingleBeads data'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=str(CURRENT_DIR / 'figure' / 'okubo_weiss'),
        help='Output directory for generated plots and data'
    )
    parser.add_argument(
        '--max_frames',
        type=int,
        default=None,
        help='Maximum number of frames to process per experiment'
    )
    parser.add_argument(
        '--frame_stride',
        type=int,
        default=1,
        help='Frame stride for sampling optical flow (default: 1)'
    )
    parser.add_argument(
        '--pixel_stride',
        type=int,
        default=2,
        help='Pixel stride for Eulerian background accumulation (default: 2)'
    )
    parser.add_argument(
        '--clip_sigmas',
        type=float,
        default=3.0,
        help='Clipping range in units of sigma_Q for Diverging Colormap (default: 3.0)'
    )
    parser.add_argument(
        '--cmap',
        type=str,
        default='RdBu_r',
        choices=['RdBu_r', 'coolwarm'],
        help='Diverging colormap name (default: RdBu_r)'
    )
    parser.add_argument(
        '--all_beads',
        action='store_true',
        help='If set, analyzes representative experiments from all bead size conditions and plots size comparison.'
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    root = Path(args.root_dir) if args.root_dir else None

    if args.all_beads:
        print("Running Okubo-Weiss residence bias analysis across ALL bead sizes...")
        all_results = {}
        for b_info in BEADS_INFO:
            b_name = b_info['name']
            ds_dir = find_default_dataset(root=root, bead_name=b_name)
            if ds_dir and ds_dir.exists():
                try:
                    res = run_single_dataset_analysis(
                        dataset_dir=ds_dir,
                        output_dir=output_dir,
                        max_frames=args.max_frames,
                        frame_stride=args.frame_stride,
                        pixel_stride=args.pixel_stride,
                        clip_sigmas=args.clip_sigmas,
                        cmap_name=args.cmap
                    )
                    all_results[b_name] = res
                except Exception as e:
                    print(f"Error processing {b_name} ({ds_dir}): {e}")
                    import traceback
                    traceback.print_exc()

        if all_results:
            comp_path = output_dir / "lagrangian_bias_vs_bead_size.png"
            plot_all_beads_comparison(all_results, comp_path)

    else:
        # 単一データセット解析
        target_dir = resolve_dataset_path(args.dataset_dir) if args.dataset_dir else find_default_dataset(root=root, bead_name='beads3um')
        if target_dir is None or not target_dir.exists():
            print("Error: Could not find a valid experiment dataset directory.")
            print(f"Looked in possible roots: {[str(r) for r in POSSIBLE_ROOTS]}")
            return

        run_single_dataset_analysis(
            dataset_dir=target_dir,
            output_dir=output_dir,
            max_frames=args.max_frames,
            frame_stride=args.frame_stride,
            pixel_stride=args.pixel_stride,
            clip_sigmas=args.clip_sigmas,
            cmap_name=args.cmap
        )

    print("\nAll Okubo-Weiss analyses and plots successfully completed!")


if __name__ == '__main__':
    main()
