"""
speed_vs_angle_analysis.py

貨物微粒子（蛍光ビーズ）の速さ比 R(Δθ) = <v>(Δθ) / <v> および速さ v [μm/s] と進行方向角度変化 Δθ の関係を一括解析・可視化するスクリプトです。

全ビーズサイズ（0.63μm, 1.18μm, 3.37μm, 5.0μm, 7.24μm, 20μm）および
複数のラグタイム Δt において、
1. 縦軸 R(Δθ) = <v>(Δθ)/<v>、横軸 符号付き角度変化 Δθ ∈ [-180°, 180°] のグラフ描画
2. 縦軸 R(|Δθ|)、横軸 絶対角度変化 |Δθ| ∈ [0°, 180°] のグラフ描画
3. 縦軸 速さ v [μm/s]、横軸 角度変化 Δθ のグラフ描画
4. 速さと方向転換角の相関係数 (Pearson r, Spearman rho, 配向コサイン相関) の算出 & 時間発展プロット
5. 2D結合確率密度 P(Δθ, v) の 6-Panel ヒートマップ
6. 統計サマリー CSV の出力
を行います。
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# 親ディレクトリのパス設定
current_dir = Path(__file__).parent.resolve()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from libs import speed_vs_angle as sva

# スタイルの適用
style_path = current_dir / 'libs' / 'my_style.mplstyle'
if style_path.exists():
    try:
        plt.style.use(str(style_path))
        style_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    except Exception:
        style_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
else:
    style_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# ビーズ条件設定
BEADS_INFO = [
    {"name": "beads06um", "diameter_um": 0.63, "marker": "^", "color": style_colors[0]},
    {"name": "beads1um",  "diameter_um": 1.18, "marker": "o", "color": style_colors[1]},
    {"name": "beads3um",  "diameter_um": 3.37, "marker": "d", "color": style_colors[2]},
    {"name": "beads5um",  "diameter_um": 5.00, "marker": 10,  "color": style_colors[3]},
    {"name": "beads7um",  "diameter_um": 7.24, "marker": 11,  "color": style_colors[4]},
    {"name": "beads20um", "diameter_um": 20.0, "marker": "s", "color": style_colors[5]},
]

POSSIBLE_ROOTS = [
    Path('/Volumes/data-1/Sasaki/MTsingleBeads'),
    Path('/Volumes/data-1/sasaki/MTsingleBeads'),
    Path('/Volumes/data/Sasaki/MTsingleBeads'),
    Path('/Volumes/data/sasaki/MTsingleBeads'),
    Path('/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads'),
    Path('/mnt/NAS-Ebanaru/sasaki/MTsingleBeads'),
]


def find_default_root():
    for r in POSSIBLE_ROOTS:
        if r.exists():
            for b in ['beads1um', 'beads06um', 'beads3um', 'beads5um', 'beads7um', 'beads20um']:
                if (r / b).exists() and len(list((r / b).glob('*/*beads_tracks.csv'))) > 0:
                    return r
    for r in POSSIBLE_ROOTS:
        if r.exists():
            return r
    return POSSIBLE_ROOTS[0]


def find_experiment_dirs(root_dir, bead_name):
    base = Path(root_dir) / bead_name
    if not base.exists():
        return []
    exp_dirs = []
    for p in sorted(base.glob("*/*")):
        if p.is_dir() and (p / "beads_tracks.csv").exists():
            exp_dirs.append(p)
    if not exp_dirs:
        for p in sorted(base.glob("*")):
            if p.is_dir() and (p / "beads_tracks.csv").exists():
                exp_dirs.append(p)
    return exp_dirs


def safe_save_csv(df, target_path, max_retries=5):
    target_path = Path(target_path)
    for attempt in range(max_retries):
        try:
            df.to_csv(target_path, index=False)
            return
        except Exception as e:
            if attempt == max_retries - 1:
                try:
                    csv_text = df.to_csv(index=False)
                    with open(str(target_path), 'w', encoding='utf-8') as f:
                        f.write(csv_text)
                    return
                except Exception:
                    print(f"[WARNING] Could not save {target_path}: {e}", flush=True)
                    return
            import time
            time.sleep(0.5)


_TRACKS_CACHE = {}


def collect_angle_speed_data(exp_dirs, tau, scale=0.11, frame_interval=4.0, unit='deg', speed_mode='mean', signed=False):
    """
    複数の実験ディレクトリから指定 tau の (角度変化, 速さ) データを集約する。
    """
    pairs_list = []
    pooled_angles = []
    pooled_speeds = []

    for d in exp_dirs:
        d_str = str(d)
        if d_str in _TRACKS_CACHE:
            df_tracks = _TRACKS_CACHE[d_str]
        else:
            tracks_path = d / "beads_tracks.csv"
            try:
                df_tracks = pd.read_csv(tracks_path)
                _TRACKS_CACHE[d_str] = df_tracks
            except Exception as e:
                print(f"[WARNING] Failed to read {tracks_path}: {e}", flush=True)
                _TRACKS_CACHE[d_str] = None
                continue

        if df_tracks is None:
            continue

        try:
            d_th, speeds = sva.calc_speed_and_turning_angle(
                df_tracks,
                tau=tau,
                scale=scale,
                frame_interval=frame_interval,
                unit=unit,
                speed_mode=speed_mode,
                signed=signed
            )
            if len(d_th) > 0:
                pairs_list.append((d_th, speeds))
                pooled_angles.extend(d_th)
                pooled_speeds.extend(speeds)
        except Exception as e:
            print(f"[ERROR] Error calculating speed and angle for {d}: {e}", flush=True)

    return {
        "pairs": pairs_list,
        "pooled_angles": np.array(pooled_angles),
        "pooled_speeds": np.array(pooled_speeds)
    }


def plot_speed_vs_angle_across_beads(beads_data, tau, frame_interval, unit, out_path,
                                     bins=24, signed=True, normalize=True, error_style='band'):
    """
    全ビーズサイズを1つの図で比較する R(Δθ) vs Δθ または <v>(Δθ) vs Δθ プロットを作成・保存する。
    """
    fig, ax = plt.subplots(figsize=(7.8, 5.5))
    tau_sec = tau * frame_interval
    unit_str = r'^\circ' if unit == 'deg' else r'\mathrm{rad}'
    
    if signed:
        xlabel = rf'Turning angle $\Delta\theta$ [${unit_str}$]'
        ylabel = r'Speed ratio $R(\Delta\theta) = \langle v \rangle(\Delta\theta) / \langle v \rangle$' if normalize else r'Speed $v$ [$\mu\mathrm{m/s}$]'
        bin_range = (-180.0, 180.0) if unit == 'deg' else (-np.pi, np.pi)
    else:
        xlabel = rf'Turning angle magnitude $|\Delta\theta|$ [${unit_str}$]'
        ylabel = r'Speed ratio $R(|\Delta\theta|) = \langle v \rangle(|\Delta\theta|) / \langle v \rangle$' if normalize else r'Speed $v$ [$\mu\mathrm{m/s}$]'
        bin_range = (0.0, 180.0) if unit == 'deg' else (0.0, np.pi)

    for item in BEADS_INFO:
        b_name = item["name"]
        if b_name not in beads_data or len(beads_data[b_name]["pairs"]) == 0:
            continue

        pairs = beads_data[b_name]["pairs"]
        angles = beads_data[b_name]["pooled_angles"]
        speeds = beads_data[b_name]["pooled_speeds"]

        # 相関係数の算出
        corr_info = sva.calc_speed_angle_correlation(angles, speeds, unit=unit)
        r_val = corr_info['pearson_r']

        if normalize:
            centers, mean_R, std_R, counts, _ = sva.calc_ensemble_speed_ratio_R(
                pairs, bins=bins, bin_range=bin_range, signed=signed, unit=unit
            )
            valid = np.isfinite(mean_R) & (counts >= 5)
            if not np.any(valid):
                continue
            plot_y = mean_R[valid]
            plot_std = std_R[valid]
            r_str = f'$r={r_val:.2f}$' if np.isfinite(r_val) else ''
            label_text = rf'{item["diameter_um"]:.2f} $\mu\mathrm{{m}}$ ({r_str})'
        else:
            centers, mean_v, std_v, counts, _ = sva.calc_ensemble_speed_vs_angle(
                pairs, bins=bins, bin_range=bin_range, signed=signed, unit=unit
            )
            valid = np.isfinite(mean_v) & (counts >= 5)
            if not np.any(valid):
                continue
            plot_y = mean_v[valid]
            plot_std = std_v[valid]
            r_str = f'$r={r_val:.2f}$' if np.isfinite(r_val) else ''
            label_text = rf'{item["diameter_um"]:.2f} $\mu\mathrm{{m}}$ ({r_str})'

        # 1. 平均プロット
        ax.plot(
            centers[valid],
            plot_y,
            marker=item["marker"],
            color=item["color"],
            label=label_text,
            markersize=5.5,
            linewidth=1.8,
            alpha=0.9
        )

        # 2. 実験間標準偏差エラーバンド
        if error_style in ['band', 'both']:
            ax.fill_between(
                centers[valid],
                np.clip(plot_y - plot_std, 0, None),
                plot_y + plot_std,
                edgecolor=item["color"],
                facecolor=mcolors.to_rgba(item["color"], alpha=0.18),
                linewidth=0.5
            )
        if error_style in ['bar', 'both']:
            ax.errorbar(
                centers[valid],
                plot_y,
                yerr=plot_std,
                fmt='none',
                ecolor=item["color"],
                elinewidth=1.0,
                capsize=2,
                alpha=0.6
            )

    if unit == 'deg':
        if signed:
            ax.set_xlim(-180, 180)
            ax.set_xticks([-180, -90, 0, 90, 180])
            ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
        else:
            ax.set_xlim(0, 180)
            ax.set_xticks([0, 30, 60, 90, 120, 150, 180])
    else:
        if signed:
            ax.set_xlim(-np.pi, np.pi)
            ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
            ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
            ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
        else:
            ax.set_xlim(0, np.pi)
            ax.set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
            ax.set_xticklabels([r'$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])

    if normalize:
        ax.axhline(1.0, color='black', linestyle=':', alpha=0.6, label='No correlation ($R=1.0$)')

    angle_var = r'\Delta\theta' if signed else r'|\Delta\theta|'
    title_prefix = rf"Speed Ratio $R({angle_var}) = \langle v \rangle({angle_var}) / \langle v \rangle$" if normalize else rf"Speed vs Turning Angle"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f'{title_prefix} ($\Delta t = {tau_sec:.1f}\\mathrm{{s}}$)')
    ax.legend(frameon=True, fontsize=8, loc='best')
    ax.grid(True, which="both", ls="--", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {out_path}", flush=True)


def plot_speed_vs_angle_grid(all_tau_data, frame_interval, unit, out_path,
                             plot_tau_list=None, bins=24, signed=True, normalize=True, error_style='band'):
    """
    複数の代表ラグタイム tau をグリッド状に並べたサマリープロットを作成・保存する。
    """
    if plot_tau_list is None:
        plot_tau_list = list(all_tau_data.keys())

    n_tau = len(plot_tau_list)
    if n_tau == 0:
        return

    n_cols = min(3, n_tau)
    n_rows = (n_tau + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.8 * n_cols, 4.6 * n_rows), squeeze=False)
    unit_str = r'^\circ' if unit == 'deg' else r'\mathrm{rad}'
    
    if signed:
        xlabel = rf'$\Delta\theta$ [${unit_str}$]'
        ylabel = r'$R(\Delta\theta)$' if normalize else r'$v$ [$\mu\mathrm{m/s}$]'
        bin_range = (-180.0, 180.0) if unit == 'deg' else (-np.pi, np.pi)
    else:
        xlabel = rf'$|\Delta\theta|$ [${unit_str}$]'
        ylabel = r'$R(|\Delta\theta|)$' if normalize else r'$v$ [$\mu\mathrm{m/s}$]'
        bin_range = (0.0, 180.0) if unit == 'deg' else (0.0, np.pi)

    for idx, tau in enumerate(plot_tau_list):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r, c]
        tau_sec = tau * frame_interval

        if tau not in all_tau_data:
            continue
        beads_data = all_tau_data[tau]

        for item in BEADS_INFO:
            b_name = item["name"]
            if b_name not in beads_data or len(beads_data[b_name]["pairs"]) == 0:
                continue

            pairs = beads_data[b_name]["pairs"]
            if normalize:
                centers, mean_R, std_R, counts, _ = sva.calc_ensemble_speed_ratio_R(
                    pairs, bins=bins, bin_range=bin_range, signed=signed, unit=unit
                )
                valid = np.isfinite(mean_R) & (counts >= 5)
                if not np.any(valid):
                    continue
                plot_y = mean_R[valid]
                plot_std = std_R[valid]
            else:
                centers, mean_v, std_v, counts, _ = sva.calc_ensemble_speed_vs_angle(
                    pairs, bins=bins, bin_range=bin_range, signed=signed, unit=unit
                )
                valid = np.isfinite(mean_v) & (counts >= 5)
                if not np.any(valid):
                    continue
                plot_y = mean_v[valid]
                plot_std = std_v[valid]

            r_val = np.nan
            if len(beads_data[b_name]["pooled_angles"]) > 0:
                corr_res = sva.calc_speed_angle_correlation(
                    beads_data[b_name]["pooled_angles"], beads_data[b_name]["pooled_speeds"], unit=unit
                )
                r_val = corr_res['pearson_r']

            r_str = f'$r={r_val:.2f}$' if np.isfinite(r_val) else ''
            label_text = rf'{item["diameter_um"]:.2f} $\mu\mathrm{{m}}$ ({r_str})' if r_str else rf'{item["diameter_um"]:.2f} $\mu\mathrm{{m}}$'

            ax.plot(
                centers[valid],
                plot_y,
                marker=item["marker"],
                color=item["color"],
                label=label_text,
                markersize=4.5,
                linewidth=1.5,
                alpha=0.9
            )

            if error_style in ['band', 'both']:
                ax.fill_between(
                    centers[valid],
                    np.clip(plot_y - plot_std, 0, None),
                    plot_y + plot_std,
                    edgecolor=item["color"],
                    facecolor=mcolors.to_rgba(item["color"], alpha=0.18),
                    linewidth=0.5
                )

        if unit == 'deg':
            if signed:
                ax.set_xlim(-180, 180)
                ax.set_xticks([-180, -90, 0, 90, 180])
                ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
            else:
                ax.set_xlim(0, 180)
                ax.set_xticks([0, 45, 90, 135, 180])
        else:
            if signed:
                ax.set_xlim(-np.pi, np.pi)
                ax.set_xticks([-np.pi, 0, np.pi])
                ax.set_xticklabels([r'$-\pi$', r'$0$', r'$\pi$'])
                ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
            else:
                ax.set_xlim(0, np.pi)
                ax.set_xticks([0, np.pi / 2, np.pi])
                ax.set_xticklabels([r'$0$', r'$\pi/2$', r'$\pi$'])

        if normalize:
            ax.axhline(1.0, color='black', linestyle=':', alpha=0.5)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'$\Delta t = {tau_sec:.1f}\\mathrm{{s}}$ ($\Delta t = {tau}\\mathrm{{ frames}}$)')
        ax.grid(True, which="both", ls="--", alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7, frameon=True, loc='best')

    for idx in range(n_tau, n_rows * n_cols):
        r = idx // n_cols
        c = idx % n_cols
        axes[r, c].axis('off')

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {out_path}", flush=True)


def plot_speed_angle_joint_2d_grid(beads_data, tau, frame_interval, unit, out_path, bins=30, signed=True):
    """
    各ビーズ条件ごとに 2D 結合確率密度 P(Δθ, v) の 6-Panel グリッド図を作成・保存する。
    """
    n_beads = len(BEADS_INFO)
    n_cols = 3
    n_rows = (n_beads + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.5 * n_rows), squeeze=False)
    tau_sec = tau * frame_interval
    unit_str = r'^\circ' if unit == 'deg' else r'\mathrm{rad}'
    xlabel = rf'$\Delta\theta$ [${unit_str}$]' if signed else rf'$|\Delta\theta|$ [${unit_str}$]'
    ylabel = r'Speed $v$ [$\mu\mathrm{m/s}$]'

    if signed:
        angle_range = [-180.0, 180.0] if unit == 'deg' else [-np.pi, np.pi]
    else:
        angle_range = [0.0, 180.0] if unit == 'deg' else [0.0, np.pi]

    for idx, item in enumerate(BEADS_INFO):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r, c]
        b_name = item["name"]

        if b_name not in beads_data or len(beads_data[b_name]["pooled_angles"]) == 0:
            ax.axis('off')
            continue

        angles = beads_data[b_name]["pooled_angles"]
        speeds = beads_data[b_name]["pooled_speeds"]

        valid = np.isfinite(angles) & np.isfinite(speeds) & (angles >= angle_range[0]) & (angles <= angle_range[1])
        ang_val = angles[valid]
        spd_val = speeds[valid]

        if len(ang_val) < 10:
            ax.axis('off')
            continue

        # 2D ヒストグラム
        v_max = float(np.percentile(spd_val, 99.5)) * 1.15
        counts, xedges, yedges = np.histogram2d(
            ang_val, spd_val,
            bins=bins,
            range=[angle_range, [0, v_max]],
            density=True
        )

        X, Y = np.meshgrid(xedges, yedges)
        im = ax.pcolormesh(X, Y, counts.T, cmap='viridis', shading='auto')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r'Density')

        # 平均プロファイル線をオーバーレイ
        bin_c, mean_v, _, cnts, _ = sva.calc_ensemble_speed_vs_angle(
            beads_data[b_name]["pairs"], bins=18 if signed else 15, bin_range=tuple(angle_range), signed=signed, unit=unit
        )
        val_prof = np.isfinite(mean_v) & (cnts >= 5)
        if np.any(val_prof):
            lbl = r'$\langle v \rangle(\Delta\theta)$' if signed else r'$\langle v \rangle(|\Delta\theta|)$'
            ax.plot(bin_c[val_prof], mean_v[val_prof], 'r-o', markersize=4, linewidth=2.0, label=lbl)
            ax.legend(fontsize=7, frameon=True, loc='upper right')

        if unit == 'deg':
            if signed:
                ax.set_xlim(-180, 180)
                ax.set_xticks([-180, -90, 0, 90, 180])
            else:
                ax.set_xlim(0, 180)
                ax.set_xticks([0, 45, 90, 135, 180])
        else:
            if signed:
                ax.set_xlim(-np.pi, np.pi)
                ax.set_xticks([-np.pi, 0, np.pi])
                ax.set_xticklabels([r'$-\pi$', r'$0$', r'$\pi$'])
            else:
                ax.set_xlim(0, np.pi)
                ax.set_xticks([0, np.pi / 2, np.pi])
                ax.set_xticklabels([r'$0$', r'$\pi/2$', r'$\pi$'])

        ax.set_ylim(0, v_max)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{item["diameter_um"]:.2f} $\\mu\\mathrm{{m}}$ ({item["name"]})')
        ax.grid(True, which="both", ls="--", alpha=0.3)

    for idx in range(n_beads, n_rows * n_cols):
        r = idx // n_cols
        c = idx % n_cols
        axes[r, c].axis('off')

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {out_path}", flush=True)


def plot_correlation_evolution(df_corr, out_path):
    """
    速さ v と方向転換角 |Δθ| の相関係数 r(v, |Δθ|) のラグタイム時間発展プロットを作成・保存する。
    """
    fig, ax = plt.subplots(figsize=(6.8, 5.0))

    for item in BEADS_INFO:
        b_name = item["name"]
        df_b = df_corr[df_corr['bead_name'] == b_name].sort_values('lag_time_s')
        if df_b.empty or 'pearson_r' not in df_b.columns:
            continue

        tau_s = df_b['lag_time_s'].to_numpy()
        r_val = df_b['pearson_r'].to_numpy()

        ax.plot(
            tau_s,
            r_val,
            marker=item["marker"],
            color=item["color"],
            label=f'{item["diameter_um"]:.2f} $\\mu\\mathrm{{m}}$',
            markersize=5.5,
            linewidth=1.6,
            alpha=0.9
        )

    ax.set_xscale('log')
    ax.set_xlabel(r'Lag time $\Delta t$ [$\mathrm{s}$]')
    ax.set_ylabel(r'Correlation coefficient $r(v, |\Delta\theta|)$')
    ax.axhline(0, color='black', linestyle=':', alpha=0.6, label='No correlation ($r=0$)')
    ax.set_ylim(-0.6, 0.25)
    ax.legend(frameon=True, fontsize=8, loc='best')
    ax.grid(True, which="both", ls="--", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {out_path}", flush=True)


def plot_correlation_vs_diameter(df_corr, out_path, plot_tau_list=None, frame_interval=4.0):
    """
    粒子直径 d [um] に対する速さ・角度相関係数 r(v, |Δθ|) のプロットを作成・保存する。
    """
    if plot_tau_list is None:
        plot_tau_list = sorted(df_corr['tau_frame'].unique())

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    cmap = plt.cm.viridis
    colors = [cmap(i) for i in np.linspace(0.1, 0.9, len(plot_tau_list))]

    for idx, tau in enumerate(plot_tau_list):
        df_t = df_corr[df_corr['tau_frame'] == tau].sort_values('diameter_um')
        if df_t.empty:
            continue
        diams = df_t['diameter_um'].to_numpy()
        r_vals = df_t['pearson_r'].to_numpy()
        tau_sec = tau * frame_interval

        valid = np.isfinite(r_vals) & np.isfinite(diams)
        if not np.any(valid):
            continue

        ax.plot(
            diams[valid],
            r_vals[valid],
            marker='o',
            color=colors[idx],
            label=rf'$\Delta t = {tau_sec:.0f}\mathrm{{s}}$',
            markersize=6.0,
            linewidth=1.8,
            alpha=0.9
        )

    ax.set_xscale('log')
    ax.set_xlabel(r'Bead diameter $d$ [$\mu\mathrm{m}$]')
    ax.set_ylabel(r'Correlation coefficient $r(v, |\Delta\theta|)$')
    ax.axhline(0, color='black', linestyle=':', alpha=0.6, label='No correlation ($r=0$)')
    ax.set_xticks([0.63, 1.18, 3.37, 5.0, 7.24, 20.0])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(-0.55, 0.25)
    ax.legend(frameon=True, fontsize=8, loc='best')
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.set_title(r'Speed–Angle Correlation $r(v, |\Delta\theta|)$ vs Bead Diameter')

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {out_path}", flush=True)


def plot_speed_contrast_evolution(df_contrast, out_path):
    """
    規格化速さコントラスト Δv(θ) = (<v>(0°) - <v>(180°)) / <v> = R(0°) - R(180°) の
    ラグタイム時間発展プロットを作成・保存する。
    """
    fig, ax = plt.subplots(figsize=(7.0, 5.2))

    for item in BEADS_INFO:
        b_name = item["name"]
        df_b = df_contrast[df_contrast['bead_name'] == b_name].sort_values('lag_time_s')
        if df_b.empty or 'delta_v_norm_mean' not in df_b.columns:
            continue

        tau_s = df_b['lag_time_s'].to_numpy()
        d_v = df_b['delta_v_norm_mean'].to_numpy()
        d_v_sem = df_b['delta_v_norm_sem'].to_numpy()

        valid = np.isfinite(d_v)
        if not np.any(valid):
            continue

        ax.errorbar(
            tau_s[valid],
            d_v[valid],
            yerr=d_v_sem[valid],
            marker=item["marker"],
            color=item["color"],
            label=f'{item["diameter_um"]:.2f} $\\mu\\mathrm{{m}}$',
            markersize=5.5,
            linewidth=1.6,
            capsize=2.5,
            alpha=0.9
        )

    ax.set_xscale('log')
    ax.set_xlabel(r'Lag time $\Delta t$ [$\mathrm{s}$]')
    ax.set_ylabel(r'Speed contrast $\Delta v = \frac{\langle v \rangle(0^\circ) - \langle v \rangle(180^\circ)}{\langle v \rangle} = R(0^\circ) - R(180^\circ)$')
    ax.axhline(0, color='black', linestyle=':', alpha=0.6, label='No contrast ($\Delta v = 0$)')
    ax.legend(frameon=True, fontsize=8, loc='best')
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.set_title(r'Normalized Speed Contrast $\Delta v(\Delta t) = R(0^\circ) - R(180^\circ)$')

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {out_path}", flush=True)


def plot_speed_contrast_vs_diameter(df_contrast, out_path, plot_tau_list=None, frame_interval=4.0):
    """
    粒子径 d [um] に対する速さコントラスト Δv = R(0°) - R(180°) のプロットを作成・保存する。
    """
    if plot_tau_list is None:
        plot_tau_list = sorted(df_contrast['tau_frame'].unique())

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    cmap = plt.cm.viridis
    colors = [cmap(i) for i in np.linspace(0.1, 0.9, len(plot_tau_list))]

    for idx, tau in enumerate(plot_tau_list):
        df_t = df_contrast[df_contrast['tau_frame'] == tau].sort_values('diameter_um')
        if df_t.empty:
            continue
        diams = df_t['diameter_um'].to_numpy()
        d_v = df_t['delta_v_norm_mean'].to_numpy()
        d_v_sem = df_t['delta_v_norm_sem'].to_numpy()
        tau_sec = tau * frame_interval

        valid = np.isfinite(d_v) & np.isfinite(diams)
        if not np.any(valid):
            continue

        ax.errorbar(
            diams[valid],
            d_v[valid],
            yerr=d_v_sem[valid],
            marker='o',
            color=colors[idx],
            label=rf'$\Delta t = {tau_sec:.0f}\mathrm{{s}}$',
            markersize=5.5,
            linewidth=1.6,
            capsize=2.5,
            alpha=0.9
        )

    ax.set_xscale('log')
    ax.set_xlabel(r'Bead diameter $d$ [$\mu\mathrm{m}$]')
    ax.set_ylabel(r'Speed contrast $\Delta v = R(0^\circ) - R(180^\circ)$')
    ax.axhline(0, color='black', linestyle=':', alpha=0.6)
    ax.legend(frameon=True, fontsize=8, loc='best')
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.set_title(r'Speed Contrast $\Delta v$ vs Bead Diameter')

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Cargo particle speed ratio R(Δθ) and speed vs turning angle analysis.")
    parser.add_argument('--root_dir', type=str, default=None,
                        help="Root directory containing bead conditions.")
    parser.add_argument('--beads', type=str, nargs='+', default=['all'],
                        help="Beads conditions to analyze (e.g. beads06um beads1um ... or 'all').")
    DEFAULT_FINE_TAU = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 25, 28, 30, 35, 40, 45, 50]
    DEFAULT_PLOT_TAU = [1, 2, 5, 10, 25]

    parser.add_argument('--tau', type=int, nargs='+', default=DEFAULT_FINE_TAU,
                        help="Lag time list in frames for evaluation.")
    parser.add_argument('--tau_seconds', type=float, nargs='+', default=None,
                        help="Lag time in seconds (overrides --tau if provided).")
    parser.add_argument('--plot_tau', type=int, nargs='+', default=DEFAULT_PLOT_TAU,
                        help="Representative lag times in frames to plot.")
    parser.add_argument('--plot_tau_seconds', type=float, nargs='+', default=None,
                        help="Representative lag times in seconds to plot.")
    parser.add_argument('--scale', type=float, default=0.11,
                        help="Spatial conversion scale (um/pixel, default: 0.11).")
    parser.add_argument('--frame_interval', type=float, default=4.0,
                        help="Time interval between frames in seconds (default: 4.0).")
    parser.add_argument('--unit', type=str, default='deg', choices=['deg', 'rad'],
                        help="Angular unit: 'deg' (degrees) or 'rad' (radians) (default: deg).")
    parser.add_argument('--speed_mode', type=str, default='mean', choices=['mean', 'incoming', 'outgoing'],
                        help="Speed definition: 'mean' ((v1+v2)/2), 'incoming' (v1), or 'outgoing' (v2).")
    parser.add_argument('--bins', type=int, default=24,
                        help="Number of angle bins for profile (default: 24).")
    parser.add_argument('--error_style', type=str, default='band', choices=['band', 'bar', 'both', 'none'],
                        help="Error representation style across experiments (default: band).")
    parser.add_argument('--out_dir', type=str, default=None,
                        help="Output directory to save plots and CSVs. Defaults to root_dir/figure/speed_vs_angle.")
    args = parser.parse_args()

    # ルートディレクトリの確定
    if args.root_dir is not None:
        root_dir = Path(args.root_dir)
    else:
        root_dir = find_default_root()

    print(f"=== Cargo Particle Speed Ratio R(Δθ) and Speed vs Angle Analysis ===", flush=True)
    print(f"Root Directory: {root_dir}", flush=True)

    # 出力先ディレクトリ
    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        out_dir = root_dir / 'figure' / 'speed_vs_angle'
    out_dir.mkdir(parents=True, exist_ok=True)

    # 対象ビーズ条件の抽出
    if 'all' in args.beads:
        selected_beads = [b['name'] for b in BEADS_INFO]
    else:
        selected_beads = []
        for item in args.beads:
            for b in item.split(','):
                b = b.strip()
                if b == 'all':
                    selected_beads = [info['name'] for info in BEADS_INFO]
                    break
                if any(info['name'] == b for info in BEADS_INFO):
                    selected_beads.append(b)
                else:
                    print(f"[WARNING] Unknown bead condition: {b}", flush=True)

    # ラグタイムの決定
    if args.tau_seconds is not None:
        tau_list = sorted(list(set([max(1, int(round(ts / args.frame_interval))) for ts in args.tau_seconds])))
    else:
        tau_list = sorted(list(set(args.tau)))

    if args.plot_tau_seconds is not None:
        plot_tau_list = sorted(list(set([max(1, int(round(ts / args.frame_interval))) for ts in args.plot_tau_seconds])))
    else:
        plot_tau_list = sorted(list(set(args.plot_tau)))

    for pt in plot_tau_list:
        if pt not in tau_list:
            tau_list.append(pt)
    tau_list = sorted(tau_list)

    print(f"Evaluation lag times (frames): {tau_list}", flush=True)
    print(f"Plotting lag times (frames): {plot_tau_list}", flush=True)

    profile_records_signed = []
    profile_records_abs = []
    corr_records = []
    contrast_records = []
    plot_tau_data_signed = {}
    plot_tau_data_abs = {}

    bin_range_signed = (-180.0, 180.0) if args.unit == 'deg' else (-np.pi, np.pi)
    bin_range_abs = (0.0, 180.0) if args.unit == 'deg' else (0.0, np.pi)

    for tau in tau_list:
        tau_sec = tau * args.frame_interval
        beads_data_signed = {}
        beads_data_abs = {}

        for bead_name in selected_beads:
            exp_dirs = find_experiment_dirs(root_dir, bead_name)
            if not exp_dirs:
                continue

            # 1. 符号付き角度データ Δθ ∈ [-180, 180]
            speed_dict_signed = collect_angle_speed_data(
                exp_dirs, tau=tau, scale=args.scale, frame_interval=args.frame_interval,
                unit=args.unit, speed_mode=args.speed_mode, signed=True
            )
            beads_data_signed[bead_name] = speed_dict_signed

            # 2. 絶対角度データ |Δθ| ∈ [0, 180]
            speed_dict_abs = collect_angle_speed_data(
                exp_dirs, tau=tau, scale=args.scale, frame_interval=args.frame_interval,
                unit=args.unit, speed_mode=args.speed_mode, signed=False
            )
            beads_data_abs[bead_name] = speed_dict_abs

            # 相関係数の算出 & 集計
            if len(speed_dict_abs["pooled_angles"]) > 0:
                item_info = next((b for b in BEADS_INFO if b["name"] == bead_name), None)
                diam = item_info["diameter_um"] if item_info else np.nan

                corr_info = sva.calc_speed_angle_correlation(
                    speed_dict_abs["pooled_angles"], speed_dict_abs["pooled_speeds"], unit=args.unit
                )
                corr_records.append({
                    'bead_name': bead_name,
                    'diameter_um': diam,
                    'tau_frame': tau,
                    'lag_time_s': tau_sec,
                    'unit': args.unit,
                    'pearson_r': corr_info['pearson_r'],
                    'pearson_pvalue': corr_info['pearson_pvalue'],
                    'spearman_rho': corr_info['spearman_rho'],
                    'spearman_pvalue': corr_info['spearman_pvalue'],
                    'cos_r': corr_info['cos_r'],
                    'linear_slope': corr_info['linear_slope'],
                    'normalized_slope': corr_info['normalized_slope'],
                    'n_points': corr_info['n_points'],
                    'mean_speed': corr_info['mean_speed'],
                    'std_speed': corr_info['std_speed'],
                    'mean_angle': corr_info['mean_angle'],
                    'std_angle': corr_info['std_angle']
                })

            # プロファイルの集計 (符号付き)
            if len(speed_dict_signed["pairs"]) > 0:
                c_s, m_v_s, s_v_s, cnts_s, _ = sva.calc_ensemble_speed_vs_angle(
                    speed_dict_signed["pairs"], bins=args.bins, bin_range=bin_range_signed, signed=True, unit=args.unit
                )
                _, m_R_s, s_R_s, _, _ = sva.calc_ensemble_speed_ratio_R(
                    speed_dict_signed["pairs"], bins=args.bins, bin_range=bin_range_signed, signed=True, unit=args.unit
                )
                overall_mean_v = float(np.mean(speed_dict_signed["pooled_speeds"])) if len(speed_dict_signed["pooled_speeds"]) > 0 else np.nan

                for b_idx in range(len(c_s)):
                    if cnts_s[b_idx] > 0:
                        profile_records_signed.append({
                            'bead_name': bead_name,
                            'tau_frame': tau,
                            'lag_time_s': tau_sec,
                            'unit': args.unit,
                            'angle_center': float(c_s[b_idx]),
                            'mean_speed_um_s': float(m_v_s[b_idx]) if np.isfinite(m_v_s[b_idx]) else np.nan,
                            'std_speed_um_s': float(s_v_s[b_idx]) if np.isfinite(s_v_s[b_idx]) else np.nan,
                            'speed_ratio_R': float(m_R_s[b_idx]) if np.isfinite(m_R_s[b_idx]) else np.nan,
                            'speed_ratio_R_std': float(s_R_s[b_idx]) if np.isfinite(s_R_s[b_idx]) else np.nan,
                            'count': int(cnts_s[b_idx]),
                            'overall_mean_speed': overall_mean_v
                        })

            # プロファイルの集計 (絶対値)
            if len(speed_dict_abs["pairs"]) > 0:
                c_a, m_v_a, s_v_a, cnts_a, _ = sva.calc_ensemble_speed_vs_angle(
                    speed_dict_abs["pairs"], bins=args.bins // 2 if args.bins >= 16 else args.bins,
                    bin_range=bin_range_abs, signed=False, unit=args.unit
                )
                _, m_R_a, s_R_a, _, _ = sva.calc_ensemble_speed_ratio_R(
                    speed_dict_abs["pairs"], bins=args.bins // 2 if args.bins >= 16 else args.bins,
                    bin_range=bin_range_abs, signed=False, unit=args.unit
                )
                overall_mean_v = float(np.mean(speed_dict_abs["pooled_speeds"])) if len(speed_dict_abs["pooled_speeds"]) > 0 else np.nan

                for b_idx in range(len(c_a)):
                    if cnts_a[b_idx] > 0:
                        profile_records_abs.append({
                            'bead_name': bead_name,
                            'tau_frame': tau,
                            'lag_time_s': tau_sec,
                            'unit': args.unit,
                            'angle_center': float(c_a[b_idx]),
                            'mean_speed_um_s': float(m_v_a[b_idx]) if np.isfinite(m_v_a[b_idx]) else np.nan,
                            'std_speed_um_s': float(s_v_a[b_idx]) if np.isfinite(s_v_a[b_idx]) else np.nan,
                            'speed_ratio_R': float(m_R_a[b_idx]) if np.isfinite(m_R_a[b_idx]) else np.nan,
                            'speed_ratio_R_std': float(s_R_a[b_idx]) if np.isfinite(s_R_a[b_idx]) else np.nan,
                            'count': int(cnts_a[b_idx]),
                            'overall_mean_speed': overall_mean_v
                        })

                # 3. 規格化速さコントラスト Δv = (<v>(0°) - <v>(180°)) / <v> = R(0°) - R(180°) の算出
                contrast_info = sva.calc_speed_contrast_delta_v(
                    speed_dict_abs["pairs"], angle_threshold_deg=20.0, unit=args.unit
                )
                contrast_records.append({
                    'bead_name': bead_name,
                    'diameter_um': diam,
                    'tau_frame': tau,
                    'lag_time_s': tau_sec,
                    'unit': args.unit,
                    'delta_v_norm_mean': contrast_info['delta_v_norm_mean'],
                    'delta_v_norm_std': contrast_info['delta_v_norm_std'],
                    'delta_v_norm_sem': contrast_info['delta_v_norm_sem'],
                    'delta_v_abs_mean': contrast_info['delta_v_abs_mean'],
                    'delta_v_abs_std': contrast_info['delta_v_abs_std'],
                    'v_0_mean': contrast_info['v_0_mean'],
                    'v_180_mean': contrast_info['v_180_mean'],
                    'v_overall_mean': contrast_info['v_overall_mean'],
                    'R_0_mean': contrast_info['R_0_mean'],
                    'R_180_mean': contrast_info['R_180_mean'],
                    'n_experiments': contrast_info['n_experiments']
                })

        # 代表ラグタイムでのプロット保存
        if tau in plot_tau_list:
            print(f"  Plotting Speed and Ratio R for tau={tau} frames ({tau_sec:.1f} s)...", flush=True)
            plot_tau_data_signed[tau] = beads_data_signed
            plot_tau_data_abs[tau] = beads_data_abs

            # 1. 縦軸 R(Δθ)、横軸 符号付き Δθ プロット
            r_signed_path = out_dir / f"speed_ratio_R_signed_tau{tau_sec:.0f}s.svg"
            plot_speed_vs_angle_across_beads(
                beads_data_signed, tau=tau, frame_interval=args.frame_interval,
                unit=args.unit, out_path=r_signed_path, bins=args.bins,
                signed=True, normalize=True, error_style=args.error_style
            )
            plot_speed_vs_angle_across_beads(
                beads_data_signed, tau=tau, frame_interval=args.frame_interval,
                unit=args.unit, out_path=r_signed_path.with_suffix('.png'), bins=args.bins,
                signed=True, normalize=True, error_style=args.error_style
            )

            # 2. 縦軸 R(|Δθ|)、横軸 絶対角度 |Δθ| プロット
            r_abs_path = out_dir / f"speed_ratio_R_abs_tau{tau_sec:.0f}s.svg"
            plot_speed_vs_angle_across_beads(
                beads_data_abs, tau=tau, frame_interval=args.frame_interval,
                unit=args.unit, out_path=r_abs_path, bins=args.bins // 2 if args.bins >= 16 else args.bins,
                signed=False, normalize=True, error_style=args.error_style
            )
            plot_speed_vs_angle_across_beads(
                beads_data_abs, tau=tau, frame_interval=args.frame_interval,
                unit=args.unit, out_path=r_abs_path.with_suffix('.png'), bins=args.bins // 2 if args.bins >= 16 else args.bins,
                signed=False, normalize=True, error_style=args.error_style
            )

            # 3. 縦軸 速さ v、横軸 符号付き Δθ プロット
            v_signed_path = out_dir / f"speed_vs_angle_tau{tau_sec:.0f}s.svg"
            plot_speed_vs_angle_across_beads(
                beads_data_signed, tau=tau, frame_interval=args.frame_interval,
                unit=args.unit, out_path=v_signed_path, bins=args.bins,
                signed=True, normalize=False, error_style=args.error_style
            )
            plot_speed_vs_angle_across_beads(
                beads_data_signed, tau=tau, frame_interval=args.frame_interval,
                unit=args.unit, out_path=v_signed_path.with_suffix('.png'), bins=args.bins,
                signed=True, normalize=False, error_style=args.error_style
            )

            # 4. 2D 結合密度グリッドプロット (絶対値 |Δθ| vs v)
            joint_2d_path = out_dir / f"speed_angle_joint_2d_tau{tau_sec:.0f}s.svg"
            plot_speed_angle_joint_2d_grid(
                beads_data_abs, tau=tau, frame_interval=args.frame_interval,
                unit=args.unit, out_path=joint_2d_path, bins=25, signed=False
            )
            plot_speed_angle_joint_2d_grid(
                beads_data_abs, tau=tau, frame_interval=args.frame_interval,
                unit=args.unit, out_path=joint_2d_path.with_suffix('.png'), bins=25, signed=False
            )

            # 符号付き 2D 結合密度グリッドプロット (signed Δθ vs v)
            joint_2d_signed_path = out_dir / f"speed_angle_joint_2d_signed_tau{tau_sec:.0f}s.svg"
            plot_speed_angle_joint_2d_grid(
                beads_data_signed, tau=tau, frame_interval=args.frame_interval,
                unit=args.unit, out_path=joint_2d_signed_path, bins=30, signed=True
            )
            plot_speed_angle_joint_2d_grid(
                beads_data_signed, tau=tau, frame_interval=args.frame_interval,
                unit=args.unit, out_path=joint_2d_signed_path.with_suffix('.png'), bins=30, signed=True
            )

    # 代表ラグタイムのグリッドプロット保存
    if plot_tau_data_signed:
        # 1. R(Δθ) 符号付き グリッド
        grid_r_signed_path = out_dir / "speed_ratio_R_signed_grid.svg"
        plot_speed_vs_angle_grid(
            plot_tau_data_signed, frame_interval=args.frame_interval,
            unit=args.unit, out_path=grid_r_signed_path, plot_tau_list=plot_tau_list,
            bins=args.bins, signed=True, normalize=True, error_style=args.error_style
        )
        plot_speed_vs_angle_grid(
            plot_tau_data_signed, frame_interval=args.frame_interval,
            unit=args.unit, out_path=grid_r_signed_path.with_suffix('.png'), plot_tau_list=plot_tau_list,
            bins=args.bins, signed=True, normalize=True, error_style=args.error_style
        )

        # 2. R(|Δθ|) 絶対値 グリッド
        grid_r_abs_path = out_dir / "speed_ratio_R_abs_grid.svg"
        plot_speed_vs_angle_grid(
            plot_tau_data_abs, frame_interval=args.frame_interval,
            unit=args.unit, out_path=grid_r_abs_path, plot_tau_list=plot_tau_list,
            bins=args.bins // 2 if args.bins >= 16 else args.bins, signed=False, normalize=True, error_style=args.error_style
        )
        plot_speed_vs_angle_grid(
            plot_tau_data_abs, frame_interval=args.frame_interval,
            unit=args.unit, out_path=grid_r_abs_path.with_suffix('.png'), plot_tau_list=plot_tau_list,
            bins=args.bins // 2 if args.bins >= 16 else args.bins, signed=False, normalize=True, error_style=args.error_style
        )

        # 3. 速さ v 符号付き グリッド
        grid_v_signed_path = out_dir / "speed_vs_angle_grid.svg"
        plot_speed_vs_angle_grid(
            plot_tau_data_signed, frame_interval=args.frame_interval,
            unit=args.unit, out_path=grid_v_signed_path, plot_tau_list=plot_tau_list,
            bins=args.bins, signed=True, normalize=False, error_style=args.error_style
        )
        plot_speed_vs_angle_grid(
            plot_tau_data_signed, frame_interval=args.frame_interval,
            unit=args.unit, out_path=grid_v_signed_path.with_suffix('.png'), plot_tau_list=plot_tau_list,
            bins=args.bins, signed=True, normalize=False, error_style=args.error_style
        )

    # 統計サマリー CSV の保存
    if profile_records_signed:
        df_prof_s = pd.DataFrame(profile_records_signed)
        csv_s_path = out_dir / "speed_ratio_R_signed_summary.csv"
        safe_save_csv(df_prof_s, csv_s_path)
        print(f"\n[SAVED] Signed R(Δθ) summary saved to {csv_s_path}", flush=True)

    if profile_records_abs:
        df_prof_a = pd.DataFrame(profile_records_abs)
        csv_a_path = out_dir / "speed_ratio_R_abs_summary.csv"
        safe_save_csv(df_prof_a, csv_a_path)
        print(f"[SAVED] Absolute R(|Δθ|) summary saved to {csv_a_path}", flush=True)

    # 相関係数サマリー CSV & 時間発展プロットの保存
    if corr_records:
        df_corr = pd.DataFrame(corr_records)
        csv_corr_path = out_dir / "speed_angle_correlation_summary.csv"
        safe_save_csv(df_corr, csv_corr_path)
        print(f"[SAVED] Speed-Angle correlation summary saved to {csv_corr_path}", flush=True)

        evol_path = out_dir / "speed_angle_correlation_evolution.svg"
        plot_correlation_evolution(df_corr, evol_path)
        plot_correlation_evolution(df_corr, evol_path.with_suffix('.png'))

        diam_corr_path = out_dir / "speed_angle_correlation_vs_diameter.svg"
        plot_correlation_vs_diameter(df_corr, diam_corr_path, plot_tau_list=plot_tau_list, frame_interval=args.frame_interval)
        plot_correlation_vs_diameter(df_corr, diam_corr_path.with_suffix('.png'), plot_tau_list=plot_tau_list, frame_interval=args.frame_interval)

    # 速さコントラスト Δv(θ) = (<v>(0°) - <v>(180°)) / <v> サマリー CSV & プロットの保存
    if contrast_records:
        df_contrast = pd.DataFrame(contrast_records)
        csv_contrast_path = out_dir / "speed_contrast_delta_v_summary.csv"
        safe_save_csv(df_contrast, csv_contrast_path)
        print(f"[SAVED] Speed contrast Δv summary saved to {csv_contrast_path}", flush=True)

        contrast_evol_path = out_dir / "speed_contrast_delta_v_evolution.svg"
        plot_speed_contrast_evolution(df_contrast, contrast_evol_path)
        plot_speed_contrast_evolution(df_contrast, contrast_evol_path.with_suffix('.png'))

        contrast_diam_path = out_dir / "speed_contrast_delta_v_vs_diameter.svg"
        plot_speed_contrast_vs_diameter(df_contrast, contrast_diam_path, plot_tau_list=plot_tau_list, frame_interval=args.frame_interval)
        plot_speed_contrast_vs_diameter(df_contrast, contrast_diam_path.with_suffix('.png'), plot_tau_list=plot_tau_list, frame_interval=args.frame_interval)

    print(f"\n[DONE] Speed ratio R(Δθ) and speed vs angle analysis finished successfully!", flush=True)


if __name__ == '__main__':
    main()
