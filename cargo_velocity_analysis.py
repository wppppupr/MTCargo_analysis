#!/usr/bin/env python3
"""
cargo_velocity_analysis.py

貨物微粒子（ビーズ）の直径に対する平均速度の依存性を解析・可視化するスクリプトです。
- x軸: 貨物粒子の直径 (Particle Diameter d [μm])
- y軸: 貨物粒子の平均速度 (Mean Velocity / Speed <v> [μm/s])
- y軸スケール: Linear（線形）および Log（対数）の両方のプロットを出力します。
- 外れ値除去（Tukey's IQR法 / 閾値法）により、5μmや7μmなどのトラッキング飛びや異常速度の影響を除外します。

出力ファイル:
1. figure/velocity/cargo_velocity_vs_diameter_simple_2panel.png / .svg (外れ値除去後: Linear & Log 2パネル)
2. figure/velocity/cargo_velocity_vs_diameter_simple_linear.png / .svg (外れ値除去後: Linear 単体図)
3. figure/velocity/cargo_velocity_vs_diameter_simple_log.png / .svg (外れ値除去後: Log 単体図)
4. figure/velocity/cargo_velocity_vs_diameter_outlier_comparison.png / .svg (外れ値除去前後の比較図)
5. figure/velocity/cargo_velocity_vs_diameter_detailed_2panel.png / .svg (詳細 2パネル: 中央値・実験別・HMM比較)
6. figure/velocity/cargo_velocity_vs_diameter_summary.csv (全統計サマリーCSV)
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# プロジェクトルートをインポートパスに追加
CURRENT_DIR = Path(__file__).parent.resolve()
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# ビーズ基本情報（論文・解析共通のマーカー＆カラー）
BEADS_INFO = [
    {"name": "beads06um", "diameter_um": 0.63, "marker": "^", "color": "#1f77b4"},
    {"name": "beads1um",  "diameter_um": 1.18, "marker": "o", "color": "#ff7f0e"},
    {"name": "beads3um",  "diameter_um": 3.37, "marker": "d", "color": "#2ca02c"},
    {"name": "beads5um",  "diameter_um": 5.00, "marker": "p", "color": "#d62728"},
    {"name": "beads7um",  "diameter_um": 7.24, "marker": "h", "color": "#9467bd"},
    {"name": "beads20um", "diameter_um": 20.0, "marker": "s", "color": "#8c564b"},
]

POSSIBLE_ROOTS = [
    Path('/Volumes/data/Sasaki/MTsingleBeads'),
    Path('/Volumes/data/sasaki/MTsingleBeads'),
    Path('/Volumes/data-1/Sasaki/MTsingleBeads'),
    Path('/Volumes/data-1/sasaki/MTsingleBeads'),
    Path('/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads'),
    Path('/mnt/NAS-Ebanaru/sasaki/MTsingleBeads'),
]


def find_default_root() -> Path:
    for r in POSSIBLE_ROOTS:
        if r.exists():
            for b in ['beads1um', 'beads06um', 'beads3um', 'beads5um', 'beads7um', 'beads20um']:
                if (r / b).exists():
                    return r
    for r in POSSIBLE_ROOTS:
        if r.exists():
            return r
    return POSSIBLE_ROOTS[0]


def apply_custom_style():
    style_path = CURRENT_DIR / 'libs' / 'my_style.mplstyle'
    if style_path.exists():
        try:
            plt.style.use(str(style_path))
        except Exception:
            pass


def load_raw_velocity_statistics(
    root_dir: Path,
    scale: float = 0.11,
    frame_interval: float = 4.0,
    iqr_multiplier: float = 1.5,
    outlier_method: str = 'iqr'
) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    """
    生軌跡データからビーズ径ごとの平均速度統計（外れ値除去前および除去後）を算出する
    """
    records = []
    raw_data_dict = {}

    for b in BEADS_INFO:
        b_name = b["name"]
        d_um = b["diameter_um"]
        b_dir = root_dir / b_name

        track_files = sorted(b_dir.glob("*/*/beads_tracks.csv"))
        if not track_files:
            track_files = sorted(b_dir.glob("*/*/*beads_tracks.csv")) + sorted(b_dir.glob("*beads_tracks.csv"))

        speeds_all = []
        exp_data = {}
        track_means = []

        for f in track_files:
            try:
                df = pd.read_csv(f)
            except Exception as e:
                print(f"Warning: Could not read {f}: {e}")
                continue

            if 'particle' not in df.columns or 'x' not in df.columns or 'y' not in df.columns or 'frame' not in df.columns:
                continue

            exp_name = f.parent.name
            if exp_name not in exp_data:
                exp_data[exp_name] = []

            for tid, g in df.groupby('particle'):
                if len(g) < 2:
                    continue
                g = g.sort_values('frame')
                dx = np.diff(g['x'].values) * scale
                dy = np.diff(g['y'].values) * scale
                dt = np.diff(g['frame'].values) * frame_interval
                valid = dt > 0
                if np.sum(valid) == 0:
                    continue
                sp = np.sqrt(dx[valid]**2 + dy[valid]**2) / dt[valid]
                sp = sp[np.isfinite(sp)]
                if len(sp) > 0:
                    speeds_all.extend(sp)
                    exp_data[exp_name].extend(sp)
                    track_means.append(float(np.mean(sp)))

        speeds_arr = np.array(speeds_all) if len(speeds_all) > 0 else np.array([])
        n_pts_raw = len(speeds_arr)

        if n_pts_raw > 0:
            # 外れ値除去前の統計
            raw_mean_v = float(np.mean(speeds_arr))
            raw_std_v = float(np.std(speeds_arr, ddof=1)) if n_pts_raw > 1 else 0.0
            raw_sem_v = float(raw_std_v / np.sqrt(n_pts_raw))
            raw_median_v = float(np.median(speeds_arr))
            raw_q25 = float(np.percentile(speeds_arr, 25))
            raw_q75 = float(np.percentile(speeds_arr, 75))
            raw_iqr = raw_q75 - raw_q25

            # 外れ値判定基準の算出 (Tukey's IQR method: Q3 + multiplier * IQR)
            upper_bound = raw_q75 + iqr_multiplier * raw_iqr
            lower_bound = max(0.0, raw_q25 - iqr_multiplier * raw_iqr)

            # 外れ値フィルタリング
            mask_valid = (speeds_arr >= lower_bound) & (speeds_arr <= upper_bound)
            speeds_filtered = speeds_arr[mask_valid]
            n_pts_filt = len(speeds_filtered)
            n_outliers = n_pts_raw - n_pts_filt
            outlier_ratio = n_outliers / n_pts_raw if n_pts_raw > 0 else 0.0

            filt_mean_v = float(np.mean(speeds_filtered)) if n_pts_filt > 0 else np.nan
            filt_std_v = float(np.std(speeds_filtered, ddof=1)) if n_pts_filt > 1 else 0.0
            filt_sem_v = float(filt_std_v / np.sqrt(n_pts_filt)) if n_pts_filt > 0 else 0.0
            filt_median_v = float(np.median(speeds_filtered)) if n_pts_filt > 0 else np.nan
            filt_q25 = float(np.percentile(speeds_filtered, 25)) if n_pts_filt > 0 else np.nan
            filt_q75 = float(np.percentile(speeds_filtered, 75)) if n_pts_filt > 0 else np.nan
            filt_iqr = filt_q75 - filt_q25 if n_pts_filt > 0 else np.nan

            # 実験ごとの平均（外れ値除去後）
            exp_means_filt = []
            exp_means_raw = []
            for exp_name, exp_sp in exp_data.items():
                esp = np.array(exp_sp)
                if len(esp) > 0:
                    exp_means_raw.append(float(np.mean(esp)))
                    esp_filt = esp[(esp >= lower_bound) & (esp <= upper_bound)]
                    if len(esp_filt) > 0:
                        exp_means_filt.append(float(np.mean(esp_filt)))

            exp_arr_filt = np.array(exp_means_filt)
            n_exp = len(exp_arr_filt)
            exp_mean_filt_v = float(np.mean(exp_arr_filt)) if n_exp > 0 else np.nan
            exp_std_filt_v = float(np.std(exp_arr_filt, ddof=1)) if n_exp > 1 else 0.0
            exp_sem_filt_v = float(exp_std_filt_v / np.sqrt(n_exp)) if n_exp > 0 else 0.0

            raw_data_dict[b_name] = {
                'speeds_raw': speeds_arr,
                'speeds_filtered': speeds_filtered,
                'upper_bound': upper_bound,
                'lower_bound': lower_bound
            }
        else:
            raw_mean_v = raw_std_v = raw_sem_v = raw_median_v = raw_q25 = raw_q75 = raw_iqr = np.nan
            upper_bound = lower_bound = filt_mean_v = filt_std_v = filt_sem_v = filt_median_v = np.nan
            filt_q25 = filt_q75 = filt_iqr = exp_mean_filt_v = exp_sem_filt_v = exp_std_filt_v = np.nan
            n_pts_filt = n_outliers = 0
            outlier_ratio = 0.0
            n_exp = 0

        records.append({
            "bead_name": b_name,
            "diameter_um": d_um,
            "n_experiments": n_exp,
            "n_tracks": len(track_means),
            "n_points_raw": n_pts_raw,
            "n_points_filtered": n_pts_filt,
            "n_outliers_removed": n_outliers,
            "outlier_ratio_pct": outlier_ratio * 100.0,
            "outlier_cutoff_um_s": upper_bound,
            # 外れ値除去後の統計 (Filtered: Primary metrics)
            "mean_velocity_um_s": filt_mean_v,
            "sem_velocity_um_s": filt_sem_v,
            "std_velocity_um_s": filt_std_v,
            "median_velocity_um_s": filt_median_v,
            "q25_velocity_um_s": filt_q25,
            "q75_velocity_um_s": filt_q75,
            "iqr_velocity_um_s": filt_iqr,
            "exp_mean_velocity_um_s": exp_mean_filt_v,
            "exp_sem_velocity_um_s": exp_sem_filt_v,
            "exp_std_velocity_um_s": exp_std_filt_v,
            # 外れ値除去前の生統計 (Raw)
            "raw_mean_velocity_um_s": raw_mean_v,
            "raw_sem_velocity_um_s": raw_sem_v,
            "raw_std_velocity_um_s": raw_std_v,
            "raw_median_velocity_um_s": raw_median_v,
        })

    return pd.DataFrame(records), raw_data_dict


def load_hmm_and_eff_diff_velocities(
    hmm_summary_path: Optional[Path] = None,
    eff_summary_path: Optional[Path] = None
) -> Optional[pd.DataFrame]:
    """HMM状態別速度および有効拡散サマリーの読み込み"""
    if hmm_summary_path is None:
        hmm_summary_path = CURRENT_DIR / 'figure' / 'hmm_1d' / 'hmm_state_parameters_summary_k2.csv'
    if eff_summary_path is None:
        eff_summary_path = CURRENT_DIR / 'figure' / 'effective_diffusion' / 'effective_diffusion_summary.csv'

    df_hmm_params = None
    if hmm_summary_path.exists():
        try:
            df_hmm = pd.read_csv(hmm_summary_path)
            tumble_rows = df_hmm[df_hmm['state'] == 0].set_index('bead_name')
            run_rows = df_hmm[df_hmm['state'] == 1].set_index('bead_name')

            hmm_records = []
            for b in BEADS_INFO:
                b_name = b["name"]
                d_um = b["diameter_um"]
                row_t = tumble_rows.loc[b_name] if b_name in tumble_rows.index else None
                row_r = run_rows.loc[b_name] if b_name in run_rows.index else None

                v_run = float(row_r['mean_speed_model_um_s']) if row_r is not None else np.nan
                v_run_geom = float(row_r['mean_speed_geom_um_s']) if row_r is not None else np.nan
                v_tumble = float(row_t['mean_speed_model_um_s']) if row_t is not None else np.nan
                v_tumble_geom = float(row_t['mean_speed_geom_um_s']) if row_t is not None else np.nan
                pi_run = float(row_r['stationary_prob']) if row_r is not None else np.nan
                pi_tumble = float(row_t['stationary_prob']) if row_t is not None else np.nan

                if np.isfinite(v_run) and np.isfinite(v_tumble) and np.isfinite(pi_run) and np.isfinite(pi_tumble):
                    v_model = pi_run * v_run + pi_tumble * v_tumble
                    v_model_geom = pi_run * v_run_geom + pi_tumble * v_tumble_geom
                else:
                    v_model = v_model_geom = np.nan

                hmm_records.append({
                    "bead_name": b_name,
                    "diameter_um": d_um,
                    "v_run_model_um_s": v_run,
                    "v_run_geom_um_s": v_run_geom,
                    "v_tumble_model_um_s": v_tumble,
                    "v_tumble_geom_um_s": v_tumble_geom,
                    "pi_run": pi_run,
                    "pi_tumble": pi_tumble,
                    "v_hmm_weighted_um_s": v_model,
                    "v_hmm_weighted_geom_um_s": v_model_geom
                })
            df_hmm_params = pd.DataFrame(hmm_records)
        except Exception as e:
            print(f"Warning: Could not process HMM summary: {e}")

    return df_hmm_params


def plot_velocity_vs_diameter(
    df_summary: pd.DataFrame,
    out_dir: Path
):
    """
    外れ値除去後の平均速度プロットおよび比較プロットを作成・保存
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_custom_style()

    d = df_summary['diameter_um'].values
    v_mean = df_summary['mean_velocity_um_s'].values
    v_sem = df_summary['sem_velocity_um_s'].values
    v_median = df_summary['median_velocity_um_s'].values
    v_exp = df_summary['exp_mean_velocity_um_s'].values
    v_exp_sem = df_summary['exp_sem_velocity_um_s'].values

    # 外れ値除去前のデータ
    v_raw = df_summary['raw_mean_velocity_um_s'].values
    v_raw_sem = df_summary['raw_sem_velocity_um_s'].values

    has_hmm = 'v_run_geom_um_s' in df_summary.columns
    if has_hmm:
        v_run = df_summary['v_run_geom_um_s'].values
        v_tumble = df_summary['v_tumble_geom_um_s'].values
        v_hmm_w = df_summary['v_hmm_weighted_geom_um_s'].values

    # =========================================================================
    # 1. 2パネル標準プロット (外れ値除去後: Linear & Log)
    # =========================================================================
    fig_s, (ax_s_lin, ax_s_log) = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, scale_type in [(ax_s_lin, 'Linear'), (ax_s_log, 'Log')]:
        # 各ビーズデータ点
        for idx, row in df_summary.iterrows():
            b_info = next((b for b in BEADS_INFO if b["name"] == row["bead_name"]), None)
            marker = b_info["marker"] if b_info else "o"
            color = b_info["color"] if b_info else "#1f77b4"
            ax.errorbar(
                row['diameter_um'], row['mean_velocity_um_s'],
                yerr=row['sem_velocity_um_s'],
                fmt=marker, color=color, ecolor=color, elinewidth=2.2,
                capsize=5.5, capthick=1.6, markersize=9.5, zorder=6,
                label=f"{row['bead_name']} ($d = {row['diameter_um']}\,\mu\mathrm{{m}}$)"
            )

        # トレンド線
        ax.plot(
            d, v_mean, color='#1f77b4', linestyle='-', linewidth=2.4, alpha=0.85,
            label=r'Mean $\langle v \rangle$ (Outliers Removed)', zorder=4
        )
        ax.fill_between(
            d, np.maximum(v_mean - v_sem, 1e-4 if scale_type == 'Log' else 0), v_mean + v_sem,
            color='#1f77b4', alpha=0.18, zorder=2, label=r'$\pm 1$ SEM'
        )

        ax.set_xlabel(r'Cargo Particle Diameter $d$ [$\mu\mathrm{m}$]', fontsize=13, fontweight='bold')
        ax.set_ylabel(r'Cargo Mean Velocity $\langle v \rangle$ [$\mu\mathrm{m/s}$]', fontsize=13, fontweight='bold')
        ax.set_title(f'({scale_type} Scale)', fontsize=14, fontweight='bold', pad=8)
        ax.grid(True, which='both', linestyle='--', alpha=0.45)

        if scale_type == 'Linear':
            ax.set_xlim(0, 22)
            ax.set_ylim(0, 0.14)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.02))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
            ax.legend(frameon=True, fontsize=9.0, loc='upper right', framealpha=0.92)
        else:
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlim(0.45, 28)
            ax.set_ylim(0.045, 0.15)
            ax.yaxis.set_major_locator(ticker.FixedLocator([0.05, 0.06, 0.08, 0.10, 0.12, 0.15]))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
            ax.yaxis.set_minor_formatter(ticker.NullFormatter())
            ax.xaxis.set_major_locator(ticker.FixedLocator([0.5, 1, 2, 5, 10, 20]))
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
            ax.legend(frameon=True, fontsize=9.0, loc='lower left', framealpha=0.92)

    fig_s.suptitle(r'Cargo Particle Mean Velocity vs Diameter ($\langle v \rangle$ vs $d$, Outliers Removed)', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()

    simple_svg = out_dir / 'cargo_velocity_vs_diameter_simple_2panel.svg'
    simple_png = out_dir / 'cargo_velocity_vs_diameter_simple_2panel.png'
    fig_s.savefig(simple_svg, bbox_inches='tight')
    fig_s.savefig(simple_png, dpi=300, bbox_inches='tight')
    plt.close(fig_s)
    print(f"  [保存完了] {simple_svg}")
    print(f"  [保存完了] {simple_png}")

    # =========================================================================
    # 2. 標準 Linear 単体プロット
    # =========================================================================
    fig_s_lin, ax_sl = plt.subplots(figsize=(7.5, 5.5))
    for idx, row in df_summary.iterrows():
        b_info = next((b for b in BEADS_INFO if b["name"] == row["bead_name"]), None)
        marker = b_info["marker"] if b_info else "o"
        color = b_info["color"] if b_info else "#1f77b4"
        ax_sl.errorbar(
            row['diameter_um'], row['mean_velocity_um_s'],
            yerr=row['sem_velocity_um_s'],
            fmt=marker, color=color, ecolor=color, elinewidth=2.2,
            capsize=6, capthick=1.8, markersize=10, zorder=6,
            label=f"{row['bead_name']} ($d = {row['diameter_um']}\,\mu\mathrm{{m}}$)"
        )
    ax_sl.plot(d, v_mean, color='#1f77b4', linestyle='-', linewidth=2.6, label=r'Mean $\langle v \rangle$ (Outliers Removed)', zorder=4)
    ax_sl.fill_between(d, np.maximum(v_mean - v_sem, 0), v_mean + v_sem, color='#1f77b4', alpha=0.2, zorder=2, label=r'$\pm 1$ SEM')
    ax_sl.set_xlabel(r'Cargo Particle Diameter $d$ [$\mu\mathrm{m}$]', fontsize=14, fontweight='bold')
    ax_sl.set_ylabel(r'Cargo Mean Velocity $\langle v \rangle$ [$\mu\mathrm{m/s}$]', fontsize=14, fontweight='bold')
    ax_sl.set_title(r'Cargo Mean Velocity vs Diameter (Linear Scale, Outliers Removed)', fontsize=14, fontweight='bold', pad=10)
    ax_sl.set_xlim(0, 22)
    ax_sl.set_ylim(0, 0.14)
    ax_sl.yaxis.set_major_locator(ticker.MultipleLocator(0.02))
    ax_sl.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax_sl.grid(True, linestyle='--', alpha=0.45)
    ax_sl.legend(frameon=True, fontsize=9.5, loc='upper right', framealpha=0.92)

    plt.tight_layout()
    s_lin_svg = out_dir / 'cargo_velocity_vs_diameter_simple_linear.svg'
    s_lin_png = out_dir / 'cargo_velocity_vs_diameter_simple_linear.png'
    fig_s_lin.savefig(s_lin_svg, bbox_inches='tight')
    fig_s_lin.savefig(s_lin_png, dpi=300, bbox_inches='tight')
    plt.close(fig_s_lin)
    print(f"  [保存完了] {s_lin_svg}")
    print(f"  [保存完了] {s_lin_png}")

    # =========================================================================
    # 3. 標準 Log 単体プロット
    # =========================================================================
    fig_s_log, ax_sg = plt.subplots(figsize=(7.5, 5.5))
    for idx, row in df_summary.iterrows():
        b_info = next((b for b in BEADS_INFO if b["name"] == row["bead_name"]), None)
        marker = b_info["marker"] if b_info else "o"
        color = b_info["color"] if b_info else "#1f77b4"
        ax_sg.errorbar(
            row['diameter_um'], row['mean_velocity_um_s'],
            yerr=row['sem_velocity_um_s'],
            fmt=marker, color=color, ecolor=color, elinewidth=2.2,
            capsize=6, capthick=1.8, markersize=10, zorder=6,
            label=f"{row['bead_name']} ($d = {row['diameter_um']}\,\mu\mathrm{{m}}$)"
        )
    ax_sg.plot(d, v_mean, color='#1f77b4', linestyle='-', linewidth=2.6, label=r'Mean $\langle v \rangle$ (Outliers Removed)', zorder=4)
    ax_sg.fill_between(d, np.maximum(v_mean - v_sem, 1e-4), v_mean + v_sem, color='#1f77b4', alpha=0.2, zorder=2, label=r'$\pm 1$ SEM')
    ax_sg.set_xscale('log')
    ax_sg.set_yscale('log')
    ax_sg.set_xlabel(r'Cargo Particle Diameter $d$ [$\mu\mathrm{m}$]', fontsize=14, fontweight='bold')
    ax_sg.set_ylabel(r'Cargo Mean Velocity $\langle v \rangle$ [$\mu\mathrm{m/s}$]', fontsize=14, fontweight='bold')
    ax_sg.set_title(r'Cargo Mean Velocity vs Diameter (Log-Log Scale, Outliers Removed)', fontsize=14, fontweight='bold', pad=10)
    ax_sg.set_xlim(0.45, 28)
    ax_sg.set_ylim(0.045, 0.15)
    ax_sg.yaxis.set_major_locator(ticker.FixedLocator([0.05, 0.06, 0.08, 0.10, 0.12, 0.15]))
    ax_sg.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax_sg.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax_sg.xaxis.set_major_locator(ticker.FixedLocator([0.5, 1, 2, 5, 10, 20]))
    ax_sg.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
    ax_sg.grid(True, which='both', linestyle='--', alpha=0.45)
    ax_sg.legend(frameon=True, fontsize=9.5, loc='lower left', framealpha=0.92)

    plt.tight_layout()
    s_log_svg = out_dir / 'cargo_velocity_vs_diameter_simple_log.svg'
    s_log_png = out_dir / 'cargo_velocity_vs_diameter_simple_log.png'
    fig_s_log.savefig(s_log_svg, bbox_inches='tight')
    fig_s_log.savefig(s_log_png, dpi=300, bbox_inches='tight')
    plt.close(fig_s_log)
    print(f"  [保存完了] {s_log_svg}")
    print(f"  [保存完了] {s_log_png}")

    # =========================================================================
    # 4. 外れ値除去前 vs 除去後 比較プロット (Raw vs Filtered 2-panel)
    # =========================================================================
    fig_cmp, (ax_c_lin, ax_c_log) = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, scale_type in [(ax_c_lin, 'Linear'), (ax_c_log, 'Log')]:
        # 外れ値除去前 (Raw)
        ax.errorbar(
            d, v_raw, yerr=v_raw_sem, fmt='o--', color='#d62728', ecolor='#d62728',
            elinewidth=1.6, capsize=4.5, capthick=1.2, markersize=8,
            linewidth=1.8, alpha=0.75, label=r'Raw Data (with outliers, $\langle v \rangle_{\mathrm{raw}}$)', zorder=3
        )

        # 外れ値除去後 (Filtered)
        ax.errorbar(
            d, v_mean, yerr=v_sem, fmt='s-', color='#1f77b4', ecolor='#1f77b4',
            elinewidth=2.2, capsize=5.5, capthick=1.6, markersize=9,
            linewidth=2.4, label=r'Outliers Removed (IQR-filtered, $\langle v \rangle_{\mathrm{filt}}$)', zorder=5
        )
        ax.fill_between(
            d, np.maximum(v_mean - v_sem, 1e-4 if scale_type == 'Log' else 0), v_mean + v_sem,
            color='#1f77b4', alpha=0.18, zorder=2
        )

        # 中央値 (Median: 本質的に外れ値に頑健)
        ax.plot(
            d, v_median, marker='^', color='#2ca02c', linestyle=':', linewidth=2.0,
            markersize=8, label=r'Median Velocity (Robust metric)', zorder=4
        )

        ax.set_xlabel(r'Cargo Particle Diameter $d$ [$\mu\mathrm{m}$]', fontsize=13, fontweight='bold')
        ax.set_ylabel(r'Cargo Mean Velocity $\langle v \rangle$ [$\mu\mathrm{m/s}$]', fontsize=13, fontweight='bold')
        ax.set_title(f'({scale_type} Scale)', fontsize=14, fontweight='bold', pad=8)
        ax.grid(True, which='both', linestyle='--', alpha=0.45)

        if scale_type == 'Linear':
            ax.set_xlim(0, 22)
            ax.set_ylim(0, 0.30)
            ax.legend(frameon=True, fontsize=9.2, loc='upper right', framealpha=0.92)
        else:
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlim(0.45, 28)
            ax.set_ylim(0.03, 0.35)
            ax.yaxis.set_major_locator(ticker.FixedLocator([0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30]))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
            ax.xaxis.set_major_locator(ticker.FixedLocator([0.5, 1, 2, 5, 10, 20]))
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
            ax.legend(frameon=True, fontsize=9.2, loc='upper left', framealpha=0.92)

    fig_cmp.suptitle(r'Cargo Velocity vs Diameter: Effect of Outlier Removal', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()

    cmp_svg = out_dir / 'cargo_velocity_vs_diameter_outlier_comparison.svg'
    cmp_png = out_dir / 'cargo_velocity_vs_diameter_outlier_comparison.png'
    fig_cmp.savefig(cmp_svg, bbox_inches='tight')
    fig_cmp.savefig(cmp_png, dpi=300, bbox_inches='tight')
    plt.close(fig_cmp)
    print(f"  [保存完了] {cmp_svg}")
    print(f"  [保存完了] {cmp_png}")

    # =========================================================================
    # 5. 詳細 2パネル比較プロット (Detailed: Filtered + Experiment-level + HMM)
    # =========================================================================
    fig_d, (ax_d_lin, ax_d_log) = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, scale_type in [(ax_d_lin, 'Linear'), (ax_d_log, 'Log')]:
        ax.plot(
            d, v_mean, color='#1f77b4', linestyle='-', linewidth=2.5,
            label=r'Mean $\langle v \rangle$ (Outliers Removed)', zorder=4
        )
        ax.fill_between(
            d, np.maximum(v_mean - v_sem, 1e-4 if scale_type == 'Log' else 0), v_mean + v_sem,
            color='#1f77b4', alpha=0.18, zorder=2, label=r'$\pm 1$ SEM'
        )

        ax.errorbar(
            d, v_exp, yerr=v_exp_sem, fmt='s', color='#2ca02c', ecolor='#2ca02c',
            elinewidth=1.6, capsize=4.5, capthick=1.2, markersize=7,
            linestyle='--', linewidth=1.6, label=r'Experiment Mean $\pm \mathrm{SEM}$ (Filtered)', zorder=5
        )

        if has_hmm:
            ax.plot(
                d, v_run, marker='^', color='#d62728', linestyle=':', linewidth=1.8,
                markersize=7.5, label=r'HMM Run State $v_{\mathrm{run}}$', zorder=5
            )
            ax.plot(
                d, v_tumble, marker='v', color='#7f7f7f', linestyle=':', linewidth=1.6,
                markersize=7.0, label=r'HMM Tumble State $v_{\mathrm{tumble}}$', zorder=5
            )
            ax.plot(
                d, v_hmm_w, marker='d', color='#9467bd', linestyle='-.', linewidth=1.8,
                markersize=7.5, label=r'HMM Weighted $\langle v \rangle_{\mathrm{HMM}}$', zorder=5
            )

        for idx, row in df_summary.iterrows():
            b_info = next((b for b in BEADS_INFO if b["name"] == row["bead_name"]), None)
            marker = b_info["marker"] if b_info else "o"
            color = b_info["color"] if b_info else "#1f77b4"
            ax.errorbar(
                row['diameter_um'], row['mean_velocity_um_s'],
                yerr=row['sem_velocity_um_s'],
                fmt=marker, color=color, ecolor=color, elinewidth=2.0,
                capsize=5, capthick=1.5, markersize=9, zorder=7
            )

        ax.set_xlabel(r'Cargo Particle Diameter $d$ [$\mu\mathrm{m}$]', fontsize=13, fontweight='bold')
        ax.set_ylabel(r'Cargo Mean Velocity $\langle v \rangle$ [$\mu\mathrm{m/s}$]', fontsize=13, fontweight='bold')
        ax.set_title(f'({scale_type} Scale)', fontsize=14, fontweight='bold', pad=8)
        ax.grid(True, which='both', linestyle='--', alpha=0.45)

        if scale_type == 'Linear':
            ax.set_xlim(0, 22)
            ax.set_ylim(-0.01, 1.1)
            ax.legend(frameon=True, fontsize=9.2, loc='upper left', framealpha=0.92)
        else:
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlim(0.45, 28)
            ax.set_ylim(0.01, 1.6)
            ax.yaxis.set_major_locator(ticker.FixedLocator([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
            ax.xaxis.set_major_locator(ticker.FixedLocator([0.5, 1, 2, 5, 10, 20]))
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
            ax.legend(frameon=True, fontsize=9.2, loc='lower left', framealpha=0.92)

    fig_d.suptitle(r'Cargo Particle Velocity vs Diameter: Detailed Model Comparison', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()

    det_svg = out_dir / 'cargo_velocity_vs_diameter_detailed_2panel.svg'
    det_png = out_dir / 'cargo_velocity_vs_diameter_detailed_2panel.png'
    fig_d.savefig(det_svg, bbox_inches='tight')
    fig_d.savefig(det_png, dpi=300, bbox_inches='tight')
    plt.close(fig_d)
    print(f"  [保存完了] {det_svg}")
    print(f"  [保存完了] {det_png}")


def main():
    parser = argparse.ArgumentParser(
        description="Cargo particle velocity vs diameter analysis with outlier filtering (Linear and Log scale)."
    )
    parser.add_argument(
        '--root_dir', type=str, default=None,
        help="Root directory of bead tracks."
    )
    parser.add_argument(
        '--out_dir', type=str, default=None,
        help="Output directory for plots and CSV summary (default: figure/velocity)."
    )
    parser.add_argument(
        '--scale', type=float, default=0.11,
        help="Pixel scale in um/pixel (default: 0.11)."
    )
    parser.add_argument(
        '--frame_interval', type=float, default=4.0,
        help="Frame interval in seconds (default: 4.0)."
    )
    parser.add_argument(
        '--iqr_multiplier', type=float, default=1.5,
        help="Tukey IQR multiplier for outlier cutoff (Q3 + mult * IQR, default: 1.5)."
    )

    args = parser.parse_args()

    root_dir = Path(args.root_dir) if args.root_dir is not None else find_default_root()
    out_dir = Path(args.out_dir) if args.out_dir is not None else CURRENT_DIR / 'figure' / 'velocity'

    print(f"--- 貨物粒子の平均速度解析（外れ値除去）を開始します ---")
    print(f"Root dir: {root_dir}")
    print(f"Output dir: {out_dir}")
    print(f"IQR multiplier: {args.iqr_multiplier}")

    # 1. 生データから速度集計（外れ値除去処理を含む）
    df_raw, raw_data_dict = load_raw_velocity_statistics(
        root_dir, scale=args.scale, frame_interval=args.frame_interval, iqr_multiplier=args.iqr_multiplier
    )

    # 2. HMMパラメータの結合
    df_hmm = load_hmm_and_eff_diff_velocities()
    if df_hmm is not None:
        df_summary = pd.merge(df_raw, df_hmm[['bead_name', 'v_run_model_um_s', 'v_run_geom_um_s',
                                               'v_tumble_model_um_s', 'v_tumble_geom_um_s',
                                               'pi_run', 'pi_tumble',
                                               'v_hmm_weighted_um_s', 'v_hmm_weighted_geom_um_s']],
                              on='bead_name', how='left')
    else:
        df_summary = df_raw

    # 3. CSV保存
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / 'cargo_velocity_vs_diameter_summary.csv'
    df_summary.to_csv(summary_csv, index=False)
    print(f"  [保存完了] {summary_csv}")

    # 4. プロット作成
    plot_velocity_vs_diameter(df_summary, out_dir)

    print("\n--- 解析サマリー（外れ値除去後 vs 除去前） ---")
    display_cols = [
        'bead_name', 'diameter_um', 'n_points_filtered', 'outlier_ratio_pct',
        'mean_velocity_um_s', 'sem_velocity_um_s', 'raw_mean_velocity_um_s', 'median_velocity_um_s'
    ]
    print(df_summary[display_cols].to_string(index=False))
    print("\nすべてのグラフおよびサマリーCSVの出力が完了しました。")


if __name__ == "__main__":
    main()
