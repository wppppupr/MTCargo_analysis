#!/usr/bin/env python3
"""
plot_run_velocity.py

0.63 μm, 1.18 μm, 3.37 μm のビーズにおける
Run（能動輸送/走行）状態の平均速度の粒子径依存性を可視化するスクリプトです。
外れ値除去は行わず、生データおよびHMM 2状態モデルのRunパラメータを忠実にプロットします。

出力ファイル:
1. figure/velocity/run_velocity_vs_diameter_2panel.png / .svg (Run速度: Linear & Log 2パネル)
2. figure/velocity/run_velocity_vs_diameter_linear.png / .svg (Run速度: Linear 単体図)
3. figure/velocity/run_velocity_vs_diameter_log.png / .svg (Run速度: Log 単体図)
4. figure/velocity/run_vs_overall_velocity_comparison_2panel.png / .svg (Run速度 vs 全体速度 比較図)
5. figure/velocity/run_velocity_summary.csv (統計サマリーCSV)
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

CURRENT_DIR = Path(__file__).parent.resolve()
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# ビーズ基本情報 (0.6, 1, 3 um)
RUN_BEADS_INFO = [
    {"name": "beads06um", "diameter_um": 0.63, "marker": "^", "label": "0.63 μm", "color": "#1f77b4"},
    {"name": "beads1um",  "diameter_um": 1.18, "marker": "o", "label": "1.18 μm", "color": "#ff7f0e"},
    {"name": "beads3um",  "diameter_um": 3.37, "marker": "d", "label": "3.37 μm", "color": "#2ca02c"},
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
            for b in ['beads1um', 'beads06um', 'beads3um']:
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


def load_data(root_dir: Path, scale: float = 0.11, frame_interval: float = 4.0) -> pd.DataFrame:
    """生軌跡データ（外れ値除去なし）および HMM Run 状態パラメータを統合取得"""
    # 1. HMM パラメータの読み込み
    hmm_path = CURRENT_DIR / 'figure' / 'hmm_1d' / 'hmm_state_parameters_summary_k2.csv'
    if not hmm_path.exists():
        raise FileNotFoundError(f"HMM summary not found: {hmm_path}")

    df_hmm = pd.read_csv(hmm_path)
    df_run = df_hmm[df_hmm['state'] == 1].set_index('bead_name')
    df_tumble = df_hmm[df_hmm['state'] == 0].set_index('bead_name')

    records = []
    for b in RUN_BEADS_INFO:
        b_name = b["name"]
        d_um = b["diameter_um"]

        row_r = df_run.loc[b_name] if b_name in df_run.index else None
        row_t = df_tumble.loc[b_name] if b_name in df_tumble.index else None

        # 生軌跡データから全体の平均速度（外れ値除去なし）を計算
        b_dir = root_dir / b_name
        track_files = sorted(b_dir.glob("*/*/beads_tracks.csv"))
        if not track_files:
            track_files = sorted(b_dir.glob("*/*/*beads_tracks.csv")) + sorted(b_dir.glob("*beads_tracks.csv"))

        all_speeds = []
        for f in track_files:
            try:
                df_t = pd.read_csv(f)
            except Exception:
                continue
            for tid, g in df_t.groupby('particle'):
                if len(g) < 2:
                    continue
                g = g.sort_values('frame')
                dx = np.diff(g['x'].values) * scale
                dy = np.diff(g['y'].values) * scale
                dt = np.diff(g['frame'].values) * frame_interval
                valid = dt > 0
                if not np.any(valid):
                    continue
                sp = np.sqrt(dx[valid]**2 + dy[valid]**2) / dt[valid]
                sp = sp[np.isfinite(sp)]
                all_speeds.extend(sp)

        sp_arr = np.array(all_speeds)
        raw_mean_v = float(np.mean(sp_arr)) if len(sp_arr) > 0 else np.nan
        raw_std_v = float(np.std(sp_arr, ddof=1)) if len(sp_arr) > 1 else np.nan
        raw_sem_v = float(raw_std_v / np.sqrt(len(sp_arr))) if len(sp_arr) > 0 else np.nan
        raw_median_v = float(np.median(sp_arr)) if len(sp_arr) > 0 else np.nan

        # HMM Run 速度
        v_run_geom = float(row_r['mean_speed_geom_um_s']) if row_r is not None else np.nan
        v_run_model = float(row_r['mean_speed_model_um_s']) if row_r is not None else np.nan
        n_run_obs = int(row_r['n_state_obs']) if row_r is not None else 0
        mu_log_v = float(row_r['mean_log_v']) if row_r is not None else np.nan
        sigma_log_v = float(row_r['std_log_v']) if row_r is not None else np.nan
        pi_run = float(row_r['stationary_prob']) if row_r is not None else np.nan

        # 幾何平均・モデル平均のSEM (対数空間からの誤差伝播)
        sem_mu = sigma_log_v / np.sqrt(n_run_obs) if n_run_obs > 0 else 0.0
        v_run_geom_sem = v_run_geom * sem_mu
        v_run_model_sem = v_run_model * sem_mu

        records.append({
            "bead_name": b_name,
            "diameter_um": d_um,
            "label": b["label"],
            "v_run_geom_um_s": v_run_geom,
            "v_run_geom_sem_um_s": v_run_geom_sem,
            "v_run_model_um_s": v_run_model,
            "v_run_model_sem_um_s": v_run_model_sem,
            "n_run_obs": n_run_obs,
            "pi_run": pi_run,
            "raw_mean_velocity_um_s": raw_mean_v,
            "raw_sem_velocity_um_s": raw_sem_v,
            "raw_median_velocity_um_s": raw_median_v,
            "n_total_points": len(sp_arr)
        })

    return pd.DataFrame(records)


def plot_run_velocity(df: pd.DataFrame, out_dir: Path):
    """Run速度のプロットを作成・保存 (ln(y) = -1/(3R_0) x + B フィッティング, xlim=(0, 20), Model Mean 削除版)"""
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_custom_style()

    d = df['diameter_um'].values
    v_run_g = df['v_run_geom_um_s'].values
    v_run_g_sem = df['v_run_geom_sem_um_s'].values
    v_raw = df['raw_mean_velocity_um_s'].values
    v_raw_sem = df['raw_sem_velocity_um_s'].values

    # =========================================================================
    # フィッティング: ln(y) = -1/(3R_0) * x + B  => y = exp(B) * exp(-x / (3R_0))
    # =========================================================================
    p = np.polyfit(d, np.log(v_run_g), 1)
    slope = p[0]       # slope = -1 / (2*R0)
    B = p[1]           # intercept = B
    two_R0 = -1.0 / slope
    R0 = two_R0 / 3.0
    v0 = np.exp(B)

    ln_y_pred = slope * d + B
    ss_tot = np.sum((np.log(v_run_g) - np.mean(np.log(v_run_g)))**2)
    ss_res = np.sum((np.log(v_run_g) - ln_y_pred)**2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    print(f"\n[フィッティング結果: ln(y) = - 1/(3*xi) * (2*Rc) + B]")
    print(f"  xi = {R0:.4f} μm  (3*xi = {two_R0:.4f} μm)")
    print(f"  B = {B:.4f}")
    print(f"  v0 = exp(B) = {v0:.4f} μm/s")
    print(f"  R^2 = {r2:.4f}")
    print(f"  式: ln(v_run) = - 1/(3 * {R0:.3f}) * (2Rc) + ({B:.4f})  =>  v_run(Rc) = {v0:.4f} * exp(-4 Rc / (3 * {R0:.3f}))\n")

    x_fit = np.linspace(0.0, 20.0, 300)
    y_fit = v0 * np.exp(-x_fit / two_R0)

    fit_label = (
        r'Fit: $\ln(v_{\mathrm{run}}) = -\frac{4 R_c}{3\xi} + B$' + '\n'
        rf'  $\xi = {R0:.2f}\,\mu\mathrm{{m}}$' + '\n'
        rf'  $B = {B:.4f}\ (v_0 = {v0:.3f}\,\mu\mathrm{{m/s}})$' + '\n'
        rf'  $R^2 = {r2:.3f}$'
    )

    # =========================================================================
    # 1. Run 速度 2パネル比較 (Linear & Log, xlim=(0, 20))
    # =========================================================================
    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(15, 6))

    for ax, scale_type in [(ax_lin, 'Linear'), (ax_log, 'Log')]:
        # 各ビーズ実測データ点
        for idx, row in df.iterrows():
            b_info = next((b for b in RUN_BEADS_INFO if b["name"] == row["bead_name"]), None)
            marker = b_info["marker"] if b_info else "o"
            color = b_info["color"] if b_info else "#d62728"
            ax.errorbar(
                row['diameter_um'], row['v_run_geom_um_s'],
                yerr=row['v_run_geom_sem_um_s'],
                fmt=marker, color=color, ecolor=color, elinewidth=2.4,
                capsize=6, capthick=1.8, markersize=11, zorder=7,
                label=f"{row['label']} ($v_{{\mathrm{{run}}}} = {row['v_run_geom_um_s']:.3f}\,\mu\mathrm{{m/s}}$)"
            )

        # 実測値同士を繋ぐ補助線（破線）
        ax.plot(d, v_run_g, color='#d62728', linestyle=':', linewidth=1.8, alpha=0.7, zorder=4)

        # フィッティング曲線 (0 ~ 20 um)
        ax.plot(
            x_fit, y_fit, color='#d62728', linestyle='-', linewidth=2.8,
            label=fit_label, zorder=5
        )

        ax.set_xlabel(r'Cargo Diameter $2R_c$ [$\mu\mathrm{m}$]', fontsize=13, fontweight='bold')
        ax.set_ylabel(r'Run Velocity $v_{\mathrm{run}}$ [$\mu\mathrm{m/s}$]', fontsize=13, fontweight='bold')
        ax.set_title(f'({scale_type} Scale, $x \in [0, 20]\,\mu\mathrm{{m}}$)', fontsize=14, fontweight='bold', pad=8)
        ax.set_xlim(0, 20.0)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2.0))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1.0))
        ax.grid(True, which='both', linestyle='--', alpha=0.45)

        if scale_type == 'Linear':
            ax.set_ylim(0, 0.25)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
            ax.legend(frameon=True, fontsize=9.2, loc='upper right', framealpha=0.92)
        else:
            ax.set_yscale('log')
            ax.set_ylim(0.01, 0.30)
            ax.yaxis.set_major_locator(ticker.FixedLocator([0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25]))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
            ax.yaxis.set_minor_formatter(ticker.NullFormatter())
            ax.legend(frameon=True, fontsize=9.2, loc='upper right', framealpha=0.92)

    fig.suptitle(r'Cargo Run Velocity vs Diameter with Fit $\ln(y) = -\frac{4 R_c}{3\xi} + B$ (0.6, 1, 3 $\mu\mathrm{m}$)', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()

    comp_svg = out_dir / 'run_velocity_vs_diameter_2panel.svg'
    comp_png = out_dir / 'run_velocity_vs_diameter_2panel.png'
    fig.savefig(comp_svg, bbox_inches='tight')
    fig.savefig(comp_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [保存完了] {comp_svg}")
    print(f"  [保存完了] {comp_png}")

    # =========================================================================
    # 2. Run 速度 Log 単体図 (フィッティング強調, xlim=(0, 20))
    # =========================================================================
    fig_log, ax_g = plt.subplots(figsize=(8.5, 6.0))
    for idx, row in df.iterrows():
        b_info = next((b for b in RUN_BEADS_INFO if b["name"] == row["bead_name"]), None)
        marker = b_info["marker"] if b_info else "o"
        color = b_info["color"] if b_info else "#d62728"
        ax_g.errorbar(
            row['diameter_um'], row['v_run_geom_um_s'],
            yerr=row['v_run_geom_sem_um_s'],
            fmt=marker, color=color, ecolor=color, elinewidth=2.5,
            capsize=6, capthick=2.0, markersize=12, zorder=8,
            label=f"{row['label']}: $v_{{\mathrm{{run}}}} = {row['v_run_geom_um_s']:.3f} \pm {row['v_run_geom_sem_um_s']:.3f}\,\mu\mathrm{{m/s}}$"
        )

    # 実測値接続破線
    ax_g.plot(d, v_run_g, color='#d62728', linestyle=':', linewidth=2.0, alpha=0.7, zorder=4)

    # フィッティング直線 (Logスケール上では直線)
    ax_g.plot(
        x_fit, y_fit, color='#d62728', linestyle='-', linewidth=2.8,
        label=fit_label, zorder=5
    )

    ax_g.set_yscale('log')
    ax_g.set_xlim(0, 20.0)
    ax_g.set_ylim(0.01, 0.30)
    ax_g.set_xlabel(r'Cargo Diameter $2R_c$ [$\mu\mathrm{m}$]', fontsize=14, fontweight='bold')
    ax_g.set_ylabel(r'Run Mean Velocity $v_{\mathrm{run}}$ [$\mu\mathrm{m/s}$]', fontsize=14, fontweight='bold')
    ax_g.set_title(r'Cargo Run Velocity vs Diameter (Log Scale with Fit $\ln(y) = -\frac{4 R_c}{3\xi} + B$)', fontsize=13, fontweight='bold', pad=10)

    ax_g.xaxis.set_major_locator(ticker.MultipleLocator(2.0))
    ax_g.xaxis.set_minor_locator(ticker.MultipleLocator(1.0))
    ax_g.yaxis.set_major_locator(ticker.FixedLocator([0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25]))
    ax_g.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax_g.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax_g.grid(True, which='both', linestyle='--', alpha=0.45)
    ax_g.legend(frameon=True, fontsize=10.0, loc='upper right', framealpha=0.92)

    # 詳細なフィッティング情報テキストボックス
    fit_info_text = (
        r"$\mathbf{Fitting\ Model:}$" + "\n"
        r"$\ln(v_{\mathrm{run}}) = -\frac{4 R_c}{3\xi} + B$" + "\n"
        r"$v_{\mathrm{run}}(R_c) = v_0 \exp\left(-\frac{4 R_c}{3\xi}\right)$" + "\n"
        rf"$\xi = {R0:.3f}\ \mu\mathrm{{m}}$" + "\n"
        rf"$B = {B:.4f}\quad (v_0 = \mathrm{{e}}^B = {v0:.4f}\ \mu\mathrm{{m/s}})$" + "\n"
        rf"$R^2 = {r2:.4f}$"
    )
    ax_g.text(
        0.04, 0.06, fit_info_text, transform=ax_g.transAxes,
        fontsize=10.0, verticalalignment='bottom',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='white', edgecolor='#d62728', alpha=0.92, linewidth=1.5)
    )

    plt.tight_layout()
    log_svg = out_dir / 'run_velocity_vs_diameter_log.svg'
    log_png = out_dir / 'run_velocity_vs_diameter_log.png'
    fig_log.savefig(log_svg, bbox_inches='tight')
    fig_log.savefig(log_png, dpi=300, bbox_inches='tight')
    plt.close(fig_log)
    print(f"  [保存完了] {log_svg}")
    print(f"  [保存完了] {log_png}")

    # =========================================================================
    # 3. Run 速度 Linear 単体図 (xlim=(0, 20))
    # =========================================================================
    fig_lin, ax_l = plt.subplots(figsize=(8.5, 6.0))
    for idx, row in df.iterrows():
        b_info = next((b for b in RUN_BEADS_INFO if b["name"] == row["bead_name"]), None)
        marker = b_info["marker"] if b_info else "o"
        color = b_info["color"] if b_info else "#d62728"
        ax_l.errorbar(
            row['diameter_um'], row['v_run_geom_um_s'],
            yerr=row['v_run_geom_sem_um_s'],
            fmt=marker, color=color, ecolor=color, elinewidth=2.5,
            capsize=6, capthick=2.0, markersize=12, zorder=8,
            label=f"{row['label']}: $v_{{\mathrm{{run}}}} = {row['v_run_geom_um_s']:.3f} \pm {row['v_run_geom_sem_um_s']:.3f}\,\mu\mathrm{{m/s}}$"
        )
    ax_l.plot(d, v_run_g, color='#d62728', linestyle=':', linewidth=2.0, alpha=0.7, zorder=4)
    ax_l.plot(x_fit, y_fit, color='#d62728', linestyle='-', linewidth=2.8, label=fit_label, zorder=5)

    ax_l.set_xlabel(r'Cargo Diameter $2R_c$ [$\mu\mathrm{m}$]', fontsize=14, fontweight='bold')
    ax_l.set_ylabel(r'Run Mean Velocity $v_{\mathrm{run}}$ [$\mu\mathrm{m/s}$]', fontsize=14, fontweight='bold')
    ax_l.set_title(r'Cargo Run Velocity vs Diameter (Linear Scale with Fit $y = v_0 \mathrm{e}^{-4 R_c / (3\xi)}$)', fontsize=13, fontweight='bold', pad=10)
    ax_l.set_xlim(0, 20.0)
    ax_l.set_ylim(0, 0.25)
    ax_l.xaxis.set_major_locator(ticker.MultipleLocator(2.0))
    ax_l.xaxis.set_minor_locator(ticker.MultipleLocator(1.0))
    ax_l.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax_l.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
    ax_l.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax_l.grid(True, linestyle='--', alpha=0.45)
    ax_l.legend(frameon=True, fontsize=10.0, loc='upper right', framealpha=0.92)

    plt.tight_layout()
    lin_svg = out_dir / 'run_velocity_vs_diameter_linear.svg'
    lin_png = out_dir / 'run_velocity_vs_diameter_linear.png'
    fig_lin.savefig(lin_svg, bbox_inches='tight')
    fig_lin.savefig(lin_png, dpi=300, bbox_inches='tight')
    plt.close(fig_lin)
    print(f"  [保存完了] {lin_svg}")
    print(f"  [保存完了] {lin_png}")

    # =========================================================================
    # 4. Run速度 vs 全体速度 比較 2パネル (xlim=(0, 20))
    # =========================================================================
    fig_comp, (ax_cp_lin, ax_cp_log) = plt.subplots(1, 2, figsize=(15, 6))

    for ax, scale_type in [(ax_cp_lin, 'Linear'), (ax_cp_log, 'Log')]:
        # Run 幾何平均実測点
        ax.errorbar(
            d, v_run_g, yerr=v_run_g_sem, fmt='^-', color='#d62728', ecolor='#d62728',
            elinewidth=2.2, capsize=5.5, capthick=1.6, markersize=9, linewidth=1.5,
            label=r'Run State Mean $v_{\mathrm{run}}$ (Geometric)', zorder=6
        )
        # フィット線
        ax.plot(x_fit, y_fit, color='#d62728', linestyle='--', linewidth=2.2, alpha=0.85,
                label=rf'Run Fit: $\ln(v) = -\frac{{4 R_c}}{{3\xi}} + B$ ($\xi={R0:.2f}\,\mu\mathrm{{m}}$)', zorder=5)

        # 全体平均（外れ値除去なし）
        ax.errorbar(
            d, v_raw, yerr=v_raw_sem, fmt='o-.', color='#1f77b4', ecolor='#1f77b4',
            elinewidth=2.0, capsize=5.0, capthick=1.5, markersize=8.5, linewidth=2.0,
            label=r'Overall Mean $\langle v \rangle_{\mathrm{raw}}$ (All steps, with outliers)', zorder=4
        )

        ax.set_xlabel(r'Cargo Diameter $2R_c$ [$\mu\mathrm{m}$]', fontsize=13, fontweight='bold')
        ax.set_ylabel(r'Velocity [$\mu\mathrm{m/s}$]', fontsize=13, fontweight='bold')
        ax.set_title(f'({scale_type} Scale, $x \in [0, 20]\,\mu\mathrm{{m}}$)', fontsize=14, fontweight='bold', pad=8)
        ax.set_xlim(0, 20.0)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2.0))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1.0))
        ax.grid(True, which='both', linestyle='--', alpha=0.45)

        if scale_type == 'Linear':
            ax.set_ylim(0, 0.25)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
            ax.legend(frameon=True, fontsize=9.2, loc='upper right', framealpha=0.92)
        else:
            ax.set_yscale('log')
            ax.set_ylim(0.01, 0.30)
            ax.yaxis.set_major_locator(ticker.FixedLocator([0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25]))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
            ax.yaxis.set_minor_formatter(ticker.NullFormatter())
            ax.legend(frameon=True, fontsize=9.2, loc='upper right', framealpha=0.92)

    fig_comp.suptitle(r'Comparison: Run State Velocity vs Overall Velocity ($x \in [0, 20]\,\mu\mathrm{m}$)', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()

    comp_all_svg = out_dir / 'run_vs_overall_velocity_comparison_2panel.svg'
    comp_all_png = out_dir / 'run_vs_overall_velocity_comparison_2panel.png'
    fig_comp.savefig(comp_all_svg, bbox_inches='tight')
    fig_comp.savefig(comp_all_png, dpi=300, bbox_inches='tight')
    plt.close(fig_comp)
    print(f"  [保存完了] {comp_all_svg}")
    print(f"  [保存完了] {comp_all_png}")


def main():
    parser = argparse.ArgumentParser(
        description="Cargo Run state velocity vs diameter (0.6, 1, 3 um) analysis and plotting (Linear and Log scale)."
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

    args = parser.parse_args()

    root_dir = Path(args.root_dir) if args.root_dir is not None else find_default_root()
    out_dir = Path(args.out_dir) if args.out_dir is not None else root_dir / 'figure' / 'velocity'

    print("--- 0.6, 1, 3 μm における Run 状態平均速度プロット（外れ値含む）を開始します ---")
    print(f"Root dir: {root_dir}")
    print(f"Output dir: {out_dir}")

    df = load_data(root_dir, scale=args.scale, frame_interval=args.frame_interval)

    # CSV保存
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'run_velocity_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f"  [保存完了] {csv_path}")

    # プロット作成
    plot_run_velocity(df, out_dir)

    print("\n--- 解析サマリー (0.6, 1, 3 μm) ---")
    display_cols = ['bead_name', 'diameter_um', 'v_run_geom_um_s', 'v_run_geom_sem_um_s', 'v_run_model_um_s', 'raw_mean_velocity_um_s', 'n_run_obs']
    print(df[display_cols].to_string(index=False))
    print("\nすべてのグラフおよびサマリーCSVの出力が完了しました。")


if __name__ == "__main__":
    main()
