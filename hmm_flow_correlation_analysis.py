"""
hmm_flow_correlation_analysis.py

微小管のアクティブオプティカルフロー（GFP_flows.h5 / angular_correlation_w.zarr / angular_correlation_bg.zarr）の空間配向相関を、
貨物微粒子の1次元対数速力 Gaussian HMM 推定状態（Run / Tumble）別に分解して一括解析・可視化・定量化するスクリプトです。

全ビーズサイズ（0.63μm, 1.18μm, 3.37μm, 5.0μm, 7.24μm, 20μm）において、
1. 各粒子径でのモード別微小管フロー空間配向相関 C_flow(r) 曲線 (Run, Tumble, All, Background) 6パネルプロット
2. 粒子径 vs モード別フロー相関関数の比較プロット (4パネル)
3. 粒子径 vs 微小管フロー相関長 xi_flow (Correlation Length) の定量プロット (Run vs Tumble vs BG)
4. 粒子径 vs ビーズ-フロー相互作用相関 C_bead(r) の比較プロット (Run vs Tumble)
5. 大局的ネマチック主軸分解 (平行 // vs 垂直 perp) の異方性プロット
6. 統計サマリー CSV (全モード曲線データ & フィッティング結果) の出力
を行います。
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# 親ディレクトリのパス設定
current_dir = Path(__file__).parent.resolve()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from libs import hmm_cargo as hc
from libs import hmm_flow_correlation as hfc

# スタイルの適用
style_path = current_dir / 'libs' / 'my_style.mplstyle'
if style_path.exists():
    try:
        plt.style.use(str(style_path))
        style_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    except Exception:
        style_colors = ['#882255', '#CC6677', '#DDCC77', '#999933', '#117733', '#44AA99']
else:
    style_colors = ['#882255', '#CC6677', '#DDCC77', '#999933', '#117733', '#44AA99']

# ビーズ条件設定
BEADS_INFO = [
    {"name": "beads06um", "diameter_um": 0.63, "marker": "^", "color": style_colors[0]},
    {"name": "beads1um",  "diameter_um": 1.18, "marker": "o", "color": style_colors[1]},
    {"name": "beads3um",  "diameter_um": 3.37, "marker": "d", "color": style_colors[2]},
    {"name": "beads5um",  "diameter_um": 5.00, "marker": "p", "color": style_colors[3]},
    {"name": "beads7um",  "diameter_um": 7.24, "marker": "h", "color": style_colors[4]},
    {"name": "beads20um", "diameter_um": 20.0, "marker": "s", "color": style_colors[5]},
]

POSSIBLE_ROOTS = [
    Path('/Volumes/data-1/Sasaki/MTsingleBeads'),
    Path('/Volumes/data-1/sasaki/MTsingleBeads'),
    Path('/Volumes/data/Sasaki/MTsingleBeads'),
    Path('/Volumes/data/sasaki/MTsingleBeads'),
    Path('/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads'),
]


def find_default_root() -> Path:
    for r in POSSIBLE_ROOTS:
        if r.exists():
            for b in ['beads1um', 'beads06um', 'beads3um', 'beads5um', 'beads7um', 'beads20um']:
                if (r / b).exists() and len(list((r / b).glob('*/*beads_tracks.csv'))) > 0:
                    return r
    for r in POSSIBLE_ROOTS:
        if r.exists():
            return r
    return POSSIBLE_ROOTS[0]


def find_experiment_dirs(root_dir: Path, bead_name: str) -> List[Path]:
    base = Path(root_dir) / bead_name
    if not base.exists():
        return []
    exp_dirs = []
    for p in sorted(base.glob("*/*")):
        if p.is_dir() and (p / "beads_tracks.csv").exists() and (p / "angular_correlation_w.zarr").exists():
            exp_dirs.append(p)
    if not exp_dirs:
        for p in sorted(base.glob("*")):
            if p.is_dir() and (p / "beads_tracks.csv").exists() and (p / "angular_correlation_w.zarr").exists():
                exp_dirs.append(p)
    return exp_dirs


def safe_save_csv(df: pd.DataFrame, target_path: Path, max_retries: int = 5):
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
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
            time.sleep(0.5)


# =========================================================================
# 可視化関数群
# =========================================================================

def plot_flow_correlation_6panel(
    results_by_bead: Dict[str, dict],
    beads_info: List[dict],
    output_path: Path,
    max_dist: float = 60.0,
):
    """
    全ビーズサイズ（6パネル）におけるモード別微小管フロー空間相関 C_flow(r) vs r をプロットする。
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    axes = axes.flatten()

    for idx, binfo in enumerate(beads_info):
        ax = axes[idx]
        bname = binfo['name']
        dia = binfo['diameter_um']

        if bname not in results_by_bead or results_by_bead[bname]['df_curves'].empty:
            ax.set_title(f"$d = {dia:.2f}\\,\\mu\\mathrm{{m}}$ (No data)", fontsize=12)
            ax.axis('off')
            continue

        res = results_by_bead[bname]
        df_curves = res['df_curves']

        title_str = f"$d = {dia:.2f}\\,\\mu\\mathrm{{m}}$"
        hfc.plot_flow_correlations_single_axis(
            df_curves,
            ax=ax,
            title=title_str,
            fit_curves=True,
            show_legend=(idx == 0),
        )

        ax.set_xlim(0, max_dist)
        if idx < 3:
            ax.set_xlabel("")
        if idx % 3 != 0:
            ax.set_ylabel("")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved 6-panel flow spatial correlation plot: {output_path}")


def plot_flow_mode_comparison_4panel(
    results_by_bead: Dict[str, dict],
    beads_info: List[dict],
    output_path: Path,
    max_dist: float = 60.0,
):
    """
    4つのモードカテゴリ（Runフロー, Tumbleフロー, BGフロー, ビーズ-フロー相関(Run)）ごとに
    全粒子径の C(r) を比較する4パネルプロット。
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()

    categories = [
        ('run', 'Flow Around Run Particles (Active Mode)'),
        ('tumble', 'Flow Around Tumble Particles (Paused Mode)'),
        ('bg', 'Background Optical Flow (Bulk Field)'),
        ('run_bead', 'Bead-Flow Interaction (Run Beads Movement vs Flow)'),
    ]

    for c_idx, (mode_key, title) in enumerate(categories):
        ax = axes[c_idx]

        for binfo in beads_info:
            bname = binfo['name']
            dia = binfo['diameter_um']
            color = binfo['color']
            marker = binfo['marker']

            if bname not in results_by_bead:
                continue

            df_c = results_by_bead[bname]['df_curves']
            if df_c.empty:
                continue

            sub = df_c[df_c['mode'] == mode_key]
            if sub.empty:
                continue

            r = sub['distance_um'].to_numpy()
            c = sub['mean_correlation'].to_numpy()
            sem = sub['sem_correlation'].to_numpy()

            ax.errorbar(
                r, c, yerr=sem,
                label=f"{dia:.2f} $\\mu$m",
                color=color,
                fmt=marker,
                markersize=4,
                linestyle='-',
                linewidth=1.5,
                capsize=2,
                alpha=0.85,
            )

        ax.axhline(0.0, color='gray', linestyle='--', linewidth=1.0, alpha=0.6)
        ax.set_xlabel(r"Distance $r$ [$\mu\mathrm{m}$]", fontsize=11)
        ax.set_ylabel(r"Angular Correlation $C(r)$", fontsize=11)
        ax.set_xlim(0, max_dist)
        ax.set_ylim(-0.2, 1.05)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_title(title, fontsize=12, fontweight='bold')
        if c_idx == 0:
            ax.legend(fontsize=9, loc='upper right', framealpha=0.9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved 4-panel flow mode comparison plot: {output_path}")


def plot_flow_correlation_length_summary(
    df_summary: pd.DataFrame,
    output_path: Path,
):
    """
    粒子径 vs 微小管フロー配向相関長 xi_flow (Correlation Length) の比較プロット。
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    target_modes = [
        ('run', 'Flow (Run Vicinity)', '#1b9e77', 'o'),
        ('tumble', 'Flow (Tumble Vicinity)', '#d95f02', 'd'),
        ('all', 'Flow (All Particles)', '#222222', 's'),
        ('bg', 'Flow (Background Bulk)', '#7570b3', '^'),
    ]

    for mode_key, label, col, m in target_modes:
        sub = df_summary[df_summary['mode'] == mode_key].dropna(subset=['xi_um'])
        if sub.empty:
            continue

        dia = sub['diameter_um'].to_numpy()
        xi = sub['xi_um'].to_numpy()
        xi_err = sub['xi_err_um'].to_numpy()

        ax.errorbar(
            dia, xi, yerr=xi_err,
            label=label,
            color=col,
            fmt=m,
            markersize=7,
            linewidth=2.0,
            capsize=4,
            alpha=0.9,
        )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r"Cargo Bead Diameter $d$ [$\mu\mathrm{m}$]", fontsize=12)
    ax.set_ylabel(r"Flow Orientational Correlation Length $\xi_{\mathrm{flow}}$ [$\mu\mathrm{m}$]", fontsize=12)
    ax.set_title("MT Optical Flow Correlation Length vs Cargo Diameter", fontsize=13, fontweight='bold')
    ax.grid(True, which="both", linestyle='--', alpha=0.4)
    ax.legend(fontsize=10, loc='best')

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved flow correlation length summary plot: {output_path}")


def plot_bead_flow_correlation_comparison(
    results_by_bead: Dict[str, dict],
    beads_info: List[dict],
    output_path: Path,
    max_dist: float = 60.0,
):
    """
    粒子径ごとのビーズ-フロー相互作用相関 C_bead(r) (Run vs Tumble) のプロット。
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax_idx, (mode_key, mode_title, col_state) in enumerate([
        ('run_bead', 'Run Beads: Movement Direction vs MT Flow', '#1b9e77'),
        ('tumble_bead', 'Tumble Beads: Movement Direction vs MT Flow', '#d95f02'),
    ]):
        ax = axes[ax_idx]

        for binfo in beads_info:
            bname = binfo['name']
            dia = binfo['diameter_um']
            color = binfo['color']
            marker = binfo['marker']

            if bname not in results_by_bead:
                continue

            df_c = results_by_bead[bname]['df_curves']
            if df_c.empty:
                continue

            sub = df_c[df_c['mode'] == mode_key]
            if sub.empty:
                continue

            r = sub['distance_um'].to_numpy()
            c = sub['mean_correlation'].to_numpy()
            sem = sub['sem_correlation'].to_numpy()

            ax.errorbar(
                r, c, yerr=sem,
                label=f"{dia:.2f} $\\mu$m",
                color=color,
                fmt=marker,
                markersize=4,
                linestyle='-',
                linewidth=1.5,
                capsize=2,
                alpha=0.85,
            )

        ax.axhline(0.0, color='gray', linestyle='--', linewidth=1.0, alpha=0.6)
        ax.set_xlabel(r"Distance $r$ from Bead Center [$\mu\mathrm{m}$]", fontsize=11)
        ax.set_ylabel(r"Bead-Flow Correlation $C_{\mathrm{bead}}(r) = \langle \hat{\mathbf{v}}_{\mathrm{bead}} \cdot \hat{\mathbf{u}}_{\mathrm{flow}}(r) \rangle$", fontsize=11)
        ax.set_xlim(0, max_dist)
        ax.set_ylim(-0.2, 1.05)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_title(mode_title, fontsize=12, fontweight='bold')
        if ax_idx == 0:
            ax.legend(fontsize=9, loc='upper right', framealpha=0.9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved bead-flow correlation plot: {output_path}")


def plot_nematic_anisotropy_summary(
    results_by_bead: Dict[str, dict],
    beads_info: List[dict],
    output_path: Path,
    max_dist: float = 60.0,
):
    """
    大局的ネマチック主軸に平行な相関 (Parallel //) vs 垂直な相関 (Perpendicular perp) の異方性プロット。
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    axes = axes.flatten()

    for idx, binfo in enumerate(beads_info):
        ax = axes[idx]
        bname = binfo['name']
        dia = binfo['diameter_um']

        if bname not in results_by_bead:
            ax.set_title(f"$d = {dia:.2f}\\,\\mu\\mathrm{{m}}$ (No data)", fontsize=12)
            ax.axis('off')
            continue

        df_c = results_by_bead[bname]['df_curves']
        if df_c.empty:
            ax.set_title(f"$d = {dia:.2f}\\,\\mu\\mathrm{{m}}$ (No data)", fontsize=12)
            ax.axis('off')
            continue

        sub_par = df_c[df_c['mode'] == 'run_par']
        sub_perp = df_c[df_c['mode'] == 'run_perp']
        sub_tot = df_c[df_c['mode'] == 'run']

        if not sub_par.empty:
            ax.errorbar(
                sub_par['distance_um'], sub_par['mean_correlation'], yerr=sub_par['sem_correlation'],
                label=r"Run Flow ($\parallel$ Nematic)", color='#1b9e77', fmt='o', markersize=3.5, linestyle='-', capsize=2, alpha=0.85
            )
        if not sub_perp.empty:
            ax.errorbar(
                sub_perp['distance_um'], sub_perp['mean_correlation'], yerr=sub_perp['sem_correlation'],
                label=r"Run Flow ($\perp$ Nematic)", color='#e7298a', fmt='^', markersize=3.5, linestyle='--', capsize=2, alpha=0.85
            )
        if not sub_tot.empty:
            ax.plot(
                sub_tot['distance_um'], sub_tot['mean_correlation'],
                label="Run Flow (Total)", color='#333333', linestyle=':', lw=1.5, alpha=0.7
            )

        ax.axhline(0.0, color='gray', linestyle='--', linewidth=1.0, alpha=0.6)
        ax.set_xlim(0, max_dist)
        ax.set_ylim(-0.2, 1.05)
        ax.set_title(f"$d = {dia:.2f}\\,\\mu\\mathrm{{m}}$ (Run Flow Anisotropy)", fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.4)
        if idx >= 3:
            ax.set_xlabel(r"Distance $r$ [$\mu\mathrm{m}$]", fontsize=11)
        if idx % 3 == 0:
            ax.set_ylabel(r"Flow Correlation $C(r)$", fontsize=11)
        if idx == 0:
            ax.legend(fontsize=8.5, loc='upper right', framealpha=0.9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved nematic anisotropy summary plot: {output_path}")


# =========================================================================
# メイン処理
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="HMM Mode-Dependent Optical Flow Angular Spatial Correlation Analysis."
    )
    parser.add_argument('--root_dir', type=str, default=None, help="Root directory containing beads data.")
    parser.add_argument('--output_dir', type=str, default='figures/hmm_flow_correlation', help="Output directory for figures & CSVs.")
    parser.add_argument('--beads', type=str, default='all', help="Target bead condition ('all' or 'beads1um' etc.)")
    parser.add_argument('--tau', type=int, default=1, help="Lag time step for velocity calculation.")
    parser.add_argument('--scale', type=float, default=0.11, help="Spatial scale (um/pixel).")
    parser.add_argument('--frame_interval', type=float, default=4.0, help="Time interval between frames (s).")
    parser.add_argument('--epsilon', type=float, default=1e-3, help="Epsilon for log-speed observation ln(v + eps).")
    parser.add_argument('--max_dist', type=float, default=60.0, help="Maximum distance in um for C(r) plotting.")
    args = parser.parse_args()

    root_dir = Path(args.root_dir) if args.root_dir else find_default_root()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print(" HMM Mode-Dependent MT Optical Flow Angular Correlation Analysis")
    print("=================================================================")
    print(f"Data Root Directory: {root_dir}")
    print(f"Output Directory:    {output_dir}")
    print(f"Lag time tau:        {args.tau} ({args.tau * args.frame_interval:.1f} s)")
    print(f"Scale:               {args.scale} um/pixel")
    print("=================================================================\n")

    if args.beads == "all":
        target_bead_infos = BEADS_INFO
    else:
        target_bead_infos = [b for b in BEADS_INFO if b['name'] == args.beads]
        if not target_bead_infos:
            print(f"[ERROR] Unknown bead name '{args.beads}'. Available: {[b['name'] for b in BEADS_INFO]}")
            return

    results_by_bead = {}
    all_binned_records = []
    all_summary_records = []

    for binfo in target_bead_infos:
        bname = binfo['name']
        dia = binfo['diameter_um']
        print(f"--- Processing {bname} (diameter: {dia:.2f} um) ---")

        edirs = find_experiment_dirs(root_dir, bname)
        if not edirs:
            print(f"[WARNING] No experiment directories with angular_correlation_w.zarr found for {bname} in {root_dir}")
            continue

        print(f"  Found {len(edirs)} experiment directories with computed correlation Zarrs.")

        # 1. 軌跡データを集約して HMM モデルを学習
        all_tracks = []
        p_offset = 0
        for edir in edirs:
            t_csv = edir / "beads_tracks.csv"
            df_t = pd.read_csv(t_csv)
            df_t['particle'] = df_t['particle'] + p_offset
            p_offset += int(df_t['particle'].max()) + 1
            all_tracks.append(df_t)

        df_all_tracks = pd.concat(all_tracks, ignore_index=True)
        X, lengths, _ = hc.extract_hmm_features(
            df_all_tracks,
            tau=args.tau,
            scale=args.scale,
            frame_interval=args.frame_interval,
            epsilon=args.epsilon,
        )

        if len(X) < 20:
            print(f"[WARNING] Insufficient data points ({len(X)}) for {bname}. Skipping.")
            continue

        hmm_model = hc.CargoGaussianHMM(
            n_components=2,
            covariance_type="full",
            epsilon=args.epsilon,
            random_state=42,
        )
        hmm_model.fit(X, lengths=lengths)
        print(f"  Gaussian HMM fit completed.")

        # 2. 各実験動画の Zarr データと HMM 状態をマッチング
        exp_results = []
        for edir in edirs:
            res = hfc.extract_experiment_mode_flow_correlations(
                edir,
                hmm_model,
                scale=args.scale,
                tau=args.tau,
                frame_interval=args.frame_interval,
                epsilon=args.epsilon,
            )
            if res is not None:
                exp_results.append(res)

        print(f"  Successfully extracted mode correlations from {len(exp_results)} experiments.")
        if not exp_results:
            continue

        # 3. 全実験のアンサンブル集計
        df_curves = hfc.aggregate_flow_correlation_dataset(exp_results)
        df_curves['bead_name'] = bname
        df_curves['diameter_um'] = dia
        all_binned_records.append(df_curves)

        # 4. 相関長フィッティング
        fits_dict = {}
        for m_key in ['run', 'tumble', 'all', 'bg', 'run_par', 'run_perp']:
            sub_c = df_curves[df_curves['mode'] == m_key]
            fit_res = hfc.fit_flow_correlation_length(sub_c)
            fits_dict[m_key] = fit_res

            all_summary_records.append({
                'bead_name': bname,
                'diameter_um': dia,
                'mode': m_key,
                'mode_label': hfc.FLOW_MODE_NAMES.get(m_key, m_key),
                'xi_um': fit_res.get('xi_um', np.nan),
                'xi_err_um': fit_res.get('xi_err_um', np.nan),
                'amplitude_A': fit_res.get('amplitude', np.nan),
                'fit_r2': fit_res.get('r2', np.nan),
            })

        results_by_bead[bname] = {
            'df_curves': df_curves,
            'fits': fits_dict,
        }

        xi_run = fits_dict.get('run', {}).get('xi_um', np.nan)
        xi_tumble = fits_dict.get('tumble', {}).get('xi_um', np.nan)
        xi_bg = fits_dict.get('bg', {}).get('xi_um', np.nan)
        print(f"  Flow correlation length xi: Run = {xi_run:.2f} um, Tumble = {xi_tumble:.2f} um, BG = {xi_bg:.2f} um")

    if not results_by_bead:
        print("[ERROR] No flow correlation results generated.")
        return

    df_all_curves = pd.concat(all_binned_records, ignore_index=True)
    df_all_summary = pd.DataFrame(all_summary_records)

    print("\n=== Generating Figures ===")

    # Figure 1: 6パネル C_flow(r) プロット
    fig1_path = output_dir / "hmm_flow_spatial_correlation_6panel.svg"
    plot_flow_correlation_6panel(results_by_bead, BEADS_INFO, fig1_path, max_dist=args.max_dist)

    # Figure 2: モード別 4パネル比較プロット
    fig2_path = output_dir / "flow_correlation_mode_comparison_4panel.svg"
    plot_flow_mode_comparison_4panel(results_by_bead, BEADS_INFO, fig2_path, max_dist=args.max_dist)

    # Figure 3: 相関長 xi vs 粒子径
    fig3_path = output_dir / "flow_correlation_length_vs_diameter.svg"
    plot_flow_correlation_length_summary(df_all_summary, fig3_path)

    # Figure 4: ビーズ-フロー相関比較
    fig4_path = output_dir / "bead_flow_correlation_vs_diameter.svg"
    plot_bead_flow_correlation_comparison(results_by_bead, BEADS_INFO, fig4_path, max_dist=args.max_dist)

    # Figure 5: ネマチック異方性比較
    fig5_path = output_dir / "flow_correlation_anisotropy_nematic.svg"
    plot_nematic_anisotropy_summary(results_by_bead, BEADS_INFO, fig5_path, max_dist=args.max_dist)

    print("\n=== Saving CSV Summaries ===")
    csv_curves_path = output_dir / "flow_spatial_correlation_binned_curves.csv"
    safe_save_csv(df_all_curves, csv_curves_path)
    print(f"  Saved binned curves: {csv_curves_path}")

    csv_sum_path = output_dir / "flow_spatial_correlation_summary.csv"
    safe_save_csv(df_all_summary, csv_sum_path)
    print(f"  Saved summary metrics: {csv_sum_path}")

    print("\n=================================================================")
    print(" All analyses and figure generations completed successfully!")
    print("=================================================================")


if __name__ == "__main__":
    main()
