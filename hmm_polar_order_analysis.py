"""
hmm_polar_order_analysis.py

微小管の局所ポーラーオーダー（local_polar_w.zarr / local_polar_bg.zarr）を、
貨物微粒子の1次元対数速力 Gaussian HMM 推定状態（Run / Tumble）別に分解して一括解析・可視化・定量化するスクリプトです。

全ビーズサイズ（0.63μm, 1.18μm, 3.37μm, 5.0μm, 7.24μm, 20μm）において、
1. 各粒子径でのモード別局所ポーラーオーダー Phi(R) 曲線 (Run, Tumble, All, Background) 6パネルプロット
2. 粒子径 vs モード別局所ポーラーオーダーの比較プロット (4パネル)
3. 粒子径 vs 局所ポーラーオーダー (代表スケール R = 10, 25, 50 um での Run vs Tumble vs BG)
4. モード間秩序差 Delta Phi(R) = Phi_Run(R) - Phi_Tumble(R) vs 粒子径プロット
5. 統計サマリー CSV (全モード曲線データ & 代表スケール要約値) の出力
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
from libs import hmm_polar_order as hpo

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
        if p.is_dir() and (p / "beads_tracks.csv").exists() and (p / "local_polar_w.zarr").exists():
            exp_dirs.append(p)
    if not exp_dirs:
        for p in sorted(base.glob("*")):
            if p.is_dir() and (p / "beads_tracks.csv").exists() and (p / "local_polar_w.zarr").exists():
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

def plot_local_polar_6panel(
    results_by_bead: Dict[str, dict],
    beads_info: List[dict],
    output_path: Path,
    max_window: float = 60.0,
):
    """
    全ビーズサイズ（6パネル）におけるモード別局所ポーラーオーダー Phi(R) vs R をプロットする。
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
        hpo.plot_polar_orders_single_axis(
            df_curves,
            ax=ax,
            title=title_str,
            show_legend=(idx == 0),
        )

        ax.set_xlim(0, max_window)
        if idx < 3:
            ax.set_xlabel("")
        if idx % 3 != 0:
            ax.set_ylabel("")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved 6-panel local polar order plot: {output_path}")


def plot_polar_mode_comparison_4panel(
    results_by_bead: Dict[str, dict],
    beads_info: List[dict],
    output_path: Path,
    max_window: float = 60.0,
):
    """
    4つのモードカテゴリ（Run, Tumble, All, Background）ごとに
    全粒子径の局所ポーラーオーダー Phi(R) を比較する4パネルプロット。
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()

    categories = [
        ('run', 'Local Polar Order Around Run Particles (Active Mode)'),
        ('tumble', 'Local Polar Order Around Tumble Particles (Paused Mode)'),
        ('all', 'Local Polar Order Around All Particles'),
        ('bg', 'Background Local Polar Order (Bulk Flow)'),
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

            w = sub['window_size_um'].to_numpy()
            p = sub['mean_polar_order'].to_numpy()
            sem = sub['sem_polar_order'].to_numpy()

            ax.errorbar(
                w, p, yerr=sem,
                label=f"{dia:.2f} $\\mu$m",
                color=color,
                fmt=marker,
                markersize=4,
                linestyle='-',
                linewidth=1.5,
                capsize=2,
                alpha=0.85,
            )

        ax.set_xlabel(r"Window Size $R$ [$\mu\mathrm{m}$]", fontsize=11)
        ax.set_ylabel(r"Local Polar Order $\Phi(R)$", fontsize=11)
        ax.set_xlim(0, max_window)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_title(title, fontsize=12, fontweight='bold')
        if c_idx == 0:
            ax.legend(fontsize=9, loc='upper right', framealpha=0.9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved 4-panel polar order mode comparison plot: {output_path}")


def plot_polar_order_vs_diameter_scales(
    df_all_curves: pd.DataFrame,
    beads_info: List[dict],
    output_path: Path,
    target_scales_um: List[float] = [10.0, 25.0, 50.0],
):
    """
    代表的な空間スケール (R = 10, 25, 50 um) における
    粒子径 vs 局所ポーラーオーダー Phi (Run vs Tumble vs BG) のプロット。
    """
    fig, axes = plt.subplots(1, len(target_scales_um), figsize=(5.0 * len(target_scales_um), 5.0))
    if len(target_scales_um) == 1:
        axes = [axes]

    for ax_idx, target_r in enumerate(target_scales_um):
        ax = axes[ax_idx]

        for mode_key, label, col, m in [
            ('run', 'Run Particle Vicinity', '#1b9e77', 'o'),
            ('tumble', 'Tumble Particle Vicinity', '#d95f02', 'd'),
            ('bg', 'Background (Bulk)', '#7570b3', '^'),
            ('all', 'All Particles', '#222222', 's'),
        ]:
            dia_list = []
            val_list = []
            sem_list = []

            for binfo in beads_info:
                bname = binfo['name']
                dia = binfo['diameter_um']

                sub = df_all_curves[(df_all_curves['bead_name'] == bname) & (df_all_curves['mode'] == mode_key)]
                if sub.empty:
                    continue

                # target_r に最も近い窓サイズを探索
                idx_closest = (sub['window_size_um'] - target_r).abs().idxmin()
                row = sub.loc[idx_closest]

                dia_list.append(dia)
                val_list.append(row['mean_polar_order'])
                sem_list.append(row['sem_polar_order'])

            if len(dia_list) > 0:
                ax.errorbar(
                    dia_list, val_list, yerr=sem_list,
                    label=label,
                    color=col,
                    fmt=m,
                    markersize=6,
                    linewidth=1.8,
                    capsize=3,
                    alpha=0.9,
                )

        ax.set_xscale('log')
        ax.set_xlabel(r"Cargo Bead Diameter $d$ [$\mu\mathrm{m}$]", fontsize=11)
        ax.set_ylabel(rf"Local Polar Order $\Phi(R={target_r:.0f}\,\mu\mathrm{{m}})$", fontsize=11)
        ax.set_ylim(0.0, 1.05)
        ax.set_title(rf"Scale $R = {target_r:.0f}\,\mu\mathrm{{m}}$", fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.4)
        if ax_idx == 0:
            ax.legend(fontsize=8.5, loc='best')

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved polar order vs diameter scales plot: {output_path}")


def plot_polar_order_enhancement_diff(
    results_by_bead: Dict[str, dict],
    beads_info: List[dict],
    output_path: Path,
    max_window: float = 60.0,
):
    """
    Run 状態による局所ポーラーオーダーの増強差
    Delta Phi(R) = Phi_Run(R) - Phi_Tumble(R) vs R のプロット。
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

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

        sub_run = df_c[df_c['mode'] == 'run'].set_index('window_size_um')
        sub_tumble = df_c[df_c['mode'] == 'tumble'].set_index('window_size_um')

        common_windows = sub_run.index.intersection(sub_tumble.index)
        common_windows = [w for w in common_windows if w <= max_window]
        if len(common_windows) < 3:
            continue

        w_arr = np.array(sorted(common_windows))
        diff_arr = sub_run.loc[w_arr, 'mean_polar_order'].values - sub_tumble.loc[w_arr, 'mean_polar_order'].values
        err_arr = np.sqrt(sub_run.loc[w_arr, 'sem_polar_order'].values**2 + sub_tumble.loc[w_arr, 'sem_polar_order'].values**2)

        ax.errorbar(
            w_arr, diff_arr, yerr=err_arr,
            label=f"{dia:.2f} $\\mu$m",
            color=color,
            fmt=marker,
            markersize=4.5,
            linewidth=1.8,
            capsize=2,
            alpha=0.9,
        )

    ax.axhline(0.0, color='gray', linestyle='--', linewidth=1.0, alpha=0.6)
    ax.set_xlabel(r"Window Size $R$ [$\mu\mathrm{m}$]", fontsize=12)
    ax.set_ylabel(r"Order Enhancement $\Delta \Phi(R) = \Phi_{\mathrm{Run}}(R) - \Phi_{\mathrm{Tumble}}(R)$", fontsize=12)
    ax.set_title("Polar Order Enhancement by Active Cargo Movement", fontsize=13, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=9.5, loc='best')

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved polar order enhancement diff plot: {output_path}")


# =========================================================================
# メイン処理
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="HMM Mode-Dependent Local Polar Order Analysis for Microtubule Active Flows."
    )
    parser.add_argument('--root_dir', type=str, default=None, help="Root directory containing beads data.")
    parser.add_argument('--output_dir', type=str, default='figures/hmm_polar_order', help="Output directory for figures & CSVs.")
    parser.add_argument('--beads', type=str, default='all', help="Target bead condition ('all' or 'beads1um' etc.)")
    parser.add_argument('--tau', type=int, default=1, help="Lag time step for velocity calculation.")
    parser.add_argument('--scale', type=float, default=0.11, help="Spatial scale (um/pixel).")
    parser.add_argument('--frame_interval', type=float, default=4.0, help="Time interval between frames (s).")
    parser.add_argument('--epsilon', type=float, default=1e-3, help="Epsilon for log-speed observation ln(v + eps).")
    parser.add_argument('--max_window', type=float, default=60.0, help="Maximum window size in um for plotting.")
    args = parser.parse_args()

    root_dir = Path(args.root_dir) if args.root_dir else find_default_root()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print(" HMM Mode-Dependent MT Local Polar Order Analysis")
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
            print(f"[WARNING] No experiment directories with local_polar_w.zarr found for {bname} in {root_dir}")
            continue

        print(f"  Found {len(edirs)} experiment directories with computed polar order Zarrs.")

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
            res = hpo.extract_experiment_mode_polar_orders(
                edir,
                hmm_model,
                scale=args.scale,
                tau=args.tau,
                frame_interval=args.frame_interval,
                epsilon=args.epsilon,
            )
            if res is not None:
                exp_results.append(res)

        print(f"  Successfully extracted mode polar orders from {len(exp_results)} experiments.")
        if not exp_results:
            continue

        # 3. 全実験のアンサンブル集計
        df_curves = hpo.aggregate_polar_order_dataset(exp_results)
        df_curves['bead_name'] = bname
        df_curves['diameter_um'] = dia
        all_binned_records.append(df_curves)

        # 4. 代表スケール (R = 10, 25, 50 um) での要約統計量
        for target_r in [10.0, 25.0, 50.0]:
            sub_r = df_curves.copy()
            idx_closest = (sub_r['window_size_um'] - target_r).abs().idxmin()
            actual_r = sub_r.loc[idx_closest, 'window_size_um']

            for m_key in ['run', 'tumble', 'all', 'bg']:
                sub_m = df_curves[(df_curves['mode'] == m_key) & (df_curves['window_size_um'] == actual_r)]
                if not sub_m.empty:
                    row = sub_m.iloc[0]
                    all_summary_records.append({
                        'bead_name': bname,
                        'diameter_um': dia,
                        'target_scale_um': target_r,
                        'actual_window_um': actual_r,
                        'mode': m_key,
                        'mode_label': hpo.POLAR_MODE_NAMES.get(m_key, m_key),
                        'mean_polar_order': row['mean_polar_order'],
                        'sem_polar_order': row['sem_polar_order'],
                        'n_samples': row['n_samples'],
                    })

        results_by_bead[bname] = {
            'df_curves': df_curves,
        }

    if not results_by_bead:
        print("[ERROR] No polar order results generated.")
        return

    df_all_curves = pd.concat(all_binned_records, ignore_index=True)
    df_all_summary = pd.DataFrame(all_summary_records)

    print("\n=== Generating Figures ===")

    # Figure 1: 6パネル Phi(R) プロット
    fig1_path = output_dir / "hmm_local_polar_order_6panel.svg"
    plot_local_polar_6panel(results_by_bead, BEADS_INFO, fig1_path, max_window=args.max_window)

    # Figure 2: モード別 4パネル比較プロット
    fig2_path = output_dir / "polar_order_mode_comparison_4panel.svg"
    plot_polar_mode_comparison_4panel(results_by_bead, BEADS_INFO, fig2_path, max_window=args.max_window)

    # Figure 3: 代表スケールでの Phi vs 粒子径
    fig3_path = output_dir / "polar_order_vs_diameter_scales.svg"
    plot_polar_order_vs_diameter_scales(df_all_curves, BEADS_INFO, fig3_path, target_scales_um=[10.0, 25.0, 50.0])

    # Figure 4: ポーラーオーダー増強差 Delta Phi vs R
    fig4_path = output_dir / "polar_order_enhancement_diff_vs_diameter.svg"
    plot_polar_order_enhancement_diff(results_by_bead, BEADS_INFO, fig4_path, max_window=args.max_window)

    print("\n=== Saving CSV Summaries ===")
    csv_curves_path = output_dir / "polar_order_binned_curves.csv"
    safe_save_csv(df_all_curves, csv_curves_path)
    print(f"  Saved binned curves: {csv_curves_path}")

    csv_sum_path = output_dir / "polar_order_summary.csv"
    safe_save_csv(df_all_summary, csv_sum_path)
    print(f"  Saved summary metrics: {csv_sum_path}")

    print("\n=================================================================")
    print(" All analyses and figure generations completed successfully!")
    print("=================================================================")


if __name__ == "__main__":
    main()
