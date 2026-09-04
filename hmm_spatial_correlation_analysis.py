"""
hmm_spatial_correlation_analysis.py

貨物微粒子の1次元対数速力 Gaussian HMM 解析（K=2: Run / Tumble）の結果をもとに、
粒子の運動モード別に空間配向相関（Spatial Orientational / Velocity Correlation）
C(r) = < v_i · v_j >_{r_{ij} ≈ r}
を一括解析・可視化・定量化するスクリプトです。

全ビーズサイズ（0.63μm, 1.18μm, 3.37μm, 5.0μm, 7.24μm, 20μm）において、
1. 各粒子径でのモード別空間配向相関 C(r) 曲線 (Run-Run, Tumble-Tumble, Run-Tumble, All)
2. 粒子径 vs モード別相関関数の比較プロット (4パネル)
3. 粒子径 vs 配向相関長 xi (Correlation Length) の定量プロット
4. 粒子径 vs 近距離配向秩序パラメータ C(r <= 10 um) の比較
5. 空間ベクトル場 & モード色分けスナップショット可視化
6. 統計サマリー CSV (曲線データ & フィッティング結果) の出力
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
from libs import spatial_correlation as sc

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

STATE_COLORS = {
    0: '#d95f02',  # Tumble: オレンジ系
    1: '#1b9e77',  # Run: 青緑系
}


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
        if p.is_dir() and (p / "beads_tracks.csv").exists():
            exp_dirs.append(p)
    if not exp_dirs:
        for p in sorted(base.glob("*")):
            if p.is_dir() and (p / "beads_tracks.csv").exists():
                exp_dirs.append(p)
    return exp_dirs


_TRACKS_CACHE = {}


def load_cached_tracks(csv_path: Path) -> Optional[pd.DataFrame]:
    str_path = str(csv_path)
    if str_path in _TRACKS_CACHE:
        return _TRACKS_CACHE[str_path]
    try:
        df = pd.read_csv(csv_path)
        if {'particle', 'frame', 'x', 'y'}.issubset(df.columns):
            _TRACKS_CACHE[str_path] = df[['particle', 'frame', 'x', 'y']].copy()
            return _TRACKS_CACHE[str_path]
    except Exception as e:
        print(f"[WARNING] Failed to load {csv_path}: {e}", flush=True)
    return None


def collect_bead_hmm_data(
    exp_dirs: List[Path],
    tau: int = 1,
    scale: float = 0.11,
    frame_interval: float = 4.0,
    epsilon: float = 1e-3,
) -> Tuple[np.ndarray, List[int], pd.DataFrame]:
    all_dfs = []
    particle_offset = 0

    for edir in exp_dirs:
        tracks_csv = edir / "beads_tracks.csv"
        if not tracks_csv.exists():
            continue
        df_tracks = load_cached_tracks(tracks_csv)
        if df_tracks is None or df_tracks.empty:
            continue

        df_copy = df_tracks.copy()
        if 'particle' not in df_copy.columns or 'frame' not in df_copy.columns:
            continue

        df_copy['particle'] = df_copy['particle'] + particle_offset
        particle_offset += int(df_copy['particle'].max()) + 1
        df_copy['exp_dir'] = edir.name
        all_dfs.append(df_copy)

    if not all_dfs:
        return np.empty((0, 1)), [], pd.DataFrame()

    df_combined = pd.concat(all_dfs, ignore_index=True)
    X, lengths, df_obs = hc.extract_hmm_features(
        df_combined,
        tau=tau,
        scale=scale,
        frame_interval=frame_interval,
        epsilon=epsilon,
    )
    return X, lengths, df_obs


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

def plot_spatial_correlation_6panel(
    results_dict: Dict[str, dict],
    beads_info: List[dict],
    output_path: Path,
    max_dist: float = 60.0,
):
    """
    全ビーズサイズ（6パネル）におけるモード別空間配向相関 C(r) vs r をプロットする。
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    axes = axes.flatten()

    for idx, binfo in enumerate(beads_info):
        ax = axes[idx]
        bname = binfo['name']
        dia = binfo['diameter_um']

        if bname not in results_dict or results_dict[bname]['df_binned'].empty:
            ax.set_title(f"$d = {dia:.2f}\\,\\mu\\mathrm{{m}}$ (No data)", fontsize=12)
            ax.axis('off')
            continue

        res = results_dict[bname]
        df_binned = res['df_binned']
        fit_results = res.get('fits', {})

        title_str = f"$d = {dia:.2f}\\,\\mu\\mathrm{{m}}$"
        sc.plot_mode_correlations_single_axis(
            df_binned,
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
    print(f"  Saved 6-panel spatial correlation plot: {output_path}")


def plot_mode_comparison_4panel(
    results_dict: Dict[str, dict],
    beads_info: List[dict],
    output_path: Path,
    max_dist: float = 60.0,
):
    """
    4つのモードカテゴリ（Run-Run, Tumble-Tumble, Run-Tumble, All）ごとに
    全粒子径の C(r) を比較する4パネルプロット。
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()

    categories = [
        ('run_run', 'Run - Run Pairs (High-speed Active Pairs)'),
        ('tumble_tumble', 'Tumble - Tumble Pairs (Paused/Stationary Pairs)'),
        ('run_tumble', 'Run - Tumble Pairs (Cross Interaction)'),
        ('all', 'All Pairs (Overall Orientational Correlation)'),
    ]

    for c_idx, (cat, title) in enumerate(categories):
        ax = axes[c_idx]

        for binfo in beads_info:
            bname = binfo['name']
            dia = binfo['diameter_um']
            color = binfo['color']
            marker = binfo['marker']

            if bname not in results_dict:
                continue

            df_b = results_dict[bname]['df_binned']
            if df_b.empty:
                continue

            sub = df_b[df_b['mode_category'] == cat]
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
        ax.set_xlabel(r"Interparticle Distance $r$ [$\mu\mathrm{m}$]", fontsize=11)
        ax.set_ylabel(r"Spatial Correlation $C(r) = \langle \hat{\mathbf{v}}_i \cdot \hat{\mathbf{v}}_j \rangle$", fontsize=11)
        ax.set_xlim(0, max_dist)
        ax.set_ylim(-0.3, 1.05)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_title(title, fontsize=12, fontweight='bold')
        if c_idx == 0:
            ax.legend(fontsize=9, loc='upper right', framealpha=0.9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved 4-panel mode comparison plot: {output_path}")


def plot_correlation_length_summary(
    df_summary: pd.DataFrame,
    output_path: Path,
):
    """
    粒子径 vs 配向相関長 xi (Correlation Length) のプロット。
    """
    fig, ax = plt.subplots(figsize=(7, 5.5))

    target_modes = [
        ('run_run', 'Run - Run', '#1b9e77', 'o'),
        ('all', 'All Pairs', '#333333', 's'),
    ]

    for cat, label, col, m in target_modes:
        sub = df_summary[df_summary['mode_category'] == cat].dropna(subset=['xi_um'])
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
    ax.set_ylabel(r"Orientational Correlation Length $\xi$ [$\mu\mathrm{m}$]", fontsize=12)
    ax.set_title("Orientational Correlation Length vs Cargo Diameter", fontsize=13, fontweight='bold')
    ax.grid(True, which="both", linestyle='--', alpha=0.4)
    ax.legend(fontsize=10, loc='best')

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved correlation length summary plot: {output_path}")


def plot_short_range_order_summary(
    df_summary: pd.DataFrame,
    output_path: Path,
):
    """
    粒子径 vs 近距離配向秩序度 C(r <= 10 um) のプロット。
    """
    fig, ax = plt.subplots(figsize=(7, 5.5))

    target_modes = [
        ('run_run', 'Run - Run', '#1b9e77', 'o'),
        ('run_tumble', 'Run - Tumble', '#7570b3', '^'),
        ('tumble_tumble', 'Tumble - Tumble', '#d95f02', 'd'),
        ('all', 'All Pairs', '#333333', 's'),
    ]

    for cat, label, col, m in target_modes:
        sub = df_summary[df_summary['mode_category'] == cat].dropna(subset=['short_range_order'])
        if sub.empty:
            continue

        dia = sub['diameter_um'].to_numpy()
        sro = sub['short_range_order'].to_numpy()

        ax.plot(
            dia, sro,
            label=label,
            color=col,
            marker=m,
            markersize=7,
            linewidth=2.0,
            alpha=0.9,
        )

    ax.set_xscale('log')
    ax.axhline(0.0, color='gray', linestyle='--', linewidth=1.0, alpha=0.6)
    ax.set_xlabel(r"Cargo Bead Diameter $d$ [$\mu\mathrm{m}$]", fontsize=12)
    ax.set_ylabel(r"Short-Range Order $C(r \leq 10\,\mu\mathrm{m})$", fontsize=12)
    ax.set_title("Short-Range Orientational Order vs Cargo Diameter", fontsize=13, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=10, loc='best')

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved short-range order summary plot: {output_path}")


def plot_snapshot_vector_field(
    df_obs: pd.DataFrame,
    output_path: Path,
    target_frame: Optional[int] = None,
):
    """
    代表フレームにおける微粒子位置・運動ベクトル・HMMモードの空間スナップショットを描画する。
    """
    if df_obs.empty:
        return

    # 最も粒子数が多いフレームを選択
    frame_counts = df_obs.groupby('frame').size()
    if target_frame is None or target_frame not in frame_counts:
        best_frame = frame_counts.idxmax()
    else:
        best_frame = target_frame

    sub = df_obs[df_obs['frame'] == best_frame].copy()
    if len(sub) < 2:
        return

    fig, ax = plt.subplots(figsize=(8, 7))

    x = sub['x_um'].to_numpy()
    y = sub['y_um'].to_numpy()
    dx = sub['dx_um'].to_numpy()
    dy = sub['dy_um'].to_numpy()
    st = sub['pred_state'].to_numpy()

    # 粒子ペア間の線（近距離のペアを薄く結ぶ）
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            d = np.hypot(x[i] - x[j], y[i] - y[j])
            if d <= 30.0:
                ax.plot([x[i], x[j]], [y[i], y[j]], color='gray', linestyle=':', lw=1.0, alpha=0.4)

    # 粒子の描画 (Tumble: 橙, Run: 緑)
    for s_val, label, col in [(0, 'Tumble / Pause', STATE_COLORS[0]), (1, 'Run', STATE_COLORS[1])]:
        mask = (st == s_val)
        if not np.any(mask):
            continue

        ax.scatter(x[mask], y[mask], color=col, s=80, label=label, edgecolors='black', lw=0.8, zorder=3)
        ax.quiver(
            x[mask], y[mask], dx[mask], dy[mask],
            color=col, angles='xy', scale_units='xy', scale=0.5,
            width=0.005, headwidth=4, headlength=5, zorder=4, alpha=0.85
        )

    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel(r"$x$ [$\mu\mathrm{m}$]", fontsize=12)
    ax.set_ylabel(r"$y$ [$\mu\mathrm{m}$]", fontsize=12)
    ax.set_title(f"Cargo Velocity Field & HMM Mode Snapshot (Frame {best_frame})", fontsize=13, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=10, loc='upper right', framealpha=0.9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved snapshot demo plot: {output_path}")


# =========================================================================
# メイン処理
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Mode-Dependent Spatial Orientational Correlation Analysis for Cargo Microparticles via HMM."
    )
    parser.add_argument('--root_dir', type=str, default=None, help="Root directory containing beads data.")
    parser.add_argument('--output_dir', type=str, default='figures/spatial_correlation', help="Output directory for figures & CSVs.")
    parser.add_argument('--beads', type=str, default='all', help="Target bead condition ('all' or 'beads1um' etc.)")
    parser.add_argument('--tau', type=int, default=1, help="Lag time step for velocity calculation.")
    parser.add_argument('--scale', type=float, default=0.11, help="Spatial scale (um/pixel).")
    parser.add_argument('--frame_interval', type=float, default=4.0, help="Time interval between frames (s).")
    parser.add_argument('--epsilon', type=float, default=1e-3, help="Epsilon for log-speed observation ln(v + eps).")
    parser.add_argument('--bin_width', type=float, default=2.0, help="Distance bin width in um.")
    parser.add_argument('--max_dist', type=float, default=60.0, help="Maximum distance in um for C(r).")
    parser.add_argument('--r_cutoff', type=float, default=10.0, help="Cutoff distance in um for short-range order.")
    args = parser.parse_args()

    root_dir = Path(args.root_dir) if args.root_dir else find_default_root()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print(" HMM Mode-Dependent Spatial Orientational Correlation Analysis")
    print("=================================================================")
    print(f"Data Root Directory: {root_dir}")
    print(f"Output Directory:    {output_dir}")
    print(f"Lag time tau:        {args.tau} ({args.tau * args.frame_interval:.1f} s)")
    print(f"Distance Bin Width:  {args.bin_width:.1f} um (Max: {args.max_dist:.1f} um)")
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
            print(f"[WARNING] No experiment directories found for {bname} in {root_dir}")
            continue

        print(f"  Found {len(edirs)} experiment directories.")

        X, lengths, df_obs = collect_bead_hmm_data(
            edirs,
            tau=args.tau,
            scale=args.scale,
            frame_interval=args.frame_interval,
            epsilon=args.epsilon,
        )

        if len(X) < 20:
            print(f"[WARNING] Insufficient data points ({len(X)}) for {bname}. Skipping.")
            continue

        # 1. 2状態 Gaussian HMM のフィッティング
        hmm_model = hc.CargoGaussianHMM(
            n_components=2,
            covariance_type="full",
            epsilon=args.epsilon,
            random_state=42,
        )
        hmm_model.fit(X, lengths=lengths)
        pred_states = hmm_model.predict(X, lengths=lengths)
        df_obs['pred_state'] = pred_states

        print(f"  HMM fit complete. Run fraction: {np.mean(pred_states == 1)*100:.1f}%")

        # 2. 空間配向相関ペアの計算
        df_pairs = sc.compute_dataset_spatial_correlation(df_obs, normalize=True)
        print(f"  Computed {len(df_pairs):,} particle pair interactions.")

        if df_pairs.empty:
            continue

        # 3. 距離ビン集計
        df_binned = sc.bin_spatial_correlation(
            df_pairs,
            bin_width=args.bin_width,
            max_dist=args.max_dist,
        )
        df_binned['bead_name'] = bname
        df_binned['diameter_um'] = dia
        all_binned_records.append(df_binned)

        # 4. 短距離秩序度と相関長フィッティング
        sro_dict = sc.compute_short_range_order(df_pairs, r_cutoff=args.r_cutoff)
        fits_dict = {}

        for cat in ['run_run', 'tumble_tumble', 'run_tumble', 'all']:
            sub_curve = df_binned[df_binned['mode_category'] == cat]
            fit_res = sc.fit_correlation_length(sub_curve, model_type='exponential')
            fits_dict[cat] = fit_res

            all_summary_records.append({
                'bead_name': bname,
                'diameter_um': dia,
                'mode_category': cat,
                'mode_label': sc.MODE_PAIR_NAMES[cat],
                'short_range_order': sro_dict.get(cat, np.nan),
                'xi_um': fit_res.get('xi_um', np.nan),
                'xi_err_um': fit_res.get('xi_err_um', np.nan),
                'amplitude_A': fit_res.get('amplitude', np.nan),
                'fit_r2': fit_res.get('r2', np.nan),
                'total_pairs': len(df_pairs[df_pairs['mode_category'] == cat]) if cat != 'all' else len(df_pairs),
            })

        results_by_bead[bname] = {
            'df_obs': df_obs,
            'df_pairs': df_pairs,
            'df_binned': df_binned,
            'fits': fits_dict,
            'sro': sro_dict,
        }

        xi_run = fits_dict.get('run_run', {}).get('xi_um', np.nan)
        xi_all = fits_dict.get('all', {}).get('xi_um', np.nan)
        print(f"  Correlation length xi: Run-Run = {xi_run:.2f} um, All = {xi_all:.2f} um")

    if not results_by_bead:
        print("[ERROR] No correlation results generated.")
        return

    df_all_binned = pd.concat(all_binned_records, ignore_index=True)
    df_all_summary = pd.DataFrame(all_summary_records)

    print("\n=== Generating Figures ===")

    # Figure 1: 6パネル C(r) プロット
    fig1_path = output_dir / "hmm_spatial_correlation_6panel.svg"
    plot_spatial_correlation_6panel(results_by_bead, BEADS_INFO, fig1_path, max_dist=args.max_dist)

    # Figure 2: モード別 4パネル比較プロット
    fig2_path = output_dir / "spatial_correlation_mode_comparison_4panel.svg"
    plot_mode_comparison_4panel(results_by_bead, BEADS_INFO, fig2_path, max_dist=args.max_dist)

    # Figure 3: 相関長 xi vs 粒子径
    fig3_path = output_dir / "correlation_length_vs_diameter.svg"
    plot_correlation_length_summary(df_all_summary, fig3_path)

    # Figure 4: 近距離秩序度 vs 粒子径
    fig4_path = output_dir / "short_range_order_vs_diameter.svg"
    plot_short_range_order_summary(df_all_summary, fig4_path)

    # Figure 5: 代表スナップショット
    first_bname = list(results_by_bead.keys())[0]
    fig5_path = output_dir / "spatial_correlation_snapshot_demo.svg"
    plot_snapshot_vector_field(results_by_bead[first_bname]['df_obs'], fig5_path)

    print("\n=== Saving CSV Summaries ===")
    csv_binned_path = output_dir / "spatial_correlation_binned_curves.csv"
    safe_save_csv(df_all_binned, csv_binned_path)
    print(f"  Saved binned curves: {csv_binned_path}")

    csv_sum_path = output_dir / "spatial_correlation_summary.csv"
    safe_save_csv(df_all_summary, csv_sum_path)
    print(f"  Saved summary metrics: {csv_sum_path}")

    print("\n=================================================================")
    print(" All analyses and figure generations completed successfully!")
    print("=================================================================")


if __name__ == "__main__":
    main()
