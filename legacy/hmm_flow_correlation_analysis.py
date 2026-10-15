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
from scipy import stats

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


def normalize_bead_name(raw_name: str) -> Optional[str]:
    """
    入力文字列（例: 'beads06um', 'bead06um', '06um', '0.6um', '0.6', '1um', '1', etc.）を
    BEADS_INFO の標準名 ('beads06um' 等) に正規化する。
    """
    s = raw_name.strip().lower()
    mapping = {
        'beads06um': 'beads06um', 'bead06um': 'beads06um', '06um': 'beads06um', '0.6um': 'beads06um', '0.6': 'beads06um', '06': 'beads06um',
        'beads1um': 'beads1um', 'bead1um': 'beads1um', '1um': 'beads1um', '1.0um': 'beads1um', '1.18um': 'beads1um', '1': 'beads1um',
        'beads3um': 'beads3um', 'bead3um': 'beads3um', '3um': 'beads3um', '3.0um': 'beads3um', '3.37um': 'beads3um', '3': 'beads3um',
        'beads5um': 'beads5um', 'bead5um': 'beads5um', '5um': 'beads5um', '5.0um': 'beads5um', '5': 'beads5um',
        'beads7um': 'beads7um', 'bead7um': 'beads7um', '7um': 'beads7um', '7.0um': 'beads7um', '7.24um': 'beads7um', '7': 'beads7um',
        'beads20um': 'beads20um', 'bead20um': 'beads20um', '20um': 'beads20um', '20.0um': 'beads20um', '20': 'beads20um',
    }
    return mapping.get(s, None)


def parse_target_beads(beads_args: Union[str, List[str]], beads_info: List[dict] = BEADS_INFO) -> List[dict]:
    """
    argparse の引数（リストまたはカンマ/スペース区切りの文字列）から対象ビーズ情報のリストを抽出・構築する。
    """
    if isinstance(beads_args, str):
        raw_list = [beads_args]
    else:
        raw_list = list(beads_args)

    tokens = []
    for item in raw_list:
        parts = item.replace(',', ' ').split()
        tokens.extend(parts)

    if not tokens or 'all' in [t.lower() for t in tokens]:
        return beads_info

    selected_names = set()
    for token in tokens:
        norm = normalize_bead_name(token)
        if norm:
            selected_names.add(norm)
        else:
            found = False
            for b in beads_info:
                if b['name'].lower() == token.lower():
                    selected_names.add(b['name'])
                    found = True
                    break
            if not found:
                print(f"[WARNING] Unrecognized bead specification: '{token}'. Available: {[b['name'] for b in beads_info]}")

    target = [b for b in beads_info if b['name'] in selected_names]
    if not target:
        raise ValueError(f"No valid beads matched from input: {beads_args}. Available: {[b['name'] for b in beads_info]}")
    return target


# =========================================================================
# 可視化関数群
# =========================================================================

def plot_flow_correlation_6panel(
    results_by_bead: Dict[str, dict],
    beads_info: List[dict],
    output_path: Path,
    max_dist: float = 60.0,
    min_fit_dist: float = 0.0,
    max_fit_dist: float = 20.0,
):
    """
    選択ビーズサイズにおけるモード別微小管フロー空間相関 C_flow(r) vs r をプロットする（動的グリッドレイアウト）。
    """
    n_beads = len(beads_info)
    ncols = min(n_beads, 3)
    nrows = int(np.ceil(n_beads / ncols)) if ncols > 0 else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.8 * nrows), squeeze=False)
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
            min_fit_dist=min_fit_dist,
            max_fit_dist=max_fit_dist,
            show_legend=(idx == 0),
        )

        ax.set_xlim(0, max_dist)
        row_idx = idx // ncols
        col_idx = idx % ncols
        if row_idx < nrows - 1:
            ax.set_xlabel("")
        if col_idx != 0:
            ax.set_ylabel("")

    for i in range(n_beads, len(axes)):
        axes[i].axis('off')

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
    大局的ネマチック主軸に平行な相関 (Parallel //) vs 垂直な相関 (Perpendicular perp) の異方性プロット（動的グリッドレイアウト）。
    """
    n_beads = len(beads_info)
    ncols = min(n_beads, 3)
    nrows = int(np.ceil(n_beads / ncols)) if ncols > 0 else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.8 * nrows), squeeze=False)
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
        row_idx = idx // ncols
        col_idx = idx % ncols
        if row_idx < nrows - 1:
            ax.set_xlabel(r"Distance $r$ [$\mu\mathrm{m}$]", fontsize=11)
        if col_idx == 0:
            ax.set_ylabel(r"Flow Correlation $C(r)$", fontsize=11)
        if idx == 0:
            ax.legend(fontsize=8.5, loc='upper right', framealpha=0.9)

    for i in range(n_beads, len(axes)):
        axes[i].axis('off')

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
    parser.add_argument('--output_dir', type=str, default='figure/hmm_flow_correlation', help="Output directory for figures & CSVs.")
    parser.add_argument('--beads', type=str, nargs='+', default=['all'], help="Target bead conditions (e.g. 'all', 'beads1um', 'beads06um beads1um beads3um', 'bead06um, bead1um, bead3um').")
    parser.add_argument('--tau', type=int, default=1, help="Lag time step for velocity calculation.")
    parser.add_argument('--scale', type=float, default=0.11, help="Spatial scale (um/pixel).")
    parser.add_argument('--frame_interval', type=float, default=4.0, help="Time interval between frames (s).")
    parser.add_argument('--epsilon', type=float, default=1e-3, help="Epsilon for log-speed observation ln(v + eps).")
    parser.add_argument('--max_dist', type=float, default=60.0, help="Maximum distance in um for C(r) plotting.")
    parser.add_argument('--fit_range', type=float, nargs=2, default=[0.0, 20.0], metavar=('MIN', 'MAX'),
                        help="Fit distance range in um for correlation length (default: 0.0 20.0).")
    parser.add_argument('--min_fit_dist', type=float, default=None, help="Minimum fit distance in um (overrides fit_range[0]).")
    parser.add_argument('--max_fit_dist', type=float, default=None, help="Maximum fit distance in um (overrides fit_range[1]).")
    parser.add_argument('--min_particle_frames', type=int, default=10, help="Min frames in a mode required for per-particle fit (default: 10).")
    args = parser.parse_args()

    min_fit = float(args.min_fit_dist if args.min_fit_dist is not None else args.fit_range[0])
    max_fit = float(args.max_fit_dist if args.max_fit_dist is not None else args.fit_range[1])

    root_dir = Path(args.root_dir) if args.root_dir else find_default_root()
    out_arg = Path(args.output_dir)
    output_dir = out_arg if out_arg.is_absolute() else (root_dir / out_arg)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        target_bead_infos = parse_target_beads(args.beads, BEADS_INFO)
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    print("=================================================================")
    print(" HMM Mode-Dependent MT Optical Flow Angular Correlation Analysis")
    print("=================================================================")
    print(f"Data Root Directory: {root_dir}")
    print(f"Output Directory:    {output_dir}")
    print(f"Target Beads:        {[b['name'] for b in target_bead_infos]}")
    print(f"Lag time tau:        {args.tau} ({args.tau * args.frame_interval:.1f} s)")
    print(f"Scale:               {args.scale} um/pixel")
    print(f"Fit distance range:  [{min_fit:.1f}, {max_fit:.1f}] um")
    print(f"Min particle frames: {args.min_particle_frames}")
    print("=================================================================\n")

    results_by_bead = {}
    all_binned_records = []
    all_summary_records = []
    all_particle_records = []

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

            # 粒子ごと（Per-Particle）の相関長抽出
            p_res_list = hfc.extract_per_particle_flow_correlations(
                edir,
                hmm_model,
                scale=args.scale,
                tau=args.tau,
                frame_interval=args.frame_interval,
                epsilon=args.epsilon,
                min_fit_dist=min_fit,
                max_fit_dist=max_fit,
                min_frames=args.min_particle_frames,
            )
            for pres in p_res_list:
                pres['bead_name'] = bname
                pres['diameter_um'] = dia
                all_particle_records.append(pres)

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
            fit_res = hfc.fit_flow_correlation_length(sub_c, min_fit_dist=min_fit, max_fit_dist=max_fit)
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
                'fit_r2_log': fit_res.get('r2_log', np.nan),
                'r_peak_um': fit_res.get('r_peak_um', np.nan),
                'r_fit_min_um': fit_res.get('r_fit_min_um', np.nan),
                'r_fit_max_um': fit_res.get('r_fit_max_um', np.nan),
            })

        results_by_bead[bname] = {
            'df_curves': df_curves,
            'fits': fits_dict,
        }

        xi_run = fits_dict.get('run', {}).get('xi_um', np.nan)
        xi_tumble = fits_dict.get('tumble', {}).get('xi_um', np.nan)
        xi_bg = fits_dict.get('bg', {}).get('xi_um', np.nan)
        print(f"  Flow correlation length xi [{min_fit:.1f}, {max_fit:.1f}] um: Run = {xi_run:.2f} um, Tumble = {xi_tumble:.2f} um, BG = {xi_bg:.2f} um")

    if not results_by_bead:
        print("[ERROR] No flow correlation results generated.")
        return

    df_all_curves = pd.concat(all_binned_records, ignore_index=True)
    df_all_summary = pd.DataFrame(all_summary_records)
    df_all_particles = pd.DataFrame(all_particle_records) if all_particle_records else pd.DataFrame()

    print("\n=== Generating Figures ===")

    # Figure 1: 空間相関プロット (アンサンブル)
    fig1_path = output_dir / "hmm_flow_spatial_correlation_6panel.svg"
    plot_flow_correlation_6panel(results_by_bead, target_bead_infos, fig1_path, max_dist=args.max_dist, min_fit_dist=min_fit, max_fit_dist=max_fit)

    # Figure 2: モード別 4パネル比較プロット
    fig2_path = output_dir / "flow_correlation_mode_comparison_4panel.svg"
    plot_flow_mode_comparison_4panel(results_by_bead, target_bead_infos, fig2_path, max_dist=args.max_dist)

    # Figure 3: 相関長 xi vs 粒子径 (アンサンブル)
    fig3_path = output_dir / "flow_correlation_length_vs_diameter.svg"
    plot_flow_correlation_length_summary(df_all_summary, fig3_path)

    # Figure 4: ビーズ-フロー相関比較
    fig4_path = output_dir / "bead_flow_correlation_vs_diameter.svg"
    plot_bead_flow_correlation_comparison(results_by_bead, target_bead_infos, fig4_path, max_dist=args.max_dist)

    # Figure 5: ネマチック異方性比較
    fig5_path = output_dir / "flow_correlation_anisotropy_nematic.svg"
    plot_nematic_anisotropy_summary(results_by_bead, target_bead_infos, fig5_path, max_dist=args.max_dist)

    # --- 粒子ごと（Per-Particle）の図表生成 ---
    if not df_all_particles.empty:
        # Figure 6: 粒子ごとの xi_Run vs xi_Tumble 散布図
        fig6_path = output_dir / "per_particle_xi_scatter_6panel.svg"
        hfc.plot_per_particle_scatter_6panel(df_all_particles, target_bead_infos, fig6_path)

        # Figure 7: 粒子ごとの xi_Run vs xi_Tumble ペア比較 (Box & Paired Strip Plot)
        fig7_path = output_dir / "per_particle_xi_box_violin_vs_diameter.svg"
        hfc.plot_per_particle_box_violin_vs_diameter(df_all_particles, target_bead_infos, fig7_path)

        # Figure 8: 粒子ごとの相関長差 Delta xi および比 Ratio vs 粒子径
        fig8_path = output_dir / "per_particle_xi_diff_and_ratio_vs_diameter.svg"
        hfc.plot_per_particle_diff_and_ratio_vs_diameter(df_all_particles, target_bead_infos, fig8_path)

        # Figure 9: 粒子ごとの相関長累積確率分布 (eCDF)
        fig9_path = output_dir / "per_particle_xi_cdf_6panel.svg"
        hfc.plot_per_particle_cdf_6panel(df_all_particles, target_bead_infos, fig9_path)

    print("\n=== Saving CSV Summaries ===")
    csv_curves_path = output_dir / "flow_spatial_correlation_binned_curves.csv"
    safe_save_csv(df_all_curves, csv_curves_path)
    print(f"  Saved binned curves: {csv_curves_path}")

    csv_sum_path = output_dir / "flow_spatial_correlation_summary.csv"
    safe_save_csv(df_all_summary, csv_sum_path)
    print(f"  Saved summary metrics: {csv_sum_path}")

    if not df_all_particles.empty:
        csv_particles_path = output_dir / "per_particle_flow_correlation_lengths.csv"
        safe_save_csv(df_all_particles, csv_particles_path)
        print(f"  Saved per-particle correlation lengths: {csv_particles_path}")

        # 粒子ごとの統計サマリー集計
        particle_summary_records = []
        for binfo in target_bead_infos:
            bname = binfo['name']
            dia = binfo['diameter_um']
            sub = df_all_particles[df_all_particles['bead_name'] == bname].dropna(subset=['xi_run_um', 'xi_tumble_um'])
            if sub.empty:
                continue
            r_vals = sub['xi_run_um'].to_numpy()
            t_vals = sub['xi_tumble_um'].to_numpy()
            d_vals = sub['delta_xi_um'].to_numpy()
            rat_vals = sub['ratio_xi_run_to_tumble'].to_numpy()

            p_val_wilcoxon_greater = np.nan
            p_val_wilcoxon_two_sided = np.nan
            t_stat = np.nan
            p_val_ttest_greater = np.nan
            p_val_ttest_two_sided = np.nan

            if len(sub) >= 3:
                try:
                    tt_greater = stats.ttest_rel(r_vals, t_vals, alternative='greater')
                    t_stat = float(tt_greater.statistic)
                    p_val_ttest_greater = float(tt_greater.pvalue)
                    p_val_ttest_two_sided = float(stats.ttest_rel(r_vals, t_vals, alternative='two-sided').pvalue)
                except Exception:
                    pass

            if len(sub) >= 5:
                try:
                    p_val_wilcoxon_greater = float(stats.wilcoxon(r_vals, t_vals, alternative='greater').pvalue)
                    p_val_wilcoxon_two_sided = float(stats.wilcoxon(r_vals, t_vals, alternative='two-sided').pvalue)
                except Exception:
                    pass

            particle_summary_records.append({
                'bead_name': bname,
                'diameter_um': dia,
                'n_particles_both_modes': len(sub),
                'pct_run_greater_tumble': float(np.mean(r_vals > t_vals) * 100.0),
                'mean_xi_run_um': float(np.mean(r_vals)),
                'sem_xi_run_um': float(np.std(r_vals, ddof=1) / np.sqrt(len(r_vals))) if len(r_vals) > 1 else 0.0,
                'median_xi_run_um': float(np.median(r_vals)),
                'mean_xi_tumble_um': float(np.mean(t_vals)),
                'sem_xi_tumble_um': float(np.std(t_vals, ddof=1) / np.sqrt(len(t_vals))) if len(t_vals) > 1 else 0.0,
                'median_xi_tumble_um': float(np.median(t_vals)),
                'mean_delta_xi_um': float(np.mean(d_vals)),
                'sem_delta_xi_um': float(np.std(d_vals, ddof=1) / np.sqrt(len(d_vals))) if len(d_vals) > 1 else 0.0,
                'mean_ratio_xi': float(np.mean(rat_vals)),
                'sem_ratio_xi': float(np.std(rat_vals, ddof=1) / np.sqrt(len(rat_vals))) if len(rat_vals) > 1 else 0.0,
                't_statistic': t_stat,
                'p_val_paired_ttest_greater': p_val_ttest_greater,
                'p_val_paired_ttest_two_sided': p_val_ttest_two_sided,
                'p_val_wilcoxon_greater': p_val_wilcoxon_greater,
                'p_val_wilcoxon_two_sided': p_val_wilcoxon_two_sided,
            })

        if particle_summary_records:
            df_part_summary = pd.DataFrame(particle_summary_records)
            csv_part_sum_path = output_dir / "per_particle_summary_statistics.csv"
            safe_save_csv(df_part_summary, csv_part_sum_path)
            print(f"  Saved per-particle summary stats: {csv_part_sum_path}")

    print("\n=================================================================")
    print(" All analyses and figure generations completed successfully!")
    print("=================================================================")


if __name__ == "__main__":
    main()
