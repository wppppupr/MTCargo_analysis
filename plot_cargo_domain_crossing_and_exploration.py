#!/usr/bin/env python3
"""
plot_cargo_domain_crossing_and_exploration.py

微粒子貨物（Cargo Beads）の無次元スケール比 x = R_c / xi に対する
ドメイン突破能 (Normalized Reach) および 自己サイズ相対探索能 (Exploration Ratio)
の2パネルグラフ（1行2列）を作成・保存するスクリプト。

【物理パラメータと計算式】
1. 粒子半径 R_c:
   R_c = d / 2
2. 微小管相関長 xi:
   実験ごとに空間配向相関を求めて取得した xi の値（bg_angular_correlation_length_summary.csv）を使用
3. 無次元スケール比 x (横軸):
   x = R_c / xi
4. パネル (a) 縦軸：ドメイン突破能 (Normalized Reach):
   Reach_90 = Delta r_90 / xi  (または Delta r_90 / xi_0)
5. パネル (b) 縦軸：自己サイズ相対探索能 (Exploration Ratio):
   Exploration = sqrt(<Delta r^2>) / R_c = sqrt(MSD) / R_c

【プロット構成とデザイン仕様】
- 横並びの 2 パネル図 (1行2列, figsize=(12, 5), dpi=300)
- パネル (a):
  - 横軸: Scale Ratio x = R_c / xi (両対数, xlim: (0.02, 1.2))
  - 縦軸: Normalized Reach Delta r_90 / xi (両対数, ylim: (0.3, 4.0))
  - 基準線: 水平破線 Delta r = xi (値 1.0)
- パネル (b):
  - 横軸: Scale Ratio x = R_c / xi (両対数, xlim: (0.02, 1.2))
  - 縦軸: Exploration Ratio sqrt(<Delta r^2>) / R_c (両対数, ylim: (0.2, 80.0))
  - 基準線: 水平破線 Delta r = R_c (値 1.0)
"""

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# libsディレクトリのインポート
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from libs import cal_vel as cv
from libs import displacement as dpm

# スタイルの適用
style_path = current_dir / 'libs' / 'my_style.mplstyle'
if style_path.exists():
    plt.style.use(str(style_path))
    style_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
else:
    style_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# ビーズ条件設定 (MSD.py に準拠)
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
    Path('/mnt/NAS-Ebanaru/sasaki/MTsingleBeads'),
    Path('/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads'),
    Path('/Volumes/data/Sasaki/MTSingleBeads'),
]


def find_default_root() -> Path:
    """有効なデータルートディレクトリを自動検出する。"""
    for r in POSSIBLE_ROOTS:
        if r.exists():
            for b in ['beads1um', 'beads06um', 'beads3um', 'beads5um', 'beads7um', 'beads20um']:
                if (r / b).exists() and len(list((r / b).glob('*/*beads_tracks.csv'))) > 0:
                    return r
    for r in POSSIBLE_ROOTS:
        if r.exists():
            return r
    return POSSIBLE_ROOTS[0]


def save_figure_to_all(fig: plt.Figure, basename: str, out_dirs: List[Path], dpi: int = 300):
    """Figure を SVG および PNG 形式で指定全ディレクトリへ高解像度保存する。"""
    for d in out_dirs:
        try:
            d.mkdir(parents=True, exist_ok=True)
            fig.savefig(d / f"{basename}.svg", bbox_inches='tight', dpi=dpi)
            fig.savefig(d / f"{basename}.png", bbox_inches='tight', dpi=dpi)
        except Exception as e:
            print(f"[WARNING] Failed to save {basename} to {d}: {e}")


def save_csv_to_all(df: pd.DataFrame, filename: str, out_dirs: List[Path]):
    """DataFrame を CSV 形式で指定全ディレクトリへ保存する。"""
    for d in out_dirs:
        try:
            d.mkdir(parents=True, exist_ok=True)
            df.to_csv(d / filename, index=False)
        except Exception as e:
            print(f"[WARNING] Failed to save {filename} to {d}: {e}")


def calc_bootstrap_percentile_sem(
    data: np.ndarray,
    percentile: float = 90.0,
    n_bootstraps: int = 500,
    random_state: int = 42
) -> float:
    """ブートストラップ法によりパーセンタイル値の標準誤差 (SEM) を算出する。"""
    n = len(data)
    if n < 2:
        return 0.0
    rng = np.random.default_rng(random_state)
    if n > 20000:
        boot_vals = [float(np.percentile(rng.choice(data, size=n, replace=True), percentile)) for _ in range(n_bootstraps)]
    else:
        boot_samples = rng.choice(data, size=(n_bootstraps, n), replace=True)
        boot_vals = np.percentile(boot_samples, percentile, axis=1)
    return float(np.std(boot_vals, ddof=1))


def load_correlation_lengths(
    root_dir: Path,
    workspace_dir: Path
) -> Tuple[Dict[str, Dict[str, float]], float, float]:
    """
    bg_angular_correlation_length_summary.csv から各ビーズ条件の相関長 xi [um] および control の xi_0 を取得する。
    """
    possible_paths = [
        workspace_dir / "figure" / "bg_angular_correlation" / "bg_angular_correlation_length_summary.csv",
        root_dir / "figure" / "bg_angular_correlation" / "bg_angular_correlation_length_summary.csv",
        Path("/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads/figure/bg_angular_correlation/bg_angular_correlation_length_summary.csv"),
        Path("/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/figure/bg_angular_correlation/bg_angular_correlation_length_summary.csv"),
    ]

    df_xi = None
    for p in possible_paths:
        if p.exists():
            try:
                df_xi = pd.read_csv(p)
                break
            except Exception:
                pass

    xi_dict = {}
    xi_0_mean = 10.743244
    xi_0_sem = 0.675331

    if df_xi is not None:
        # control の取得
        ctrl_row = df_xi[df_xi['bead_name'] == 'control']
        if not ctrl_row.empty:
            xi_0_mean = float(ctrl_row['xi_bg_mean_um'].iloc[0])
            xi_0_sem = float(ctrl_row['xi_bg_sem_um'].iloc[0])

        for item in BEADS_INFO:
            b_name = item["name"]
            row = df_xi[df_xi['bead_name'] == b_name]
            if not row.empty:
                xi_dict[b_name] = {
                    "xi_mean": float(row['xi_bg_mean_um'].iloc[0]),
                    "xi_sem": float(row['xi_bg_sem_um'].iloc[0]),
                    "xi_median": float(row['xi_bg_median_um'].iloc[0]) if 'xi_bg_median_um' in row.columns else float(row['xi_bg_mean_um'].iloc[0]),
                }
            else:
                xi_dict[b_name] = {"xi_mean": xi_0_mean, "xi_sem": xi_0_sem, "xi_median": xi_0_mean}
    else:
        print("[WARNING] Correlation length summary CSV not found. Using default values.")
        for item in BEADS_INFO:
            xi_dict[item["name"]] = {"xi_mean": 6.0, "xi_sem": 0.5, "xi_median": 6.0}

    return xi_dict, xi_0_mean, xi_0_sem


def load_dataset(
    root_dir: Path,
    workspace_dir: Path,
    msd_time_s: float = 300.0,
    ccdf_time_s: float = 300.0,
    frame_interval: float = 4.0,
    scale: float = 0.11,
    alpha_threshold: float = 0.5,
    use_xi_control: bool = False
) -> pd.DataFrame:
    """
    MSD, CCDF 到達距離 Delta r_90, および微小管相関長 xi を統合したデータセットを構築する。
    """
    import MSD
    tau_frames_ccdf = int(round(ccdf_time_s / frame_interval))
    
    xi_dict, xi_0_mean, xi_0_sem = load_correlation_lengths(root_dir, workspace_dir)

    records = []

    for item in BEADS_INFO:
        b_name = item["name"]
        d_um = item["diameter_um"]
        r_c = d_um / 2.0
        color = item["color"]
        marker = item["marker"]

        b_folder = root_dir / b_name
        files = sorted(glob.glob(str(b_folder / "*" / "*" / "beads_tracks.csv")))

        # 1. MSD(msd_time_s) の算出 (alpha > 0.5 フィルタリング)
        intervals = [int(frame_interval)] * len(files)
        pool_filtered, pool_raw, n_tot, n_filt = MSD.concat_pooled_particles_MSD(
            b_folder, intervals, scale=scale, alpha_threshold=alpha_threshold
        )
        msd_res = MSD.extract_msd_at_lag(pool_filtered, target_t=msd_time_s, tol=12.0)
        msd_mean = msd_res["mean"]
        msd_sem = msd_res["sem"]

        # 2. CCDF 長距離到達距離 Δr_90 (P(R <= r) = 0.90) の算出
        all_disps = []
        for f in files:
            try:
                df = pd.read_csv(f)
                disp = dpm.calc_displacement_magnitudes(df, tau=tau_frames_ccdf, scale=scale, component='norm')
                disp = disp[np.isfinite(disp) & (disp > 0)]
                all_disps.extend(disp)
            except Exception:
                pass
        arr_disp = np.array(all_disps, dtype=float)

        if len(arr_disp) > 0:
            r90_val = float(np.percentile(arr_disp, 90))
            r90_sem = calc_bootstrap_percentile_sem(arr_disp, percentile=90.0)
        else:
            r90_val = r90_sem = np.nan

        # 3. 相関長 xi の設定 (ビーズ条件固有 or control)
        xi_info = xi_dict.get(b_name, {"xi_mean": xi_0_mean, "xi_sem": xi_0_sem})
        xi_val = xi_0_mean if use_xi_control else xi_info["xi_mean"]
        xi_err = xi_0_sem if use_xi_control else xi_info["xi_sem"]

        # 4. 無次元量の計算
        # (1) スケール比 x = R_c / xi
        x_scale = r_c / xi_val
        x_scale_err = (r_c / (xi_val ** 2)) * xi_err  # 誤差伝播

        # (2) ドメイン突破能 Normalized Reach = Delta r_90 / xi (または / xi_0)
        norm_reach = r90_val / xi_val
        norm_reach_err = norm_reach * np.sqrt((r90_sem / r90_val)**2 + (xi_err / xi_val)**2) if (r90_val > 0 and xi_val > 0) else np.nan

        # (3) 自己サイズ相対探索能 Exploration Ratio = sqrt(MSD) / R_c
        sqrt_msd = np.sqrt(msd_mean) if msd_mean > 0 else np.nan
        exploration_ratio = sqrt_msd / r_c
        # 誤差伝播: d(sqrt(MSD))/d(MSD) = 1 / (2*sqrt(MSD))
        exploration_err = (msd_sem / (2.0 * sqrt_msd * r_c)) if (sqrt_msd > 0 and r_c > 0) else np.nan

        records.append({
            "bead_name": b_name,
            "diameter_um": d_um,
            "radius_um": r_c,
            "color": color,
            "marker": marker,
            "xi_um": xi_val,
            "xi_sem_um": xi_err,
            "xi_0_mean_um": xi_0_mean,
            "scale_ratio_x": x_scale,
            "scale_ratio_x_err": x_scale_err,
            "msd_time_s": msd_time_s,
            "msd_mean_um2": msd_mean,
            "msd_sem_um2": msd_sem,
            "sqrt_msd_um": sqrt_msd,
            "exploration_ratio": exploration_ratio,
            "exploration_ratio_err": exploration_err,
            "ccdf_time_s": ccdf_time_s,
            "delta_r_90_um": r90_val,
            "delta_r_90_sem_um": r90_sem,
            "normalized_reach_90": norm_reach,
            "normalized_reach_90_err": norm_reach_err,
            "n_particles": msd_res["n"],
            "n_displacements": len(arr_disp),
        })

    return pd.DataFrame(records)


def plot_dimensionless_scaling_2panel(
    df: pd.DataFrame,
    out_dirs: List[Path],
    msd_time_s: float = 300.0,
    ccdf_time_s: float = 300.0,
    use_xi_control: bool = False
):
    """
    要件に基づいた横並び 2 パネル図 (1行2列, figsize=(12, 5), dpi=300) を作成する。
    """
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12.0, 5.2), dpi=300)

    # -------------------------------------------------------------------------
    # パネル (a): Domain Crossing Ability vs Scale Ratio
    # -------------------------------------------------------------------------
    for _, row in df.iterrows():
        x = row["scale_ratio_x"]
        x_err = row["scale_ratio_x_err"]
        y_a = row["normalized_reach_90"]
        y_a_err = row["normalized_reach_90_err"]
        d_um = row["diameter_um"]
        color = row["color"]
        marker = row["marker"]

        ax_a.errorbar(
            x, y_a,
            xerr=x_err, yerr=y_a_err,
            fmt=marker, color=color, ecolor=color,
            elinewidth=1.5, capsize=4.5, capthick=1.1,
            markersize=9.5, alpha=0.92,
            label=rf'{d_um:.2f} $\mu\mathrm{{m}}$'
        )

    # 基準線: 水平破線 Delta r = xi (値 1.0)
    ax_a.axhline(
        y=1.0, color='#333333', linestyle='--', linewidth=1.5, alpha=0.8,
        label=r'Threshold $\Delta r_{90} = \xi$ ($1.0$)'
    )

    # 軸設定 (a)
    ax_a.set_xscale('log')
    ax_a.set_yscale('log')
    ax_a.set_xlim(0.02, 2.5)
    ax_a.set_ylim(0.25, 5.5)
    
    # 目盛り設定 (a)
    ax_a.yaxis.set_major_locator(ticker.FixedLocator([0.3, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]))
    ax_a.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax_a.yaxis.set_minor_locator(ticker.NullLocator())

    ax_a.xaxis.set_major_locator(ticker.FixedLocator([0.03, 0.1, 0.3, 1.0, 2.0]))
    ax_a.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax_a.xaxis.set_minor_locator(ticker.NullLocator())

    ax_a.set_xlabel(r'Scale Ratio $x = R_c / \xi$', fontsize=12.5, fontweight='bold')
    ylabel_a = r'Normalized Reach $\Delta r_{90} / \xi_0$' if use_xi_control else r'Normalized Reach $\Delta r_{90} / \xi$'
    ax_a.set_ylabel(ylabel_a, fontsize=12.5, fontweight='bold')
    ax_a.set_title(r'(a) Domain Crossing Ability vs Scale Ratio', fontsize=13.0, fontweight='bold', pad=10)
    ax_a.grid(True, which='major', linestyle='--', alpha=0.35)
    ax_a.legend(loc='lower left', fontsize=9.0, framealpha=0.92, edgecolor='#dddddd')

    # -------------------------------------------------------------------------
    # パネル (b): Self-Size Relative Exploration vs Scale Ratio
    # -------------------------------------------------------------------------
    for _, row in df.iterrows():
        x = row["scale_ratio_x"]
        x_err = row["scale_ratio_x_err"]
        y_b = row["exploration_ratio"]
        y_b_err = row["exploration_ratio_err"]
        d_um = row["diameter_um"]
        color = row["color"]
        marker = row["marker"]

        ax_b.errorbar(
            x, y_b,
            xerr=x_err, yerr=y_b_err,
            fmt=marker, color=color, ecolor=color,
            elinewidth=1.5, capsize=4.5, capthick=1.1,
            markersize=9.5, alpha=0.92,
            label=rf'{d_um:.2f} $\mu\mathrm{{m}}$'
        )

    # 基準線: 水平破線 Delta r = R_c (値 1.0)
    ax_b.axhline(
        y=1.0, color='#333333', linestyle='--', linewidth=1.5, alpha=0.8,
        label=r'Threshold $\sqrt{\langle \Delta r^2 \rangle} = R_c$ ($1.0$)'
    )

    # 軸設定 (b)
    ax_b.set_xscale('log')
    ax_b.set_yscale('log')
    ax_b.set_xlim(0.02, 2.5)
    ax_b.set_ylim(0.2, 90.0)

    # 目盛り設定 (b)
    ax_b.yaxis.set_major_locator(ticker.FixedLocator([0.3, 1.0, 3.0, 10.0, 30.0, 80.0]))
    ax_b.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax_b.yaxis.set_minor_locator(ticker.NullLocator())

    ax_b.xaxis.set_major_locator(ticker.FixedLocator([0.03, 0.1, 0.3, 1.0, 2.0]))
    ax_b.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax_b.xaxis.set_minor_locator(ticker.NullLocator())

    ax_b.set_xlabel(r'Scale Ratio $x = R_c / \xi$', fontsize=12.5, fontweight='bold')
    ax_b.set_ylabel(r'Exploration Ratio $\sqrt{\langle \Delta r^2 \rangle} / R_c$', fontsize=12.5, fontweight='bold')
    ax_b.set_title(r'(b) Self-Size Relative Exploration vs Scale Ratio', fontsize=13.0, fontweight='bold', pad=10)
    ax_b.grid(True, which='major', linestyle='--', alpha=0.35)
    ax_b.legend(loc='lower left', fontsize=9.0, framealpha=0.92, edgecolor='#dddddd')

    plt.suptitle(
        rf'Cargo Transport Regime Scaling Diagram ($\Delta t = {int(msd_time_s)}\,\mathrm{{s}}$)',
        fontsize=14.0, fontweight='bold', y=0.985
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    basename = f"cargo_domain_crossing_and_exploration_tau{int(msd_time_s)}s"
    save_figure_to_all(fig, basename, out_dirs, dpi=300)
    save_figure_to_all(fig, "cargo_domain_crossing_and_exploration_2panel", out_dirs, dpi=300)
    plt.close(fig)
    print(f"[SAVED] 2-Panel Scaling Plot: {basename} (.png & .svg)")


def main():
    parser = argparse.ArgumentParser(
        description="Plot Domain Crossing Ability (a) and Self-Size Relative Exploration (b) vs Scale Ratio x = R_c / xi."
    )
    parser.add_argument("--root_dir", type=str, default=None, help="Root directory containing beads folders.")
    parser.add_argument("--msd_time_s", type=float, default=300.0, help="Target MSD lag time in seconds (default: 300.0s).")
    parser.add_argument("--ccdf_time_s", type=float, default=300.0, help="Target CCDF lag time in seconds (default: 300.0s).")
    parser.add_argument("--frame_interval", type=float, default=4.0, help="Frame interval in seconds (default: 4.0s).")
    parser.add_argument("--use_xi_control", action="store_true", help="Use control correlation length xi_0 for all beads.")
    args = parser.parse_args()

    root_dir = Path(args.root_dir) if args.root_dir else find_default_root()
    workspace_dir = Path(__file__).resolve().parent

    out_dirs = [
        workspace_dir / "figure" / "displacement_scaling",
        workspace_dir / "figure" / "displacement_ccdf",
        workspace_dir / "figure" / "msd",
        root_dir / "figure" / "displacement_scaling",
        root_dir / "figure" / "displacement_ccdf",
    ]
    for d in out_dirs:
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    print(f"Data Root: {root_dir}")
    print(f"Loading data for MSD({int(args.msd_time_s)}s) & CCDF Δr_90({int(args.ccdf_time_s)}s)...")

    df = load_dataset(
        root_dir=root_dir,
        workspace_dir=workspace_dir,
        msd_time_s=args.msd_time_s,
        ccdf_time_s=args.ccdf_time_s,
        frame_interval=args.frame_interval,
        use_xi_control=args.use_xi_control
    )

    save_csv_to_all(df, f"cargo_domain_crossing_and_exploration_summary_tau{int(args.msd_time_s)}s.csv", out_dirs)
    save_csv_to_all(df, "cargo_domain_crossing_and_exploration_summary.csv", out_dirs)

    print("\n=== Summary Table ===")
    cols_to_print = [
        "bead_name", "diameter_um", "radius_um", "xi_um",
        "scale_ratio_x", "normalized_reach_90", "exploration_ratio"
    ]
    print(df[cols_to_print].to_string(index=False))

    print("\nPlotting 2-panel dimensionless scaling figure...")
    plot_dimensionless_scaling_2panel(
        df, out_dirs=out_dirs,
        msd_time_s=args.msd_time_s,
        ccdf_time_s=args.ccdf_time_s,
        use_xi_control=args.use_xi_control
    )
    print("Completed successfully!")


if __name__ == "__main__":
    main()
