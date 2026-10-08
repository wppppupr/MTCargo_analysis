#!/usr/bin/env python3
"""
plot_msd300_and_ccdf_r90.py

MSD(Δt = 300s) を第1軸（左軸: 青系）、
CCDF 90% 到達距離 Δr_90(Δt = 300s, P(R >= Δr) = 0.90) を第2軸（右軸: 赤系）にして、
横軸を貨物粒子径 2R_c [μm] とした 2軸プロットを作成・保存するスクリプト。

マーカーおよびラインカラーもそれぞれ青系・赤系に合わせています。
"""

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

BEADS_INFO = [
    {"name": "beads06um", "diameter_um": 0.63, "marker": "^"},
    {"name": "beads1um",  "diameter_um": 1.18, "marker": "o"},
    {"name": "beads3um",  "diameter_um": 3.37, "marker": "d"},
    {"name": "beads5um",  "diameter_um": 5.00, "marker": "p"},
    {"name": "beads7um",  "diameter_um": 7.24, "marker": "h"},
    {"name": "beads20um", "diameter_um": 20.0, "marker": "s"},
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


def save_figure_to_all(fig: plt.Figure, basename: str, out_dirs: List[Path]):
    """Figure を SVG および PNG 形式で指定全ディレクトリへ保存する。"""
    for d in out_dirs:
        try:
            d.mkdir(parents=True, exist_ok=True)
            fig.savefig(d / f"{basename}.svg", bbox_inches='tight')
            fig.savefig(d / f"{basename}.png", bbox_inches='tight')
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
    """
    ブートストラップ法 (復元抽出) により指定パーセンタイル値の標準誤差 (SEM) を算出する。
    """
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


def load_data_from_csv_or_raw(
    root_dir: Path,
    workspace_dir: Path,
    msd_time_s: float = 300.0,
    ccdf_time_s: float = 100.0,
    frame_interval: float = 4.0,
    scale: float = 0.11,
    alpha_threshold: float = 0.5
) -> pd.DataFrame:
    """
    MSD(msd_time_s) と CCDF Δr_90(ccdf_time_s) を算出または CSV から取得する。
    """
    import MSD
    tau_frames_ccdf = int(round(ccdf_time_s / frame_interval))
    records = []

    for item in BEADS_INFO:
        b_name = item["name"]
        d_um = item["diameter_um"]
        b_folder = root_dir / b_name
        files = sorted(glob.glob(str(b_folder / "*" / "*" / "beads_tracks.csv")))

        # 1. MSD の算出 (alpha > 0.5 フィルタリング)
        intervals = [int(frame_interval)] * len(files)
        pool_filtered, pool_raw, n_tot, n_filt = MSD.concat_pooled_particles_MSD(
            b_folder, intervals, scale=scale, alpha_threshold=alpha_threshold
        )
        msd_res = MSD.extract_msd_at_lag(pool_filtered, target_t=msd_time_s, tol=12.0)

        # 2. 長距離テール到達距離 Δr_90 (P(R <= r) = 0.90, 90th percentile) の算出
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
            # P(R <= r) = 0.90: 90% percentile (長距離テール / 上位10%の到達境界)
            r90_val = float(np.percentile(arr_disp, 90))
            r90_sem = calc_bootstrap_percentile_sem(arr_disp, percentile=90.0)
            
            r50_val = float(np.percentile(arr_disp, 50))  # median
            r50_sem = calc_bootstrap_percentile_sem(arr_disp, percentile=50.0)
            
            r10_val = float(np.percentile(arr_disp, 10))  # 10% percentile
            r10_sem = calc_bootstrap_percentile_sem(arr_disp, percentile=10.0)
            
            mean_d = float(np.mean(arr_disp))
            std_d = float(np.std(arr_disp, ddof=1)) if len(arr_disp) > 1 else 0.0
            sem_d = float(std_d / np.sqrt(len(arr_disp))) if len(arr_disp) > 1 else 0.0
        else:
            r90_val = r90_sem = r50_val = r50_sem = r10_val = r10_sem = mean_d = std_d = sem_d = np.nan

        records.append({
            "bead_name": b_name,
            "diameter_um": d_um,
            "msd_time_s": msd_time_s,
            "ccdf_time_s": ccdf_time_s,
            f"msd_{int(msd_time_s)}s_mean_um2": msd_res["mean"],
            f"msd_{int(msd_time_s)}s_sem_um2": msd_res["sem"],
            f"msd_{int(msd_time_s)}s_std_um2": msd_res["std"],
            f"msd_{int(msd_time_s)}s_median_um2": msd_res["median"],
            f"msd_{int(msd_time_s)}s_n_particles": msd_res["n"],
            f"delta_r_90pct_tail_{int(ccdf_time_s)}s_um": r90_val,
            f"delta_r_90pct_tail_{int(ccdf_time_s)}s_sem_um": r90_sem,
            f"delta_r_50pct_median_{int(ccdf_time_s)}s_um": r50_val,
            f"delta_r_50pct_median_{int(ccdf_time_s)}s_sem_um": r50_sem,
            f"delta_r_10pct_{int(ccdf_time_s)}s_um": r10_val,
            f"delta_r_10pct_{int(ccdf_time_s)}s_sem_um": r10_sem,
            "mean_displacement_um": mean_d,
            "sem_displacement_um": sem_d,
            "n_displacements": len(arr_disp)
        })

    return pd.DataFrame(records)


def plot_msd_and_ccdf_r90(
    df: pd.DataFrame,
    out_dirs: List[Path],
    msd_time_s: float = 300.0,
    ccdf_time_s: float = 100.0
):
    """
    第1軸 (左軸: 青系) MSD(msd_time_s), 第2軸 (右軸: 赤系) 長距離テール Δr_90 (P(R <= r)=0.90, ccdf_time_s) の 2軸プロットを作成する。
    両軸ともエラーバー（SEM）を表示します。
    """
    fig, ax_msd = plt.subplots(figsize=(8.4, 5.8))
    ax_r90 = ax_msd.twinx()

    d_vals = df["diameter_um"].to_numpy(dtype=float)
    msd_col_mean = f"msd_{int(msd_time_s)}s_mean_um2"
    msd_col_sem = f"msd_{int(msd_time_s)}s_sem_um2"
    r90_col = f"delta_r_90pct_tail_{int(ccdf_time_s)}s_um"
    r90_sem_col = f"delta_r_90pct_tail_{int(ccdf_time_s)}s_sem_um"

    msd_mean = df[msd_col_mean].to_numpy(dtype=float)
    msd_sem = df[msd_col_sem].to_numpy(dtype=float)
    r90_vals = df[r90_col].to_numpy(dtype=float)
    r90_sems = df[r90_sem_col].to_numpy(dtype=float) if r90_sem_col in df.columns else np.zeros_like(r90_vals)

    # カラーパレットの定義
    color_msd = '#1f77b4'  # 青系 (MSD)
    color_r90 = '#d62728'  # 赤系 (長距離テール Δr_90)

    # ---------------------------------------------------------
    # 第1軸 (左軸): MSD [um^2] (log scale, 青系)
    # ---------------------------------------------------------
    line1 = ax_msd.errorbar(
        d_vals, msd_mean, yerr=msd_sem,
        fmt='o-', color=color_msd, ecolor=color_msd, elinewidth=1.6,
        capsize=4.5, capthick=1.2, markersize=8.5, linewidth=2.0,
        label=rf'$\mathrm{{MSD}}(\Delta t = {int(msd_time_s)}\,\mathrm{{s}})$',
        zorder=4
    )

    # ---------------------------------------------------------
    # 第2軸 (右軸): 長距離テール到達距離 Δr_90 [um] (linear scale, 赤系) + エラーバー
    # ---------------------------------------------------------
    line2 = ax_r90.errorbar(
        d_vals, r90_vals, yerr=r90_sems,
        fmt='s--', color=color_r90, ecolor=color_r90, elinewidth=1.6,
        capsize=4.5, capthick=1.2, markersize=8.5, linewidth=2.0,
        label=rf'Tail Reach $\Delta r_{{90}}(\Delta t = {int(ccdf_time_s)}\,\mathrm{{s}})$',
        zorder=5
    )

    # ---------------------------------------------------------
    # 軸・スケール・目盛り設定
    # ---------------------------------------------------------
    # 横軸 (貨物粒子径)
    ax_msd.set_xscale('log')
    ax_msd.set_xlim(0.4, 28.0)
    ax_msd.xaxis.set_major_locator(ticker.FixedLocator([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0]))
    ax_msd.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax_msd.set_xlabel(r'Cargo Diameter $2R_c$ [$\mu\mathrm{m}$]', fontsize=12.5, fontweight='bold')

    # 第1軸 (左軸: MSD)
    ax_msd.set_yscale('log')
    ax_msd.set_ylim(0.05, 5000.0)
    ax_msd.set_ylabel(
        rf'MSD $\langle \Delta \boldsymbol{{r}}^2(\Delta t = {int(msd_time_s)}\,\mathrm{{s}}) \rangle$ [$\mu\mathrm{{m}}^2$]',
        fontsize=12.0, fontweight='bold', color=color_msd
    )
    ax_msd.tick_params(axis='y', colors=color_msd, which='both')
    ax_msd.spines['left'].set_color(color_msd)
    ax_msd.spines['left'].set_linewidth(1.3)

    # 第2軸 (右軸: 長距離テール到達距離 Δr_90)
    max_r90 = np.nanmax(r90_vals) if len(r90_vals) > 0 else 15.0
    y2_max = float(np.ceil(max_r90 * 1.25 / 5) * 5) if max_r90 > 5.0 else float(np.ceil(max_r90 * 1.25))
    ax_r90.set_ylim(0.0, max(y2_max, 10.0))
    step = 5.0 if y2_max > 15.0 else (2.0 if y2_max > 8.0 else 1.0)
    ax_r90.yaxis.set_major_locator(ticker.MultipleLocator(step))
    ax_r90.yaxis.set_minor_locator(ticker.MultipleLocator(step / 2.0))
    ax_r90.set_ylabel(
        rf'Reach Distance $\Delta r_{{90}}(\Delta t = {int(ccdf_time_s)}\,\mathrm{{s}})$ [$\mu\mathrm{{m}}$]' '\n'
        r'($P(R \leq \Delta r_{90}) = 0.90$, Top 10% Tail)',
        fontsize=12.0, fontweight='bold', color=color_r90
    )
    ax_r90.tick_params(axis='y', colors=color_r90, which='both')
    ax_r90.spines['right'].set_color(color_r90)
    ax_r90.spines['right'].set_linewidth(1.3)

    # グリッド
    ax_msd.grid(True, which='both', linestyle='--', alpha=0.35)

    # 凡例を1つに統合
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax_msd.legend(lines, labels, loc='upper right', framealpha=0.94, fontsize=9.5, edgecolor='#dddddd')

    # タイトル
    ax_msd.set_title(
        rf'Cargo MSD ($\Delta t = {int(msd_time_s)}\,\mathrm{{s}}$) & Tail $\Delta r_{{90}}$ ($\Delta t = {int(ccdf_time_s)}\,\mathrm{{s}}$) vs Diameter',
        fontsize=12.0, fontweight='bold', pad=12
    )

    plt.tight_layout()
    filename = f"msd{int(msd_time_s)}_and_ccdf_r90_{int(ccdf_time_s)}s_vs_diameter"
    save_figure_to_all(fig, filename, out_dirs)
    # デフォルト名でも保存
    save_figure_to_all(fig, "msd300_and_ccdf_r90_vs_diameter", out_dirs)
    plt.close(fig)
    print(f"[SAVED] Dual-Axis Plot: {filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot MSD (Axis 1, Blue) and CCDF Delta r_90 (Axis 2, Red) vs Cargo Diameter."
    )
    parser.add_argument("--root_dir", type=str, default=None, help="Root directory containing beads folders.")
    parser.add_argument("--msd_time_s", type=float, default=300.0, help="Target MSD lag time in seconds (default: 300.0s).")
    parser.add_argument("--ccdf_time_s", type=float, default=100.0, help="Target CCDF lag time in seconds (default: 100.0s).")
    parser.add_argument("--frame_interval", type=float, default=4.0, help="Frame interval in seconds (default: 4.0s).")
    args = parser.parse_args()

    root_dir = Path(args.root_dir) if args.root_dir else find_default_root()
    workspace_dir = Path(__file__).resolve().parent
    
    out_dirs = [
        workspace_dir / "figure" / "displacement_ccdf",
        workspace_dir / "figure" / "msd",
        root_dir / "figure" / "displacement_ccdf",
        root_dir / "figure" / "msd",
    ]
    for d in out_dirs:
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    print(f"Data Root: {root_dir}")
    print(f"Loading / Calculating MSD({int(args.msd_time_s)}s) and CCDF Δr_90({int(args.ccdf_time_s)}s)...")
    
    df_combined = load_data_from_csv_or_raw(
        root_dir=root_dir,
        workspace_dir=workspace_dir,
        msd_time_s=args.msd_time_s,
        ccdf_time_s=args.ccdf_time_s,
        frame_interval=args.frame_interval
    )

    csv_name = f"msd{int(args.msd_time_s)}_and_ccdf_r90_{int(args.ccdf_time_s)}s_summary.csv"
    save_csv_to_all(df_combined, csv_name, out_dirs)
    save_csv_to_all(df_combined, "msd300_and_ccdf_r90_summary.csv", out_dirs)

    print("\n=== Summary Table ===")
    cols_to_show = [
        "bead_name", "diameter_um",
        f"msd_{int(args.msd_time_s)}s_mean_um2", f"msd_{int(args.msd_time_s)}s_sem_um2",
        f"delta_r_90pct_tail_{int(args.ccdf_time_s)}s_um",
        f"delta_r_50pct_median_{int(args.ccdf_time_s)}s_um",
        f"delta_r_10pct_{int(args.ccdf_time_s)}s_um"
    ]
    print(df_combined[cols_to_show].to_string(index=False))

    print("\nPlotting dual-axis figure...")
    plot_msd_and_ccdf_r90(
        df_combined, out_dirs=out_dirs,
        msd_time_s=args.msd_time_s,
        ccdf_time_s=args.ccdf_time_s
    )
    print("Completed successfully!")


if __name__ == "__main__":
    main()
