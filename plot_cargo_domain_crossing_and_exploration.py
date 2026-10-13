#!/usr/bin/env python3
"""
plot_cargo_domain_crossing_and_exploration.py

微粒子貨物（Cargo Beads）の無次元スケール比 x = R_c / xi に対する
ドメイン突破能 (Normalized Reach) および 自己サイズ相対探索能 (Exploration Ratio)
の2パネルグラフ（1行2列）および個別粒子探索能グラフを作成・保存するスクリプト。

【物理パラメータと計算式】
1. 粒子半径 R_c:
   R_c = d / 2
2. 微小管相関長 xi:
   実験ごとに空間配向相関を求めて取得した xi の値（bg_angular_correlation_length_per_experiment.csv / summary.csv）を使用
3. 無次元スケール比 x (横軸):
   x = R_c / xi
4. パネル (a) 縦軸：ドメイン突破能 (Normalized Reach):
   Reach_90 = Delta r_90 / xi  (または Delta r_90 / xi_0)
5. パネル (b) 縦軸：自己サイズ相対探索能 (Exploration Ratio):
   Exploration = sqrt(<Delta r^2>) / R_c = sqrt(MSD) / R_c
   ※ 個別粒子プロット時: sqrt(MSD_i(300s)) / R_c を全粒子同一マーカーで描画

【プロット構成とデザイン仕様】
- 横並びの 2 パネル図 (1行2列, figsize=(12, 5), dpi=300)
- パネル (a):
  - 横軸: Scale Ratio x = R_c / xi (両対数, xlim: (0.02, 2.5))
  - 縦軸: Normalized Reach Delta r_90 / xi (両対数, ylim: (0.25, 5.5))
  - 基準線: 水平破線 Delta r = xi (値 1.0)
- パネル (b):
  - 横軸: Scale Ratio x = R_c / xi (両対数, xlim: (0.02, 2.5))
  - 縦軸: Exploration Ratio sqrt(MSD) / R_c (両対数, ylim: (0.1, 200.0))
  - 基準線: 水平破線 Delta r = R_c (値 1.0)
  - 全粒子（N=96）の個別データを同一マーカー ('o') でプロット（アンサンブル平均も比較表示）
- 単体図:
  - Self-Size Relative Exploration vs Scale Ratio (全粒子個別プロット)
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
from scipy import stats

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
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float], float, float]:
    """
    bg_angular_correlation_length_summary.csv および per_experiment.csv から
    条件別相関長 xi [um], 実験別相関長 xi [um], control の xi_0 を取得する。
    """
    possible_summary_paths = [
        workspace_dir / "figure" / "bg_angular_correlation" / "bg_angular_correlation_length_summary.csv",
        root_dir / "figure" / "bg_angular_correlation" / "bg_angular_correlation_length_summary.csv",
        Path("/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads/figure/bg_angular_correlation/bg_angular_correlation_length_summary.csv"),
        Path("/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/figure/bg_angular_correlation/bg_angular_correlation_length_summary.csv"),
    ]

    possible_per_exp_paths = [
        workspace_dir / "figure" / "bg_angular_correlation" / "bg_angular_correlation_length_per_experiment.csv",
        root_dir / "figure" / "bg_angular_correlation" / "bg_angular_correlation_length_per_experiment.csv",
        Path("/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads/figure/bg_angular_correlation/bg_angular_correlation_length_per_experiment.csv"),
        Path("/mnt/NAS-Ebanaru/sasaki/MTsingleBeads/figure/bg_angular_correlation/bg_angular_correlation_length_per_experiment.csv"),
    ]

    df_xi = None
    for p in possible_summary_paths:
        if p.exists():
            try:
                df_xi = pd.read_csv(p)
                break
            except Exception:
                pass

    df_per_exp = None
    for p in possible_per_exp_paths:
        if p.exists():
            try:
                df_per_exp = pd.read_csv(p)
                break
            except Exception:
                pass

    xi_dict = {}
    xi_0_mean = 10.743244
    xi_0_sem = 0.675331

    if df_xi is not None:
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

    exp_xi_dict = {}
    if df_per_exp is not None:
        for _, row in df_per_exp.iterrows():
            exp_name = str(row['exp_dir']).strip()
            if 'xi_um' in row and pd.notna(row['xi_um']):
                exp_xi_dict[exp_name] = float(row['xi_um'])

    return xi_dict, exp_xi_dict, xi_0_mean, xi_0_sem


def load_dataset(
    root_dir: Path,
    workspace_dir: Path,
    msd_time_s: float = 300.0,
    ccdf_time_s: float = 300.0,
    frame_interval: float = 4.0,
    scale: float = 0.11,
    alpha_threshold: float = 0.5,
    use_xi_control: bool = False,
    use_exp_xi: bool = True,
    bead_cache: Optional[Dict[str, Tuple]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Tuple]]:
    """
    アンサンブル統計 DataFrame および 個別粒子 DataFrame を構築して返す。
    bead_cache を渡すことで複数ラグタイム計算時のファイル再読込・MSD再計算を高速化する。
    """
    import MSD
    if bead_cache is None:
        bead_cache = {}

    tau_frames_ccdf = int(round(ccdf_time_s / frame_interval))
    tol = min(12.0, max(2.0, msd_time_s * 0.1))

    xi_dict, exp_xi_dict, xi_0_mean, xi_0_sem = load_correlation_lengths(root_dir, workspace_dir)

    ensemble_records = []
    individual_records = []

    for item in BEADS_INFO:
        b_name = item["name"]
        d_um = item["diameter_um"]
        r_c = d_um / 2.0
        color = item["color"]
        marker = item["marker"]

        b_folder = root_dir / b_name

        if b_name in bead_cache:
            files, pool_filtered, pool_raw, arr_disp = bead_cache[b_name]
        else:
            files = sorted(glob.glob(str(b_folder / "*" / "*" / "beads_tracks.csv")))
            intervals = [int(frame_interval)] * len(files)
            pool_filtered, pool_raw, n_tot, n_filt = MSD.concat_pooled_particles_MSD(
                b_folder, intervals, scale=scale, alpha_threshold=alpha_threshold
            )
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
            bead_cache[b_name] = (files, pool_filtered, pool_raw, arr_disp)

        # 1. MSD(msd_time_s) の算出
        msd_res = MSD.extract_msd_at_lag(pool_filtered, target_t=msd_time_s, tol=tol)
        msd_mean = msd_res["mean"]
        msd_sem = msd_res["sem"]

        # 2. CCDF 到達距離
        if len(arr_disp) > 0:
            r90_val = float(np.percentile(arr_disp, 90))
            r90_sem = calc_bootstrap_percentile_sem(arr_disp, percentile=90.0)
        else:
            r90_val = r90_sem = np.nan

        # 3. 相関長 xi の設定
        xi_info = xi_dict.get(b_name, {"xi_mean": xi_0_mean, "xi_sem": xi_0_sem})
        xi_cond = xi_0_mean if use_xi_control else xi_info["xi_mean"]
        xi_cond_err = xi_0_sem if use_xi_control else xi_info["xi_sem"]

        # 4. 個別粒子の MSD(msd_time_s) と Exploration Ratio の抽出
        for pid, grp in pool_filtered.groupby('unique_particle_id'):
            sub = grp[(grp['lag time'] >= msd_time_s - tol) & (grp['lag time'] <= msd_time_s + tol) & (grp['MSD'] > 0)]
            if not sub.empty:
                best_row = sub.iloc[(sub['lag time'] - msd_time_s).abs().argsort()[:1]]
                p_msd = float(best_row['MSD'].values[0])
                v_idx = int(best_row['video_idx'].values[0]) if 'video_idx' in best_row.columns else 0
                exp_name = Path(files[v_idx]).parent.name if (v_idx < len(files)) else b_name

                # 実験ごとの xi を取得（なければ条件平均値）
                if use_xi_control:
                    p_xi = xi_0_mean
                elif use_exp_xi and (exp_name in exp_xi_dict):
                    p_xi = exp_xi_dict[exp_name]
                else:
                    p_xi = xi_cond

                p_sqrt_msd = np.sqrt(p_msd)
                p_explor = p_sqrt_msd / r_c
                p_x_scale = r_c / p_xi
                p_alpha = float(best_row['alpha'].values[0]) if 'alpha' in best_row.columns else np.nan

                individual_records.append({
                    "unique_particle_id": pid,
                    "bead_name": b_name,
                    "diameter_um": d_um,
                    "radius_um": r_c,
                    "exp_dir": exp_name,
                    "xi_um": p_xi,
                    "scale_ratio_x": p_x_scale,
                    f"msd_{int(msd_time_s)}s_um2": p_msd,
                    "sqrt_msd_um": p_sqrt_msd,
                    "exploration_ratio": p_explor,
                    "alpha": p_alpha,
                    "color": color,
                    "marker": marker
                })

        # 5. アンサンブル統計
        x_scale = r_c / xi_cond
        x_scale_err = (r_c / (xi_cond ** 2)) * xi_cond_err

        norm_reach = r90_val / xi_cond
        norm_reach_err = norm_reach * np.sqrt((r90_sem / r90_val)**2 + (xi_cond_err / xi_cond)**2) if (r90_val > 0 and xi_cond > 0) else np.nan

        sqrt_msd = np.sqrt(msd_mean) if msd_mean > 0 else np.nan
        exploration_ratio = sqrt_msd / r_c
        exploration_err = (msd_sem / (2.0 * sqrt_msd * r_c)) if (sqrt_msd > 0 and r_c > 0) else np.nan

        ensemble_records.append({
            "bead_name": b_name,
            "diameter_um": d_um,
            "radius_um": r_c,
            "color": color,
            "marker": marker,
            "xi_um": xi_cond,
            "xi_sem_um": xi_cond_err,
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

    return pd.DataFrame(ensemble_records), pd.DataFrame(individual_records), bead_cache


def plot_dimensionless_scaling_2panel(
    df_ensemble: pd.DataFrame,
    df_individual: pd.DataFrame,
    out_dirs: List[Path],
    msd_time_s: float = 300.0,
    ccdf_time_s: float = 300.0,
    use_xi_control: bool = False,
    show_ensemble_in_b: bool = True
):
    """
    横並び 2 パネル図を作成する。
    パネル (a): Domain Crossing Ability vs Scale Ratio (アンサンブル)
    パネル (b): Self-Size Relative Exploration vs Scale Ratio (全粒子個別プロット: 同一マーカー 'o')
    """
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12.0, 5.2), dpi=300)

    # -------------------------------------------------------------------------
    # パネル (a): Domain Crossing Ability vs Scale Ratio
    # -------------------------------------------------------------------------
    for _, row in df_ensemble.iterrows():
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
    # パネル (b): Self-Size Relative Exploration vs Scale Ratio (全粒子個別プロット)
    # -------------------------------------------------------------------------
    # ユーザーリクエスト: 「粒子サイズごとのアンサンブルを取らずに全粒子のデータを同じマーカーでプロット」
    common_marker = 'o'
    
    # 1. 全粒子データを同一マーカー 'o' でプロット (粒子サイズ色で散布)
    for item in BEADS_INFO:
        b_name = item["name"]
        d_um = item["diameter_um"]
        color = item["color"]
        sub_p = df_individual[df_individual["bead_name"] == b_name]
        
        if not sub_p.empty:
            ax_b.scatter(
                sub_p["scale_ratio_x"],
                sub_p["exploration_ratio"],
                marker=common_marker,
                color=color,
                edgecolors='#222222',
                linewidths=0.6,
                s=42,
                alpha=0.75,
                zorder=3,
                label=rf'{d_um:.2f} $\mu\mathrm{{m}}$ ($N={len(sub_p)}$)'
            )

    # 2. オプション: アンサンブル代表値（平均±SEM）を白抜きシンボルで重ねる
    if show_ensemble_in_b:
        for _, row in df_ensemble.iterrows():
            x = row["scale_ratio_x"]
            x_err = row["scale_ratio_x_err"]
            y_b = row["exploration_ratio"]
            y_b_err = row["exploration_ratio_err"]
            color = row["color"]
            marker = row["marker"]

            ax_b.errorbar(
                x, y_b,
                xerr=x_err, yerr=y_b_err,
                fmt=marker, color=color, ecolor=color,
                elinewidth=1.6, capsize=4.5, capthick=1.1,
                markersize=9.5, markeredgecolor='#111111', markeredgewidth=1.2,
                alpha=0.95, zorder=5
            )

    # 基準線: 水平破線 Delta r = R_c (値 1.0)
    ax_b.axhline(
        y=1.0, color='#333333', linestyle='--', linewidth=1.5, alpha=0.8, zorder=2,
        label=r'Threshold $\sqrt{\langle \Delta r^2 \rangle} = R_c$ ($1.0$)'
    )

    # 軸設定 (b)
    ax_b.set_xscale('log')
    ax_b.set_yscale('log')
    ax_b.set_xlim(0.02, 2.5)
    ax_b.set_ylim(0.1, 250.0)

    ax_b.yaxis.set_major_locator(ticker.FixedLocator([0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]))
    ax_b.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax_b.yaxis.set_minor_locator(ticker.NullLocator())

    ax_b.xaxis.set_major_locator(ticker.FixedLocator([0.03, 0.1, 0.3, 1.0, 2.0]))
    ax_b.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax_b.xaxis.set_minor_locator(ticker.NullLocator())

    ax_b.set_xlabel(r'Scale Ratio $x = R_c / \xi$', fontsize=12.5, fontweight='bold')
    ax_b.set_ylabel(r'Exploration Ratio $\sqrt{\Delta r_i^2} / R_c$', fontsize=12.5, fontweight='bold')
    title_b = r'(b) Self-Size Relative Exploration vs Scale Ratio (All Particles)'
    ax_b.set_title(title_b, fontsize=13.0, fontweight='bold', pad=10)
    ax_b.grid(True, which='major', linestyle='--', alpha=0.35)
    ax_b.legend(loc='lower left', fontsize=8.2, framealpha=0.92, edgecolor='#dddddd')

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


def plot_single_individual_exploration(
    df_individual: pd.DataFrame,
    df_ensemble: pd.DataFrame,
    out_dirs: List[Path],
    msd_time_s: float = 300.0,
    show_ensemble: bool = False,
    single_color: bool = False
):
    """
    全粒子データを同じマーカー ('o') でプロットした単体グラフを作成・保存する。
    show_ensemble=False の場合、アンサンブル平均を描画せず全粒子の散布点のみをプロットする。
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.8), dpi=300)
    common_marker = 'o'

    if single_color:
        ax.scatter(
            df_individual["scale_ratio_x"],
            df_individual["exploration_ratio"],
            marker=common_marker,
            color='#1f77b4',
            edgecolors='#111111',
            linewidths=0.6,
            s=48,
            alpha=0.75,
            zorder=3,
            label=rf'Individual Particles ($N={len(df_individual)}$)'
        )
    else:
        for item in BEADS_INFO:
            b_name = item["name"]
            d_um = item["diameter_um"]
            color = item["color"]
            sub_p = df_individual[df_individual["bead_name"] == b_name]

            if not sub_p.empty:
                ax.scatter(
                    sub_p["scale_ratio_x"],
                    sub_p["exploration_ratio"],
                    marker=common_marker,
                    color=color,
                    edgecolors='#222222',
                    linewidths=0.6,
                    s=48,
                    alpha=0.75,
                    zorder=3,
                    label=rf'{d_um:.2f} $\mu\mathrm{{m}}$ ($N={len(sub_p)}$)'
                )

    if show_ensemble:
        for _, row in df_ensemble.iterrows():
            x = row["scale_ratio_x"]
            x_err = row["scale_ratio_x_err"]
            y_b = row["exploration_ratio"]
            y_b_err = row["exploration_ratio_err"]
            color = row["color"]
            marker = row["marker"]

            ax.errorbar(
                x, y_b,
                xerr=x_err, yerr=y_b_err,
                fmt=marker, color=color, ecolor=color,
                elinewidth=1.6, capsize=4.5, capthick=1.1,
                markersize=10.0, markeredgecolor='#111111', markeredgewidth=1.2,
                alpha=0.95, zorder=5
            )

    # 基準線: 水平破線 Delta r = R_c (値 1.0)
    ax.axhline(
        y=1.0, color='#333333', linestyle='--', linewidth=1.5, alpha=0.8, zorder=2,
        label=r'Self-size threshold $\sqrt{\Delta r^2} = R_c$ ($1.0$)'
    )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.02, 2.5)
    ax.set_ylim(0.1, 250.0)

    ax.yaxis.set_major_locator(ticker.FixedLocator([0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.yaxis.set_minor_locator(ticker.NullLocator())

    ax.xaxis.set_major_locator(ticker.FixedLocator([0.03, 0.1, 0.3, 1.0, 2.0]))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.xaxis.set_minor_locator(ticker.NullLocator())

    ax.set_xlabel(r'Scale Ratio $x = R_c / \xi$', fontsize=13.0, fontweight='bold')
    ax.set_ylabel(r'Exploration Ratio $\sqrt{\Delta r_i^2} / R_c$', fontsize=13.0, fontweight='bold')
    title_suffix = " (Raw Particles Only)" if not show_ensemble else ""
    ax.set_title(rf'Self-Size Relative Exploration vs Scale Ratio{title_suffix} ($\Delta t = {int(msd_time_s)}\,\mathrm{{s}}$)', fontsize=12.5, fontweight='bold', pad=12)
    ax.grid(True, which='major', linestyle='--', alpha=0.35)
    ax.legend(loc='lower left', fontsize=8.8, framealpha=0.92, edgecolor='#dddddd')

    plt.tight_layout()
    suffix = "_no_ensemble" if not show_ensemble else "_with_ensemble"
    if single_color:
        suffix += "_single_color"
    basename = f"cargo_individual_exploration_vs_scale_ratio{suffix}_tau{int(msd_time_s)}s"
    save_figure_to_all(fig, basename, out_dirs, dpi=300)
    if not show_ensemble and not single_color:
        save_figure_to_all(fig, "cargo_individual_exploration_vs_scale_ratio", out_dirs, dpi=300)
    plt.close(fig)
    print(f"[SAVED] Individual Exploration Single Plot: {basename} (.png & .svg)")


def plot_single_exploration_vs_cargo_size(
    df_individual: pd.DataFrame,
    df_ensemble: pd.DataFrame,
    out_dirs: List[Path],
    msd_time_s: float = 300.0,
    show_individual: bool = True,
    show_ensemble: bool = False,
    single_color: bool = False,
    use_radius: bool = False,
    fit_powerlaw: bool = False,
    custom_basename: Optional[str] = None
):
    """
    横軸をスケール比 x = R_c / xi ではなく、単純な粒子サイズ（直径 d または 半径 R_c）にしてプロットする。
    show_individual=False の場合、生データの点を消してアンサンブル平均のみを表示する。
    fit_powerlaw=True の場合、両対数をとった空間で線形フィッティングを行い、べき乗則（y = A * x^beta）を描画する。
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.8), dpi=300)
    common_marker = 'o'
    col_x = "radius_um" if use_radius else "diameter_um"
    var_symbol = "R_c" if use_radius else "d"
    x_label_tex = r'Cargo Radius $R_c\ [\mu\mathrm{m}]$' if use_radius else r'Cargo Diameter $d\ [\mu\mathrm{m}]$'
    name_suffix = "radius" if use_radius else "diameter"

    # 1. 生データのプロット (show_individual=True の場合のみ)
    if show_individual:
        if single_color:
            ax.scatter(
                df_individual[col_x],
                df_individual["exploration_ratio"],
                marker=common_marker,
                color='#1f77b4',
                edgecolors='#111111',
                linewidths=0.6,
                s=48,
                alpha=0.75,
                zorder=3,
                label=rf'Individual Particles ($N={len(df_individual)}$)'
            )
        else:
            for item in BEADS_INFO:
                b_name = item["name"]
                d_um = item["diameter_um"]
                r_um = d_um / 2.0
                color = item["color"]
                sub_p = df_individual[df_individual["bead_name"] == b_name]

                if not sub_p.empty:
                    label_txt = rf'{r_um:.2f} $\mu\mathrm{{m}}$ ($N={len(sub_p)}$)' if use_radius else rf'{d_um:.2f} $\mu\mathrm{{m}}$ ($N={len(sub_p)}$)'
                    ax.scatter(
                        sub_p[col_x],
                        sub_p["exploration_ratio"],
                        marker=common_marker,
                        color=color,
                        edgecolors='#222222',
                        linewidths=0.6,
                        s=48,
                        alpha=0.75,
                        zorder=3,
                        label=label_txt
                    )

    # 2. アンサンブル平均プロット
    if show_ensemble:
        for _, row in df_ensemble.iterrows():
            x = row[col_x]
            y_b = row["exploration_ratio"]
            y_b_err = row["exploration_ratio_err"]
            color = row["color"]
            marker = row["marker"]
            d_um = row["diameter_um"]
            r_um = row["radius_um"]
            pt_label = rf'{r_um:.2f} $\mu\mathrm{{m}}$' if use_radius else rf'{d_um:.2f} $\mu\mathrm{{m}}$'

            ax.errorbar(
                x, y_b,
                yerr=y_b_err,
                fmt=marker, color=color, ecolor=color,
                elinewidth=1.8, capsize=5.0, capthick=1.2,
                markersize=10.5, markeredgecolor='#111111', markeredgewidth=1.3,
                alpha=0.95, zorder=5,
                label=pt_label if not show_individual else None
            )

    # 3. べき乗則フィッティング (両軸対数をとってから線形回帰)
    if fit_powerlaw and show_ensemble and len(df_ensemble) >= 2:
        valid_mask = (df_ensemble[col_x] > 0) & (df_ensemble["exploration_ratio"] > 0)
        x_pts = df_ensemble.loc[valid_mask, col_x].values
        y_pts = df_ensemble.loc[valid_mask, "exploration_ratio"].values

        log_x = np.log10(x_pts)
        log_y = np.log10(y_pts)

        slope, intercept, r_val, p_val, std_err = stats.linregress(log_x, log_y)
        r2 = r_val ** 2
        prefactor = 10 ** intercept

        if use_radius:
            x_fit = np.logspace(np.log10(0.22), np.log10(13.0), 100)
        else:
            x_fit = np.logspace(np.log10(0.45), np.log10(25.0), 100)
        y_fit = prefactor * (x_fit ** slope)

        fit_label = (
            rf'Power-law fit: $y \propto {var_symbol}^{{{slope:.2f}}}$' + '\n' +
            rf'($y = {prefactor:.2f}\,{var_symbol}^{{{slope:.2f}}},\ R^2 = {r2:.4f}$)'
        )
        ax.plot(
            x_fit, y_fit,
            color='#d62728', linestyle='--', linewidth=2.0, alpha=0.9, zorder=4,
            label=fit_label
        )
        print(f"\n[Power-law Fit vs {col_x}]")
        print(f"  Exponent (slope): {slope:.4f} +/- {std_err:.4f}")
        print(f"  Prefactor A: {prefactor:.4f} (log10 intercept: {intercept:.4f})")
        print(f"  R^2: {r2:.4f}, p-value: {p_val:.4e}")

    # 基準線: 水平破線 Delta r = R_c (値 1.0)
    ax.axhline(
        y=1.0, color='#333333', linestyle=':', linewidth=1.5, alpha=0.8, zorder=2,
        label=r'Threshold $\sqrt{\langle \Delta r^2 \rangle} = R_c$ ($1.0$)'
    )

    ax.set_xscale('log')
    ax.set_yscale('log')
    if use_radius:
        ax.set_xlim(0.2, 15.0)
        ax.xaxis.set_major_locator(ticker.FixedLocator([0.3, 0.5, 1.0, 2.0, 5.0, 10.0]))
    else:
        ax.set_xlim(0.4, 30.0)
        ax.xaxis.set_major_locator(ticker.FixedLocator([0.5, 1.0, 2.0, 5.0, 10.0, 20.0]))
    ax.set_ylim(0.1, 250.0)

    ax.yaxis.set_major_locator(ticker.FixedLocator([0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.yaxis.set_minor_locator(ticker.NullLocator())

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.xaxis.set_minor_locator(ticker.NullLocator())

    ax.set_xlabel(x_label_tex, fontsize=13.0, fontweight='bold')
    ylabel_tex = r'Exploration Ratio $\sqrt{\langle \Delta r^2 \rangle} / R_c$' if not show_individual else r'Exploration Ratio $\sqrt{\Delta r_i^2} / R_c$'
    ax.set_ylabel(ylabel_tex, fontsize=13.0, fontweight='bold')

    if not show_individual and fit_powerlaw:
        title_txt = rf'Self-Size Relative Exploration vs Cargo Size ($\Delta t = {int(msd_time_s)}\,\mathrm{{s}}$)'
    elif not show_ensemble:
        title_txt = rf'Self-Size Relative Exploration vs Cargo Size (Raw Particles Only) ($\Delta t = {int(msd_time_s)}\,\mathrm{{s}}$)'
    else:
        title_txt = rf'Self-Size Relative Exploration vs Cargo Size ($\Delta t = {int(msd_time_s)}\,\mathrm{{s}}$)'

    ax.set_title(title_txt, fontsize=12.5, fontweight='bold', pad=12)
    ax.grid(True, which='major', linestyle='--', alpha=0.35)
    ax.legend(loc='lower left', fontsize=8.8, framealpha=0.92, edgecolor='#dddddd')

    plt.tight_layout()

    if custom_basename:
        basename = custom_basename
        save_figure_to_all(fig, basename, out_dirs, dpi=300)
    else:
        suffix = "_no_ensemble" if not show_ensemble else "_with_ensemble"
        if not show_individual:
            suffix = "_ensemble_only"
        if single_color:
            suffix += "_single_color"
        if fit_powerlaw:
            suffix += "_powerlaw_fit"
        basename = f"cargo_individual_exploration_vs_{name_suffix}{suffix}_tau{int(msd_time_s)}s"
        save_figure_to_all(fig, basename, out_dirs, dpi=300)
        if not show_ensemble and not single_color and show_individual:
            save_figure_to_all(fig, f"cargo_individual_exploration_vs_{name_suffix}", out_dirs, dpi=300)

    plt.close(fig)
    print(f"[SAVED] Individual Exploration vs Cargo Size Plot: {basename} (.png & .svg)")


def plot_all_lags_comparison(
    all_ensemble_dfs: Dict[float, pd.DataFrame],
    out_dirs: List[Path],
    use_radius: bool = False
) -> pd.DataFrame:
    """
    全ラグタイム (4, 40, 80, 100, 300s) のアンサンブル平均点とべき乗則フィッティング曲線を
    1枚の比較グラフに重ね合わせて描画・保存する。
    """
    fig, ax = plt.subplots(figsize=(8.0, 6.0), dpi=300)
    col_x = "radius_um" if use_radius else "diameter_um"
    var_symbol = "R_c" if use_radius else "d"
    x_label_tex = r'Cargo Radius $R_c\ [\mu\mathrm{m}]$' if use_radius else r'Cargo Diameter $d\ [\mu\mathrm{m}]$'
    name_suffix = "radius" if use_radius else "diameter"

    cmap = plt.cm.viridis
    sorted_taus = sorted(all_ensemble_dfs.keys())
    norm_vals = np.linspace(0.10, 0.90, len(sorted_taus))
    tau_colors = {tau: cmap(v) for tau, v in zip(sorted_taus, norm_vals)}

    fit_summary = []

    for tau in sorted_taus:
        df_e = all_ensemble_dfs[tau]
        color = tau_colors[tau]
        marker = 'o'

        valid = (df_e[col_x] > 0) & (df_e["exploration_ratio"] > 0)
        x_pts = df_e.loc[valid, col_x].values
        y_pts = df_e.loc[valid, "exploration_ratio"].values
        log_x = np.log10(x_pts)
        log_y = np.log10(y_pts)
        slope, intercept, r_val, p_val, std_err = stats.linregress(log_x, log_y)
        r2 = r_val ** 2
        prefactor = 10 ** intercept

        fit_summary.append({
            "tau_s": tau,
            "slope_beta": slope,
            "slope_err": std_err,
            "prefactor_A": prefactor,
            "r2": r2,
            "p_val": p_val
        })

        ax.errorbar(
            df_e[col_x], df_e["exploration_ratio"],
            yerr=df_e["exploration_ratio_err"],
            fmt=marker, color=color, ecolor=color,
            elinewidth=1.6, capsize=4.0, capthick=1.0,
            markersize=8.0, markeredgecolor='#111111', markeredgewidth=1.0,
            alpha=0.9, zorder=5,
            label=rf'$\Delta t = {int(tau)}\,\mathrm{{s}}$ ($y \propto {var_symbol}^{{{slope:.2f}}},\ R^2={r2:.3f}$)'
        )

        if use_radius:
            x_fit = np.logspace(np.log10(0.25), np.log10(12.0), 100)
        else:
            x_fit = np.logspace(np.log10(0.5), np.log10(23.0), 100)
        y_fit = prefactor * (x_fit ** slope)
        ax.plot(x_fit, y_fit, color=color, linestyle='--', linewidth=1.8, alpha=0.8, zorder=4)

    # 閾値 1.0
    ax.axhline(
        y=1.0, color='#333333', linestyle=':', linewidth=1.5, alpha=0.8, zorder=2,
        label=r'Threshold $\sqrt{\langle \Delta r^2 \rangle} = R_c$ ($1.0$)'
    )

    ax.set_xscale('log')
    ax.set_yscale('log')
    if use_radius:
        ax.set_xlim(0.2, 15.0)
        ax.xaxis.set_major_locator(ticker.FixedLocator([0.3, 0.5, 1.0, 2.0, 5.0, 10.0]))
    else:
        ax.set_xlim(0.4, 30.0)
        ax.xaxis.set_major_locator(ticker.FixedLocator([0.5, 1.0, 2.0, 5.0, 10.0, 20.0]))
    ax.set_ylim(0.05, 300.0)

    ax.yaxis.set_major_locator(ticker.FixedLocator([0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.yaxis.set_minor_locator(ticker.NullLocator())

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.xaxis.set_minor_locator(ticker.NullLocator())

    ax.set_xlabel(x_label_tex, fontsize=13.0, fontweight='bold')
    ax.set_ylabel(r'Exploration Ratio $\sqrt{\langle \Delta r^2 \rangle} / R_c$', fontsize=13.0, fontweight='bold')
    ax.set_title(r'Cargo Exploration Scaling Across Multiple Lag Times $\Delta t$', fontsize=13.0, fontweight='bold', pad=12)
    ax.grid(True, which='major', linestyle='--', alpha=0.35)
    ax.legend(loc='lower left', fontsize=8.5, framealpha=0.92, edgecolor='#dddddd')

    plt.tight_layout()
    basename = f"cargo_exploration_vs_{name_suffix}_all_lags_powerlaw_fit"
    save_figure_to_all(fig, basename, out_dirs, dpi=300)
    plt.close(fig)
    print(f"[SAVED] All Lags Comparison Plot: {basename} (.png & .svg)")

    return pd.DataFrame(fit_summary)


def plot_all_lags_fit_lines_only(
    all_ensemble_dfs: Dict[float, pd.DataFrame],
    out_dirs: List[Path],
    use_radius: bool = False
) -> pd.DataFrame:
    """
    全ての lag time (4, 40, 80, 100, 300s) のアンサンブル平均データ点と
    べき乗則フィット直線を 1 つのグラフにまとめて描画・保存する。
    色には viridis を採用し、時間経過に伴って暗い色から明るい色へ変化させ、
    データ点とフィット線の色を同期させる。
    """
    fig, ax = plt.subplots(figsize=(8.4, 6.2), dpi=300)
    col_x = "radius_um" if use_radius else "diameter_um"
    var_symbol = "R_c" if use_radius else "d"
    x_label_tex = r'Cargo Radius $R_c\ [\mu\mathrm{m}]$' if use_radius else r'Cargo Diameter $d\ [\mu\mathrm{m}]$'
    name_suffix = "radius" if use_radius else "diameter"

    # viridis カラーマップの適用 (時間経過ごとに暗い色 -> 明るい色へ)
    # 4s (濃紺/紫) -> 40s -> 80s -> 100s -> 300s (明るい黄緑/黄色)
    # 白背景での視認性を考慮して 0.10 〜 0.90 の範囲を使用
    cmap = plt.cm.viridis
    sorted_taus = sorted(all_ensemble_dfs.keys())
    norm_vals = np.linspace(0.10, 0.90, len(sorted_taus))
    tau_colors = {tau: cmap(v) for tau, v in zip(sorted_taus, norm_vals)}

    fit_summary = []

    for tau in sorted_taus:
        df_e = all_ensemble_dfs[tau]
        color = tau_colors[tau]

        valid = (df_e[col_x] > 0) & (df_e["exploration_ratio"] > 0)
        x_pts = df_e.loc[valid, col_x].values
        y_pts = df_e.loc[valid, "exploration_ratio"].values
        log_x = np.log10(x_pts)
        log_y = np.log10(y_pts)
        slope, intercept, r_val, p_val, std_err = stats.linregress(log_x, log_y)
        r2 = r_val ** 2
        prefactor = 10 ** intercept

        fit_summary.append({
            "tau_s": tau,
            "slope_beta": slope,
            "slope_err": std_err,
            "prefactor_A": prefactor,
            "r2": r2,
            "p_val": p_val
        })

        # 1. データ点 (アンサンブル平均 ± SEM) を時間と同じ viridis カラーで描画
        ax.errorbar(
            df_e[col_x], df_e["exploration_ratio"],
            yerr=df_e["exploration_ratio_err"],
            fmt='o', color=color, ecolor=color,
            elinewidth=1.6, capsize=4.0, capthick=1.1,
            markersize=8.0, markeredgecolor='#222222', markeredgewidth=0.9,
            alpha=0.92, zorder=5
        )

        # 2. べき乗則フィット直線を同じ色で描画
        if use_radius:
            x_fit = np.logspace(np.log10(0.25), np.log10(12.0), 100)
        else:
            x_fit = np.logspace(np.log10(0.5), np.log10(23.0), 100)
        y_fit = prefactor * (x_fit ** slope)

        fit_label = (
            rf'$\Delta t = {int(tau)}\,\mathrm{{s}}$: ' +
            rf'$y = {prefactor:.2f}\,{var_symbol}^{{{slope:.2f}}}$ ($R^2 = {r2:.3f}$)'
        )
        ax.plot(
            x_fit, y_fit,
            color=color, linestyle='--', linewidth=2.2, alpha=0.95, zorder=4,
            label=fit_label
        )

    # 基準線: 水平破線 Delta r = R_c (値 1.0)
    ax.axhline(
        y=1.0, color='#333333', linestyle=':', linewidth=1.5, alpha=0.8, zorder=2,
        label=r'Self-size threshold $\sqrt{\langle \Delta r^2 \rangle} = R_c$ ($1.0$)'
    )

    ax.set_xscale('log')
    ax.set_yscale('log')
    if use_radius:
        ax.set_xlim(0.2, 15.0)
        ax.xaxis.set_major_locator(ticker.FixedLocator([0.3, 0.5, 1.0, 2.0, 5.0, 10.0]))
    else:
        ax.set_xlim(0.4, 30.0)
        ax.xaxis.set_major_locator(ticker.FixedLocator([0.5, 1.0, 2.0, 5.0, 10.0, 20.0]))
    ax.set_ylim(0.05, 300.0)

    ax.yaxis.set_major_locator(ticker.FixedLocator([0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.yaxis.set_minor_locator(ticker.NullLocator())

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.xaxis.set_minor_locator(ticker.NullLocator())

    ax.set_xlabel(x_label_tex, fontsize=13.0, fontweight='bold')
    ax.set_ylabel(r'Exploration Ratio $\sqrt{\langle \Delta r^2 \rangle} / R_c$', fontsize=13.0, fontweight='bold')
    ax.set_title(r'Power-Law Scaling Across Lag Times ($\Delta t = 4 \sim 300\,\mathrm{s}$)', fontsize=13.0, fontweight='bold', pad=12)
    ax.grid(True, which='major', linestyle='--', alpha=0.35)
    ax.legend(loc='lower left', fontsize=8.8, framealpha=0.94, edgecolor='#cccccc')

    plt.tight_layout()
    basename = f"cargo_exploration_vs_{name_suffix}_all_lags_fit_lines_only"
    save_figure_to_all(fig, basename, out_dirs, dpi=300)
    save_figure_to_all(fig, f"cargo_exploration_vs_{name_suffix}_all_lags_powerlaw_fit", out_dirs, dpi=300)
    plt.close(fig)
    print(f"[SAVED] All Lags Viridis Scaling Plot: {basename} (.png & .svg)")

    return pd.DataFrame(fit_summary)


def main():
    parser = argparse.ArgumentParser(
        description="Plot Self-Size Relative Exploration vs Cargo Size for Multiple Lag Times (4, 40, 80, 100, 300 s)."
    )
    parser.add_argument("--root_dir", type=str, default=None, help="Root directory containing beads folders.")
    parser.add_argument("--taus", type=str, default="4,40,80,100,300", help="Comma-separated lag times in seconds (default: '4,40,80,100,300').")
    parser.add_argument("--frame_interval", type=float, default=4.0, help="Frame interval in seconds (default: 4.0s).")
    parser.add_argument("--use_xi_control", action="store_true", help="Use control correlation length xi_0 for all beads.")
    parser.add_argument("--no_exp_xi", action="store_true", help="Use bead condition average xi instead of per-experiment xi.")
    args = parser.parse_args()

    root_dir = Path(args.root_dir) if args.root_dir else find_default_root()
    workspace_dir = Path(__file__).resolve().parent

    target_taus = [float(t.strip()) for t in args.taus.split(",") if t.strip()]

    # 基底出力ディレクトリ
    base_out_dirs = [
        workspace_dir / "figure" / "displacement_scaling",
        root_dir / "figure" / "displacement_scaling",
    ]

    print(f"Data Root: {root_dir}")
    print(f"Target Lag Times Δt: {target_taus} s")

    bead_cache = {}
    all_ensemble_dfs = {}

    for tau in target_taus:
        tau_int = int(tau)
        print(f"\n=======================================================")
        print(f"  Processing Lag Time Δt = {tau_int} s ...")
        print(f"=======================================================")

        df_ensemble, df_individual, bead_cache = load_dataset(
            root_dir=root_dir,
            workspace_dir=workspace_dir,
            msd_time_s=tau,
            ccdf_time_s=tau,
            frame_interval=args.frame_interval,
            use_xi_control=args.use_xi_control,
            use_exp_xi=not args.no_exp_xi,
            bead_cache=bead_cache
        )

        all_ensemble_dfs[tau] = df_ensemble

        # 各ラグタイム専用のサブフォルダ (tau_{tau}s) に保存
        tau_out_dirs = [
            d / f"tau_{tau_int}s" for d in base_out_dirs
        ]
        current_out_dirs = tau_out_dirs

        # CSV 保存
        save_csv_to_all(df_ensemble, f"cargo_domain_crossing_and_exploration_summary_tau{tau_int}s.csv", current_out_dirs)
        save_csv_to_all(df_individual, f"cargo_individual_particles_exploration_tau{tau_int}s.csv", current_out_dirs)

        # 1. vs Cargo Diameter
        # (A) 生粒子データのみ
        plot_single_exploration_vs_cargo_size(
            df_individual,
            df_ensemble,
            out_dirs=current_out_dirs,
            msd_time_s=tau,
            show_individual=True,
            show_ensemble=False,
            single_color=False,
            use_radius=False
        )
        # (B) 生データ消去＋アンサンブル平均＋べき乗フィッティング
        plot_single_exploration_vs_cargo_size(
            df_individual,
            df_ensemble,
            out_dirs=current_out_dirs,
            msd_time_s=tau,
            show_individual=False,
            show_ensemble=True,
            single_color=False,
            use_radius=False,
            fit_powerlaw=True,
            custom_basename=f"cargo_individual_exploration_vs_diameter_with_ensemble_tau{tau_int}s"
        )
        # 別名
        plot_single_exploration_vs_cargo_size(
            df_individual,
            df_ensemble,
            out_dirs=current_out_dirs,
            msd_time_s=tau,
            show_individual=False,
            show_ensemble=True,
            single_color=False,
            use_radius=False,
            fit_powerlaw=True,
            custom_basename=f"cargo_exploration_vs_diameter_powerlaw_fit_tau{tau_int}s"
        )

        # 2. vs Cargo Radius
        # (A) 生粒子データのみ
        plot_single_exploration_vs_cargo_size(
            df_individual,
            df_ensemble,
            out_dirs=current_out_dirs,
            msd_time_s=tau,
            show_individual=True,
            show_ensemble=False,
            single_color=False,
            use_radius=True
        )
        # (B) 生データ消去＋アンサンブル平均＋べき乗フィッティング
        plot_single_exploration_vs_cargo_size(
            df_individual,
            df_ensemble,
            out_dirs=current_out_dirs,
            msd_time_s=tau,
            show_individual=False,
            show_ensemble=True,
            single_color=False,
            use_radius=True,
            fit_powerlaw=True,
            custom_basename=f"cargo_individual_exploration_vs_radius_with_ensemble_tau{tau_int}s"
        )

    # 3. 全ラグタイムの重ね合わせ比較プロット (all_lags フォルダおよび親フォルダに保存)
    print(f"\n=======================================================")
    print(f"  Generating All-Lags Comparison Plots & Summary ...")
    print(f"=======================================================")
    all_lags_out_dirs = [d / "all_lags" for d in base_out_dirs] + base_out_dirs

    # (A) フィット線のみをまとめたグラフ (Fit Lines Only)
    fit_summary_df = plot_all_lags_fit_lines_only(all_ensemble_dfs, all_lags_out_dirs, use_radius=False)
    plot_all_lags_fit_lines_only(all_ensemble_dfs, all_lags_out_dirs, use_radius=True)

    # (B) データ点＋フィット線の比較グラフ
    plot_all_lags_comparison(all_ensemble_dfs, all_lags_out_dirs, use_radius=False)
    plot_all_lags_comparison(all_ensemble_dfs, all_lags_out_dirs, use_radius=True)

    save_csv_to_all(fit_summary_df, "cargo_exploration_powerlaw_fit_summary_all_lags.csv", all_lags_out_dirs)

    print("\n=== Power-law Fit Summary Across Lag Times ===")
    print(fit_summary_df.to_string(index=False))
    print("\nCompleted successfully!")


if __name__ == "__main__":
    main()
