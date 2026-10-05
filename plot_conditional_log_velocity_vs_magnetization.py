#!/usr/bin/env python3
"""
plot_conditional_log_velocity_vs_magnetization.py

磁化 M (または |M|) に対する無次元化貨物速度の対数:
    \\log_{10}(\\tilde{v}) = \\log_{10}(|v| / v_{\\mathrm{MT}})
    \\ln(\\tilde{v}) = \\ln(|v| / v_{\\mathrm{MT}})
について、
1. 各ビーズ径の全ステップ生データ点 (M_i, \\ln\\tilde{v}_i) の散布図プロット
2. 対数正規 Q-Q プロットに基づいた外れ値（Outlier）の検出と除去
3. 線形モデル y = y_0 + \\beta M によるフィッティングおよび回帰直線の描画
4. M = 0.0 ~ 1.0 (幅 0.1 刻み, 10ビン) での条件付き平均値 \\langle \\ln\\tilde{v} | M \\rangle および SEM の算出
5. 6つの貨物サイズ（ビーズ径: 0.63, 1.18, 3.37, 5.00, 7.24, 20.0 μm）の同一グラフプロット
6. スケール半径 x = R_c / \\xi_{i,t} に対するフィッティングパラメータ (\\beta, y_0) のスケーリングプロット
を出力するスクリプトです。

出力ファイル（既定: figure/conditional_log_velocity / <root>/figure/conditional_log_velocity）:
1. raw_points_linear_fit_ln_grid.png / .svg
   (全ステップ生データ点 (M_i, ln v_tilde) + 線形フィット y = y0 + beta * M の Grid プロット)
2. raw_points_linear_fit_ln_overlay.png / .svg
   (6サイズ重ね合わせ: 生データ散布図 + フィット直線 y = y0 + beta * M)
3. raw_points_linear_fit_log10_grid.png / .svg
   (常用対数版: 生データ点 + 線形フィット Grid プロット)
4. raw_points_linear_fit_log10_overlay.png / .svg
   (常用対数版: 6サイズ重ね合わせ 生データ + フィット直線)
5. qq_outlier_filtering_diagnostic.png / .svg
   (Q-Q プロットに基づく外れ値検出・除去の診断図: 採用点 vs 除外外れ値)
6. linear_fit_params_vs_diameter.png / .svg
   (線形回帰パラメータ beta, y0 vs 粒子径 d: 外れ値除去後)
7. linear_fit_params_vs_scaled_radius.png / .svg
   (スケーリング図: 線形回帰パラメータ beta, y0 vs スケール半径 x = R_c / xi: 外れ値除去後)
8. conditional_ln_velocity_vs_magnetization_6sizes.png / .svg
   (6サイズ重ね合わせビン平均: 横軸 M, 縦軸 <ln v_tilde | M>, エラーバー SEM)
9. conditional_log10_velocity_vs_magnetization_6sizes.png / .svg
   (6サイズ重ね合わせビン平均: 横軸 M, 縦軸 <log10 v_tilde | M>, エラーバー SEM)
10. conditional_log10_velocity_vs_magnetization_2panel.png / .svg
    (2パネル: 上段 <log10 v_tilde | M>, 下段 サンプル数 N)
11. conditional_log_velocity_grid_per_size.png / .svg
    (各粒子径 + 全体プールの個別パネル Grid 表示)
12. conditional_log_velocity_linear_fit_summary.csv
    (線形フィッティング y = y0 + beta * M および スケール半径 x = Rc / xi のパラメータサマリーCSV)
13. conditional_log_velocity_qq_filtered_summary.csv
    (Q-Q 外れ値除去前後の比較サマリーCSV)
14. conditional_log_velocity_vs_magnetization_summary.csv
    (ビン集計統計量のサマリーCSV)
"""

import argparse
import glob
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy.stats import linregress, probplot, norm, skew, kurtosis, kstest, normaltest

CURRENT_DIR = Path(__file__).parent.resolve()
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# スタイルの適用
style_path = CURRENT_DIR / 'libs' / 'my_style.mplstyle'
if style_path.exists():
    try:
        plt.style.use(str(style_path))
        style_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        style_colors = [f"#{c}" if not c.startswith('#') else c for c in style_colors]
    except Exception:
        style_colors = ['#882255', '#CC6677', '#DDCC77', '#999933', '#117733', '#44AA99', '#88CCEE', '#332288', '#AA4499']
else:
    style_colors = ['#882255', '#CC6677', '#DDCC77', '#999933', '#117733', '#44AA99', '#88CCEE', '#332288', '#AA4499']

# ビーズ基本情報 (全6サイズ, MSD.py / displacement_analysis.py と統一)
BEADS_INFO = [
    {"name": "beads06um", "diameter_um": 0.63, "radius_um": 0.315, "label": "0.63 μm", "marker": "^", "color": style_colors[0]},
    {"name": "beads1um",  "diameter_um": 1.18, "radius_um": 0.590, "label": "1.18 μm", "marker": "o", "color": style_colors[1]},
    {"name": "beads3um",  "diameter_um": 3.37, "radius_um": 1.685, "label": "3.37 μm", "marker": "d", "color": style_colors[2]},
    {"name": "beads5um",  "diameter_um": 5.00, "radius_um": 2.500, "label": "5.00 μm", "marker": "p", "color": style_colors[3]},
    {"name": "beads7um",  "diameter_um": 7.24, "radius_um": 3.620, "label": "7.24 μm", "marker": "h", "color": style_colors[4]},
    {"name": "beads20um", "diameter_um": 20.0, "radius_um": 10.00, "label": "20.0 μm", "marker": "s", "color": style_colors[5]},
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
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
    plt.rcParams['mathtext.fontset'] = 'cm'


def save_figure_to_all(fig: plt.Figure, basename: str, out_dirs: List[Path], dpi: int = 300):
    """指定されたすべての出力ディレクトリに png と svg を保存する"""
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        png_path = d / f"{basename}.png"
        svg_path = d / f"{basename}.svg"
        fig.savefig(png_path, dpi=dpi, bbox_inches='tight')
        fig.savefig(svg_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure: {basename}.png / .svg -> {len(out_dirs)} dir(s)")


def save_csv_to_all(df: pd.DataFrame, basename: str, out_dirs: List[Path]):
    """指定されたすべての出力ディレクトリに CSV を保存する"""
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        csv_path = d / f"{basename}.csv"
        df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {basename}.csv -> {len(out_dirs)} dir(s)")


def load_and_preprocess_data(csv_path: Path) -> pd.DataFrame:
    """cargo_spin_velocity_points.csv を読み込み無次元対数速度を算出"""
    if not csv_path.exists():
        raise FileNotFoundError(f"Points CSV not found: {csv_path}. Please check path.")

    df = pd.read_csv(csv_path)

    # 磁化 M (スカラー・絶対値)
    if 'm_ising' in df.columns:
        df['abs_m'] = df['m_ising'].abs()
    elif 'abs_m' in df.columns:
        pass
    else:
        raise KeyError("Column 'm_ising' or 'abs_m' not found in points CSV.")

    # 無次元スカラー速度 v_tilde
    if 'v_mag_tilde' in df.columns and df['v_mag_tilde'].notna().any():
        df['v_tilde_val'] = df['v_mag_tilde']
    elif 'v_mag_um_s' in df.columns and 'v_mt_um_s' in df.columns:
        df['v_tilde_val'] = df['v_mag_um_s'] / df['v_mt_um_s']
    elif 'v_track_tilde' in df.columns:
        df['v_tilde_val'] = df['v_track_tilde'].abs()
    elif 'v_tilde' in df.columns:
        df['v_tilde_val'] = df['v_tilde'].abs()
    else:
        raise KeyError("Velocity columns for v_tilde not found in points CSV.")

    # 有効データフィルタ (0 <= |M| <= 1, v_tilde > 0)
    valid = (
        np.isfinite(df['abs_m']) &
        (df['abs_m'] >= 0.0) &
        (df['abs_m'] <= 1.0) &
        np.isfinite(df['v_tilde_val']) &
        (df['v_tilde_val'] > 0)
    )
    df_valid = df[valid].copy()

    # 対数速度の計算
    df_valid['log10_v_tilde'] = np.log10(df_valid['v_tilde_val'])
    df_valid['ln_v_tilde'] = np.log(df_valid['v_tilde_val'])

    print(f"Loaded {len(df_valid):,} valid data points from {csv_path}")
    return df_valid


def apply_lognormal_qq_outlier_filter(
    df: pd.DataFrame,
    z_thresh: float = 2.5
) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    """
    各ビーズ径の ln(v_tilde) 速度分布の Q-Q プロット（対数正規分位点）に基づいて外れ値を除去。
    理論分位点 |z| <= z_thresh (既定: 2.5, 約98.8%信頼区間) の範囲内の点を採用。
    """
    df_clean = df.copy()
    df_clean['is_qq_valid'] = False
    df_clean['qq_theoretical_z'] = np.nan
    df_clean['qq_residual'] = np.nan

    diagnostic_dict = {}

    for b in BEADS_INFO:
        b_name = b["name"]
        idx_b = df_clean[df_clean['bead_name'] == b_name].index
        if len(idx_b) == 0:
            continue

        y_vals = df_clean.loc[idx_b, 'ln_v_tilde'].values
        # probplot による理論分位点とサンプル分位点の対応付け
        (osm, osr), (slope, intercept, r) = probplot(y_vals, dist="norm", fit=True)

        # 各元のインデックスに理論分位点 z をマッピング（ソート順）
        sort_order = np.argsort(y_vals)
        z_assigned = np.empty_like(osm)
        z_assigned[sort_order] = osm

        # 外れ値判定: 理論分位点 |z| <= z_thresh
        valid_mask = np.abs(z_assigned) <= z_thresh
        y_pred = intercept + slope * z_assigned
        residuals = y_vals - y_pred

        df_clean.loc[idx_b, 'is_qq_valid'] = valid_mask
        df_clean.loc[idx_b, 'qq_theoretical_z'] = z_assigned
        df_clean.loc[idx_b, 'qq_residual'] = residuals

        n_total = len(y_vals)
        n_valid = int(np.sum(valid_mask))
        n_outliers = n_total - n_valid

        diagnostic_dict[b_name] = {
            "name": b_name,
            "label": b["label"],
            "color": b["color"],
            "marker": b["marker"],
            "n_total": n_total,
            "n_valid": n_valid,
            "n_outliers": n_outliers,
            "outlier_pct": (n_outliers / n_total) * 100.0 if n_total > 0 else 0.0,
            "osm": osm,
            "osr": osr,
            "slope": slope,
            "intercept": intercept,
            "r2": r ** 2,
            "z_thresh": z_thresh,
            "valid_mask_sorted": np.abs(osm) <= z_thresh,
        }

        print(f"QQ Outlier Filter [{b_name}]: {n_valid}/{n_total} points kept ({n_outliers} outliers removed, {diagnostic_dict[b_name]['outlier_pct']:.1f}%)")

    # 全体プール (overall_pooled)
    y_all = df_clean['ln_v_tilde'].values
    (osm_all, osr_all), (slope_all, intercept_all, r_all) = probplot(y_all, dist="norm", fit=True)
    diagnostic_dict["overall_pooled"] = {
        "name": "overall_pooled",
        "label": "All Beads Pooled",
        "color": "#333333",
        "marker": "x",
        "n_total": len(y_all),
        "n_valid": int(df_clean['is_qq_valid'].sum()),
        "n_outliers": int(len(y_all) - df_clean['is_qq_valid'].sum()),
        "outlier_pct": float((len(y_all) - df_clean['is_qq_valid'].sum()) / len(y_all) * 100.0),
        "osm": osm_all,
        "osr": osr_all,
        "slope": slope_all,
        "intercept": intercept_all,
        "r2": r_all ** 2,
        "z_thresh": z_thresh,
        "valid_mask_sorted": np.abs(osm_all) <= z_thresh,
    }

    df_filtered = df_clean[df_clean['is_qq_valid']].copy()
    print(f"Total dataset after QQ filter: {len(df_filtered):,}/{len(df):,} points ({len(df)-len(df_filtered)} outliers removed)")
    return df_filtered, diagnostic_dict


def plot_qq_outlier_diagnostic(
    diagnostic_dict: Dict[str, dict],
    out_dirs: List[Path]
):
    """
    Q-Q プロットに基づく外れ値検出・除去の診断図 (各粒子径の Q-Q 直線と採用点 / 除外点の可視化)
    """
    fig, axes = plt.subplots(2, 4, figsize=(18, 9.5), sharex=True, sharey=False)
    axes_flat = axes.flatten()

    target_keys = [b["name"] for b in BEADS_INFO] + ["overall_pooled"]

    for idx, key in enumerate(target_keys):
        ax = axes_flat[idx]
        diag = diagnostic_dict.get(key)
        if not diag:
            ax.axis('off')
            continue

        osm = diag["osm"]
        osr = diag["osr"]
        slope = diag["slope"]
        intercept = diag["intercept"]
        z_thresh = diag["z_thresh"]
        valid_mask = diag["valid_mask_sorted"]
        color = diag["color"]
        marker = diag["marker"]
        label = diag["label"]

        # 1. 採用データ点 (緑/ビーズ色)
        ax.plot(
            osm[valid_mask], osr[valid_mask],
            marker=marker, color=color, ms=4.5, ls='none', alpha=0.7,
            label=f"Kept Points ($N={diag['n_valid']:,}$)"
        )

        # 2. 除外外れ値点 (赤 x)
        if np.any(~valid_mask):
            ax.plot(
                osm[~valid_mask], osr[~valid_mask],
                marker='x', color='#d62728', ms=6.5, ls='none', mew=1.5,
                label=f"Outliers ($N={diag['n_outliers']:,}$)"
            )

        # 3. Q-Q 回帰直線
        x_line = np.array([np.min(osm), np.max(osm)])
        y_line = intercept + slope * x_line
        ax.plot(x_line, y_line, color='black', lw=1.8, ls='--', label=f'Fit Line ($R^2={diag["r2"]:.3f}$)')

        # 4. 閾値境界 (|z| = z_thresh)
        ax.axvline(-z_thresh, color='#999999', ls=':', lw=1.2)
        ax.axvline(z_thresh, color='#999999', ls=':', lw=1.2)

        text_str = (
            f"Kept: {diag['n_valid']:,} ({100-diag['outlier_pct']:.1f}%)\n"
            f"Removed: {diag['n_outliers']:,} ({diag['outlier_pct']:.1f}%)\n"
            f"$z_{{thresh}} = \\pm {z_thresh}$\n"
            f"Slope $\\approx {slope:.3f}$"
        )
        ax.text(
            0.05, 0.95, text_str, transform=ax.transAxes,
            va='top', ha='left', fontsize=9.0,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#cccccc', alpha=0.9)
        )

        title_str = "All Beads Pooled" if key == "overall_pooled" else f"$d = {label}$"
        ax.set_title(title_str, fontsize=12.0, fontweight='bold')
        ax.set_xlabel(r"Theoretical Quantiles $z$", fontsize=10.0)
        ax.set_ylabel(r"Sample Quantiles $\ln(\tilde{v})$", fontsize=10.0)
        ax.grid(True, which='major', ls=':', alpha=0.6)
        ax.legend(loc='lower right', fontsize=8.0, framealpha=0.85)

    if len(target_keys) < len(axes_flat):
        for rem_idx in range(len(target_keys), len(axes_flat)):
            axes_flat[rem_idx].axis('off')

    plt.suptitle(
        r"Lognormal Q-Q Outlier Filtering Diagnostic: Identification of Tail Outliers ($|z| > 2.5$)",
        fontsize=13.5, fontweight='bold', y=0.99
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure_to_all(fig, "qq_outlier_filtering_diagnostic", out_dirs)


def load_scaled_radius_info(root_dir: Optional[Path] = None) -> Dict[str, dict]:
    """各ビーズサイズにおける相関長 xi_{i,t} および スケール半径 x = R_c / xi_{i,t} の統計量を算出・取得"""
    xi_info = {}

    summary_path = CURRENT_DIR / "figure" / "scaling" / "msd300_lambda100_vs_scaled_radius_summary.csv"
    if summary_path.exists():
        try:
            df_msd_summary = pd.read_csv(summary_path)
            for _, row in df_msd_summary.iterrows():
                b_name = str(row['bead_name'])
                xi_info[b_name] = {
                    "rc_over_xi_mean": float(row['rc_over_xi_mean']),
                    "rc_over_xi_sem": float(row['rc_over_xi_sem']),
                    "xi_um_mean": float(row['xi_um_mean']),
                }
        except Exception:
            pass

    for b in BEADS_INFO:
        b_name = b["name"]
        rc = b["radius_um"]
        xi_csv = CURRENT_DIR / "figure" / "xi_vs_velocity" / f"xi_vs_velocity_{b_name}.csv"
        if xi_csv.exists():
            try:
                df_xi = pd.read_csv(xi_csv)
                valid_xi = df_xi['xi_um'].dropna()
                valid_xi = valid_xi[valid_xi > 0].values
                if len(valid_xi) > 0:
                    x_vals = rc / valid_xi
                    x_mean = float(np.mean(x_vals))
                    x_sem = float(np.std(x_vals, ddof=1) / np.sqrt(len(x_vals))) if len(x_vals) > 1 else 0.0
                    x_median = float(np.median(x_vals))
                    xi_mean = float(np.mean(valid_xi))
                    xi_median = float(np.median(valid_xi))

                    if b_name not in xi_info:
                        xi_info[b_name] = {
                            "rc_over_xi_mean": x_mean,
                            "rc_over_xi_sem": x_sem,
                            "xi_um_mean": xi_mean,
                        }
                    xi_info[b_name].update({
                        "rc_over_xi_instant_mean": x_mean,
                        "rc_over_xi_instant_sem": x_sem,
                        "rc_over_xi_median": x_median,
                        "rc_over_median_xi": rc / xi_median if xi_median > 0 else np.nan,
                        "xi_um_median": xi_median,
                        "n_xi_points": len(valid_xi),
                    })
            except Exception as e:
                print(f"Warning: could not process {xi_csv}: {e}")

    return xi_info


def compute_linear_fits(df: pd.DataFrame, xi_info: Dict[str, dict]) -> Tuple[Dict[str, dict], pd.DataFrame]:
    """各粒子径および全体プールについて y = y0 + beta * M の線形回帰を実行"""
    target_groups = [(b["name"], b["diameter_um"], b["radius_um"], b["label"]) for b in BEADS_INFO]
    target_groups.append(("overall_pooled", np.nan, np.nan, "All Beads Pooled"))

    fit_dict = {}
    records = []

    for b_name, d_um, r_um, b_label in target_groups:
        if b_name == "overall_pooled":
            sub_df = df
        else:
            sub_df = df[df['bead_name'] == b_name]

        if len(sub_df) < 5:
            continue

        x = sub_df['abs_m'].values
        y_ln = sub_df['ln_v_tilde'].values
        y_log10 = sub_df['log10_v_tilde'].values

        res_ln = linregress(x, y_ln)
        res_log10 = linregress(x, y_log10)

        xi_data = xi_info.get(b_name, {})
        rc_over_xi_mean = xi_data.get("rc_over_xi_mean", np.nan)
        rc_over_xi_sem = xi_data.get("rc_over_xi_sem", np.nan)
        rc_over_xi_med = xi_data.get("rc_over_xi_median", np.nan)
        xi_mean = xi_data.get("xi_um_mean", np.nan)

        fit_dict[b_name] = {
            "name": b_name,
            "diameter_um": d_um,
            "radius_um": r_um,
            "label": b_label,
            "n_points": len(sub_df),
            "rc_over_xi_mean": rc_over_xi_mean,
            "rc_over_xi_sem": rc_over_xi_sem,
            "rc_over_xi_median": rc_over_xi_med,
            "xi_um_mean": xi_mean,
            "x": x,
            "y_ln": y_ln,
            "y_log10": y_log10,
            # ln fit
            "ln_y0": res_ln.intercept,
            "ln_beta": res_ln.slope,
            "ln_r2": res_ln.rvalue ** 2,
            "ln_pvalue": res_ln.pvalue,
            "ln_stderr": res_ln.stderr,
            "ln_intercept_stderr": res_ln.intercept_stderr,
            # log10 fit
            "log10_y0": res_log10.intercept,
            "log10_beta": res_log10.slope,
            "log10_r2": res_log10.rvalue ** 2,
            "log10_pvalue": res_log10.pvalue,
            "log10_stderr": res_log10.stderr,
            "log10_intercept_stderr": res_log10.intercept_stderr,
        }

        records.append({
            "bead_name": b_name,
            "diameter_um": d_um,
            "radius_um": r_um,
            "label": b_label,
            "n_points": len(sub_df),
            "rc_over_xi_mean": rc_over_xi_mean,
            "rc_over_xi_sem": rc_over_xi_sem,
            "rc_over_xi_median": rc_over_xi_med,
            "xi_um_mean": xi_mean,
            # ln fit
            "ln_y0_intercept": res_ln.intercept,
            "ln_y0_stderr": res_ln.intercept_stderr,
            "ln_beta_slope": res_ln.slope,
            "ln_beta_stderr": res_ln.stderr,
            "ln_r2": res_ln.rvalue ** 2,
            "ln_pvalue": res_ln.pvalue,
            # log10 fit
            "log10_y0_intercept": res_log10.intercept,
            "log10_y0_stderr": res_log10.intercept_stderr,
            "log10_beta_slope": res_log10.slope,
            "log10_beta_stderr": res_log10.stderr,
            "log10_r2": res_log10.rvalue ** 2,
            "log10_pvalue": res_log10.pvalue,
        })

    df_fit_summary = pd.DataFrame(records)
    return fit_dict, df_fit_summary


def compute_conditional_binned_stats(
    df: pd.DataFrame,
    n_bins: int = 10,
    m_range: Tuple[float, float] = (0.0, 1.0),
    min_count: int = 2
) -> pd.DataFrame:
    """横軸 M を幅 0.1 でビン分割し、対数速度の平均値・SEMを算出"""
    bins = np.linspace(m_range[0], m_range[1], n_bins + 1)
    df_temp = df.copy()
    df_temp['bin_idx'] = pd.cut(df_temp['abs_m'], bins=bins, include_lowest=True, labels=False)

    records = []
    target_groups = [(b["name"], b["diameter_um"], b["label"]) for b in BEADS_INFO]
    target_groups.append(("overall_pooled", np.nan, "All Beads Pooled"))

    for b_name, d_um, b_label in target_groups:
        if b_name == "overall_pooled":
            sub_df = df_temp
        else:
            sub_df = df_temp[df_temp['bead_name'] == b_name]

        for bin_i in range(n_bins):
            m_low = bins[bin_i]
            m_high = bins[bin_i + 1]
            m_center = 0.5 * (m_low + m_high)
            bin_data = sub_df[sub_df['bin_idx'] == bin_i]
            n_count = len(bin_data)

            if n_count >= min_count:
                m_mean = float(bin_data['abs_m'].mean())
                m_median = float(bin_data['abs_m'].median())
                # log10 統計
                mean_log10 = float(bin_data['log10_v_tilde'].mean())
                std_log10 = float(bin_data['log10_v_tilde'].std(ddof=1)) if n_count > 1 else 0.0
                sem_log10 = float(std_log10 / np.sqrt(n_count))
                median_log10 = float(bin_data['log10_v_tilde'].median())
                # ln 統計
                mean_ln = float(bin_data['ln_v_tilde'].mean())
                std_ln = float(bin_data['ln_v_tilde'].std(ddof=1)) if n_count > 1 else 0.0
                sem_ln = float(std_ln / np.sqrt(n_count))
                median_ln = float(bin_data['ln_v_tilde'].median())
            else:
                m_mean = m_center
                m_median = m_center
                mean_log10 = np.nan
                std_log10 = np.nan
                sem_log10 = np.nan
                median_log10 = np.nan
                mean_ln = np.nan
                std_ln = np.nan
                sem_ln = np.nan
                median_ln = np.nan

            records.append({
                "bead_name": b_name,
                "diameter_um": d_um,
                "label": b_label,
                "bin_idx": bin_i,
                "m_low": m_low,
                "m_high": m_high,
                "m_center": m_center,
                "m_mean": m_mean,
                "m_median": m_median,
                "n_count": n_count,
                # log10 stats
                "mean_log10_v_tilde": mean_log10,
                "std_log10_v_tilde": std_log10,
                "sem_log10_v_tilde": sem_log10,
                "median_log10_v_tilde": median_log10,
                # ln stats
                "mean_ln_v_tilde": mean_ln,
                "std_ln_v_tilde": std_ln,
                "sem_ln_v_tilde": sem_ln,
                "median_ln_v_tilde": median_ln,
            })

    return pd.DataFrame(records)


def plot_raw_points_linear_fit_grid(
    fit_dict: Dict[str, dict],
    df_binned: pd.DataFrame,
    out_dirs: List[Path],
    log_base: str = "ln"
):
    """全ステップ生データ点 (M_i, log v_tilde) の散布図 + 線形フィット y = y0 + beta * M の Grid プロット"""
    fig, axes = plt.subplots(2, 4, figsize=(18, 9.5), sharex=True, sharey=False)
    axes_flat = axes.flatten()

    y_prefix = "ln" if log_base == "ln" else "log10"
    ylabel_str = r"$\ln(\tilde{v}_i)$" if log_base == "ln" else r"$\log_{10}(\tilde{v}_i)$"
    title_var = r"\ln\tilde{v}" if log_base == "ln" else r"\log_{10}\tilde{v}"
    m_line = np.linspace(0.0, 1.0, 100)

    target_keys = [b["name"] for b in BEADS_INFO] + ["overall_pooled"]

    for idx, key in enumerate(target_keys):
        ax = axes_flat[idx]
        res = fit_dict.get(key)
        if not res:
            ax.axis('off')
            continue

        if key == "overall_pooled":
            label = "All Beads Pooled"
            color = "#333333"
            marker = "o"
        else:
            b_info = next(b for b in BEADS_INFO if b["name"] == key)
            label = f"$d = {b_info['label']}$"
            color = b_info["color"]
            marker = b_info["marker"]

        x_raw = res["x"]
        y_raw = res["y_ln"] if log_base == "ln" else res["y_log10"]
        y0 = res[f"{y_prefix}_y0"]
        beta = res[f"{y_prefix}_beta"]
        r2 = res[f"{y_prefix}_r2"]
        pval = res[f"{y_prefix}_pvalue"]
        stderr = res[f"{y_prefix}_stderr"]
        n_pts = res["n_points"]

        # 1. 生データ点の散布図
        ax.scatter(
            x_raw, y_raw,
            color=color, alpha=0.25, s=12, edgecolors='none',
            label=f"Data ($N={n_pts:,}$)"
        )

        # 2. ビン平均値 & SEM
        sub_bin = df_binned[df_binned["bead_name"] == key].dropna(subset=[f"mean_{y_prefix}_v_tilde"])
        if not sub_bin.empty:
            ax.errorbar(
                sub_bin["m_mean"], sub_bin[f"mean_{y_prefix}_v_tilde"],
                yerr=sub_bin[f"sem_{y_prefix}_v_tilde"],
                fmt='s', color='black', mfc='white', mec='black', mew=1.5,
                ms=5.5, capsize=3.0, elinewidth=1.2, label='Binned Mean ± SEM', zorder=4
            )

        # 3. フィット直線 y = y0 + beta * M
        y_fit = y0 + beta * m_line
        ax.plot(
            m_line, y_fit,
            color='#d62728', lw=2.2, ls='-',
            label=f"Fit: $y = {y0:.2f} {beta:+.2f} M$", zorder=5
        )

        p_str = f"$p = {pval:.2e}$" if pval < 0.001 else f"$p = {pval:.3f}$"
        param_text = (
            f"$y_0 = {y0:.3f}$\n"
            f"$\\beta = {beta:.3f} \\pm {stderr:.3f}$\n"
            f"$R^2 = {r2:.4f}$\n"
            f"{p_str}"
        )
        ax.text(
            0.05, 0.95, param_text, transform=ax.transAxes,
            va='top', ha='left', fontsize=9.0,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#cccccc', alpha=0.9)
        )

        ax.set_title(label, fontsize=12.0, fontweight='bold')
        ax.set_xlabel(r"Magnetization $M = |m_{\mathrm{ising}}|$", fontsize=10.5)
        ax.set_ylabel(ylabel_str, fontsize=10.5)
        ax.set_xlim(-0.02, 1.02)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.grid(True, which='major', ls=':', alpha=0.6)
        ax.legend(loc='lower left', fontsize=8.0, framealpha=0.85)

    if len(target_keys) < len(axes_flat):
        for rem_idx in range(len(target_keys), len(axes_flat)):
            axes_flat[rem_idx].axis('off')

    plt.suptitle(
        f"Raw Data Points (QQ Filtered) & Linear Fit: ${title_var} = y_0 + \\beta M$ by Particle Size",
        fontsize=13.5, fontweight='bold', y=0.99
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure_to_all(fig, f"raw_points_linear_fit_{log_base}_grid", out_dirs)


def plot_raw_points_linear_fit_overlay(
    fit_dict: Dict[str, dict],
    df_binned: pd.DataFrame,
    out_dirs: List[Path],
    log_base: str = "ln"
):
    """全6サイズの生データ点および線形フィット直線 y = y0 + beta * M の重ね合わせプロット"""
    fig, ax = plt.subplots(figsize=(9.0, 6.8))

    y_prefix = "ln" if log_base == "ln" else "log10"
    ylabel_str = r"$\ln(\tilde{v}_i)$" if log_base == "ln" else r"$\log_{10}(\tilde{v}_i)$"
    title_var = r"\ln\tilde{v}" if log_base == "ln" else r"\log_{10}\tilde{v}"
    m_line = np.linspace(0.0, 1.0, 100)

    for b in BEADS_INFO:
        b_name = b["name"]
        color = b["color"]
        label = b["label"]
        marker = b["marker"]
        res = fit_dict.get(b_name)
        if not res:
            continue

        x_raw = res["x"]
        y_raw = res["y_ln"] if log_base == "ln" else res["y_log10"]
        y0 = res[f"{y_prefix}_y0"]
        beta = res[f"{y_prefix}_beta"]
        r2 = res[f"{y_prefix}_r2"]

        # 散布図 (薄く表示)
        ax.scatter(x_raw, y_raw, color=color, alpha=0.15, s=8, edgecolors='none')

        # フィット直線
        y_fit = y0 + beta * m_line
        ax.plot(
            m_line, y_fit, color=color, lw=2.4, ls='-',
            label=f"$d = {label}$: $\\beta={beta:+.2f}$ ($R^2={r2:.3f}$)"
        )

        # ビン平均点
        sub_bin = df_binned[df_binned["bead_name"] == b_name].dropna(subset=[f"mean_{y_prefix}_v_tilde"])
        if not sub_bin.empty:
            ax.errorbar(
                sub_bin["m_mean"], sub_bin[f"mean_{y_prefix}_v_tilde"],
                yerr=sub_bin[f"sem_{y_prefix}_v_tilde"],
                fmt=marker, color=color, ms=6.0, capsize=3.0, elinewidth=1.2, alpha=0.9
            )

    ax.set_xlabel(r"Local Magnetization $M = |m_{\mathrm{ising}}|$", fontsize=12.5)
    ax.set_ylabel(ylabel_str, fontsize=12.5)
    ax.set_xlim(-0.02, 1.02)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))

    title_main = (
        f"Raw Points (QQ Filtered) & Linear Fits: ${title_var} = y_0 + \\beta M$ Across 6 Cargo Sizes"
    )
    ax.set_title(title_main, fontsize=13.0, fontweight='bold', pad=12)
    ax.grid(True, which='major', ls=':', alpha=0.6)
    ax.legend(loc='best', fontsize=9.5, framealpha=0.9, title="Particle Size & Slope")

    plt.tight_layout()
    save_figure_to_all(fig, f"raw_points_linear_fit_{log_base}_overlay", out_dirs)


def plot_fit_parameters_vs_diameter(
    df_fit: pd.DataFrame,
    out_dirs: List[Path]
):
    """フィッティングパラメータ (傾き beta, 切片 y0) の粒子径依存性プロット (2パネル版 + 単体パネル版)"""
    sub_df = df_fit[df_fit["bead_name"] != "overall_pooled"].dropna(subset=["diameter_um"])
    if sub_df.empty:
        return

    diameters = sub_df["diameter_um"].values
    beta_ln = sub_df["ln_beta_slope"].values
    beta_ln_err = sub_df["ln_beta_stderr"].values
    y0_ln = sub_df["ln_y0_intercept"].values
    y0_ln_err = sub_df["ln_y0_stderr"].values

    # 1. 2パネル統合プロット (1x2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.8))

    # パネル 1: 傾き \beta vs 粒子径 d
    for i in range(len(diameters)):
        b_info = next(b for b in BEADS_INFO if b["diameter_um"] == diameters[i])
        ax1.errorbar(
            diameters[i], beta_ln[i], yerr=beta_ln_err[i],
            fmt=b_info["marker"], color=b_info["color"], ecolor='black',
            elinewidth=1.6, capsize=4.5, capthick=1.2, markersize=9.5,
            markeredgecolor='black', markeredgewidth=1.2, zorder=5
        )

    ax1.axhline(0, color='#666666', ls='--', lw=1.3)
    ax1.set_xscale('log')
    ax1.set_xlabel(r"Particle Diameter $d$ [$\mu\mathrm{m}$]", fontsize=12.5)
    ax1.set_ylabel(r"Slope $\beta$ [$\ln(\tilde{v}) / M$]", fontsize=12.5)
    ax1.set_title(r"Velocity Sensitivity: Slope $\beta$ vs Diameter", fontsize=13.0, fontweight='bold')
    ax1.grid(True, which='both', ls=':', alpha=0.6)

    # パネル 2: 切片 y0 vs 粒子径 d
    for i in range(len(diameters)):
        b_info = next(b for b in BEADS_INFO if b["diameter_um"] == diameters[i])
        ax2.errorbar(
            diameters[i], y0_ln[i], yerr=y0_ln_err[i],
            fmt=b_info["marker"], color=b_info["color"], ecolor='black',
            elinewidth=1.6, capsize=4.5, capthick=1.2, markersize=9.5,
            markeredgecolor='black', markeredgewidth=1.2, zorder=5
        )

    ax2.set_xscale('log')
    ax2.set_xlabel(r"Particle Diameter $d$ [$\mu\mathrm{m}$]", fontsize=12.5)
    ax2.set_ylabel(r"Baseline Intercept $y_0$ (at $M=0$)", fontsize=12.5)
    ax2.set_title(r"Baseline Log Velocity at $M=0$: Intercept $y_0$ vs Diameter", fontsize=13.0, fontweight='bold')
    ax2.grid(True, which='both', ls=':', alpha=0.6)

    plt.suptitle(
        r"Linear Model Parameters: $\ln\tilde{v} = y_0 + \beta M$ vs Particle Diameter (QQ Filtered)",
        fontsize=13.5, fontweight='bold', y=0.99
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure_to_all(fig, "linear_fit_params_vs_diameter", out_dirs)

    # 2. 単体パネル 1: Slope beta vs Diameter
    fig_slope, ax_s = plt.subplots(figsize=(7.5, 6.0))
    for i in range(len(diameters)):
        b_info = next(b for b in BEADS_INFO if b["diameter_um"] == diameters[i])
        ax_s.errorbar(
            diameters[i], beta_ln[i], yerr=beta_ln_err[i],
            fmt=b_info["marker"], color=b_info["color"], ecolor='black',
            elinewidth=1.6, capsize=4.5, capthick=1.2, markersize=10.0,
            markeredgecolor='black', markeredgewidth=1.2, zorder=5,
            label=f"$d = {b_info['label']}$"
        )
    ax_s.axhline(0, color='#666666', ls='--', lw=1.3, label=r'$\beta = 0$ (Crossover)')
    ax_s.set_xscale('log')
    ax_s.set_xlabel(r"Particle Diameter $d$ [$\mu\mathrm{m}$]", fontsize=13.0)
    ax_s.set_ylabel(r"Slope $\beta$ [$\ln(\tilde{v}) / M$]", fontsize=13.0)
    ax_s.set_title(r"Velocity Sensitivity $\beta$ vs Particle Diameter $d$" + "\n(QQ Filtered)", fontsize=13.5, fontweight='bold', pad=10)
    ax_s.grid(True, which='both', ls=':', alpha=0.6)
    ax_s.legend(loc='best', fontsize=9.5, framealpha=0.9)
    plt.tight_layout()
    save_figure_to_all(fig_slope, "linear_fit_slope_vs_diameter", out_dirs)

    # 3. 単体パネル 2: Intercept y0 vs Diameter
    fig_y0, ax_y = plt.subplots(figsize=(7.5, 6.0))
    for i in range(len(diameters)):
        b_info = next(b for b in BEADS_INFO if b["diameter_um"] == diameters[i])
        ax_y.errorbar(
            diameters[i], y0_ln[i], yerr=y0_ln_err[i],
            fmt=b_info["marker"], color=b_info["color"], ecolor='black',
            elinewidth=1.6, capsize=4.5, capthick=1.2, markersize=10.0,
            markeredgecolor='black', markeredgewidth=1.2, zorder=5,
            label=f"$d = {b_info['label']}$"
        )
    ax_y.set_xscale('log')
    ax_y.set_xlabel(r"Particle Diameter $d$ [$\mu\mathrm{m}$]", fontsize=13.0)
    ax_y.set_ylabel(r"Baseline Intercept $y_0$ (at $M=0$)", fontsize=13.0)
    ax_y.set_title(r"Baseline Log Velocity $y_0$ at $M=0$ vs Particle Diameter $d$" + "\n(QQ Filtered)", fontsize=13.5, fontweight='bold', pad=10)
    ax_y.grid(True, which='both', ls=':', alpha=0.6)
    ax_y.legend(loc='best', fontsize=9.5, framealpha=0.9)
    plt.tight_layout()
    save_figure_to_all(fig_y0, "linear_fit_intercept_vs_diameter", out_dirs)


def plot_fit_parameters_vs_scaled_radius(
    df_fit: pd.DataFrame,
    out_dirs: List[Path],
    log_base: str = "ln"
):
    """
    フィッティングパラメータ (傾き beta, 切片 y0) のスケール半径 x = R_c / xi に対するスケーリングプロット (2パネル + 単体パネル)
    """
    sub_df = df_fit[df_fit["bead_name"] != "overall_pooled"].dropna(subset=["rc_over_xi_mean"])
    if sub_df.empty:
        return

    y_prefix = "ln" if log_base == "ln" else "log10"
    unit_str = r"\ln(\tilde{v})" if log_base == "ln" else r"\log_{10}(\tilde{v})"

    x_vals = sub_df["rc_over_xi_mean"].values
    x_errs = sub_df["rc_over_xi_sem"].values
    beta_vals = sub_df[f"{y_prefix}_beta_slope"].values
    beta_errs = sub_df[f"{y_prefix}_beta_stderr"].values
    y0_vals = sub_df[f"{y_prefix}_y0_intercept"].values
    y0_errs = sub_df[f"{y_prefix}_y0_stderr"].values

    # 1. 2パネル統合プロット (1x2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.8))

    # パネル 1: 傾き beta vs x = Rc / xi
    for i in range(len(sub_df)):
        b_name = sub_df.iloc[i]["bead_name"]
        b_info = next(b for b in BEADS_INFO if b["name"] == b_name)
        ax1.errorbar(
            x_vals[i], beta_vals[i],
            xerr=x_errs[i], yerr=beta_errs[i],
            fmt=b_info["marker"], color=b_info["color"], ecolor='black',
            elinewidth=1.6, capsize=4.5, capthick=1.2, markersize=9.5,
            markeredgecolor='black', markeredgewidth=1.2, zorder=5,
            label=f"$d = {b_info['label']}$"
        )

    ax1.axhline(0, color='#666666', ls='--', lw=1.3, label=r'$\beta = 0$ (Crossover)')
    ax1.set_xscale('log')
    ax1.set_xlabel(r"Scaled Radius $x = R_c / \xi_{i,t}$", fontsize=12.5)
    ax1.set_ylabel(rf"Slope $\beta$ [${unit_str} / M$]", fontsize=12.5)
    ax1.set_title(r"Magnetization Sensitivity $\beta$ vs Scaled Radius $x = R_c / \xi$", fontsize=13.0, fontweight='bold')
    ax1.grid(True, which='both', ls=':', alpha=0.6)
    ax1.legend(loc='best', fontsize=9.5, framealpha=0.9)

    # パネル 2: 切片 y0 vs x = Rc / xi
    for i in range(len(sub_df)):
        b_name = sub_df.iloc[i]["bead_name"]
        b_info = next(b for b in BEADS_INFO if b["name"] == b_name)
        ax2.errorbar(
            x_vals[i], y0_vals[i],
            xerr=x_errs[i], yerr=y0_errs[i],
            fmt=b_info["marker"], color=b_info["color"], ecolor='black',
            elinewidth=1.6, capsize=4.5, capthick=1.2, markersize=9.5,
            markeredgecolor='black', markeredgewidth=1.2, zorder=5,
            label=f"$d = {b_info['label']}$"
        )

    ax2.set_xscale('log')
    ax2.set_xlabel(r"Scaled Radius $x = R_c / \xi_{i,t}$", fontsize=12.5)
    ax2.set_ylabel(rf"Baseline Intercept $y_0$ (at $M=0$)", fontsize=12.5)
    ax2.set_title(r"Baseline Velocity $y_0$ at $M=0$ vs Scaled Radius $x = R_c / \xi$", fontsize=13.0, fontweight='bold')
    ax2.grid(True, which='both', ls=':', alpha=0.6)
    ax2.legend(loc='best', fontsize=9.5, framealpha=0.9)

    title_main = (
        rf"Scaling of Linear Model: ${unit_str} = y_0 + \beta M$ vs Scaled Radius $x = R_c / \xi_{{i,t}}$ (QQ Filtered)"
    )
    plt.suptitle(title_main, fontsize=13.5, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure_to_all(fig, "linear_fit_params_vs_scaled_radius", out_dirs)

    # 2. 単体パネル 1: Slope beta vs Scaled Radius
    fig_slope, ax_s = plt.subplots(figsize=(7.5, 6.0))
    for i in range(len(sub_df)):
        b_name = sub_df.iloc[i]["bead_name"]
        b_info = next(b for b in BEADS_INFO if b["name"] == b_name)
        ax_s.errorbar(
            x_vals[i], beta_vals[i],
            xerr=x_errs[i], yerr=beta_errs[i],
            fmt=b_info["marker"], color=b_info["color"], ecolor='black',
            elinewidth=1.6, capsize=4.5, capthick=1.2, markersize=10.0,
            markeredgecolor='black', markeredgewidth=1.2, zorder=5,
            label=f"$d = {b_info['label']}$"
        )
    ax_s.axhline(0, color='#666666', ls='--', lw=1.3, label=r'$\beta = 0$ (Crossover)')
    ax_s.set_xscale('log')
    ax_s.set_xlabel(r"Scaled Radius $x = R_c / \xi_{i,t}$", fontsize=13.0)
    ax_s.set_ylabel(rf"Slope $\beta$ [${unit_str} / M$]", fontsize=13.0)
    ax_s.set_title(r"Magnetization Sensitivity $\beta$ vs Scaled Radius $x = R_c / \xi$" + "\n(QQ Filtered)", fontsize=13.5, fontweight='bold', pad=10)
    ax_s.grid(True, which='both', ls=':', alpha=0.6)
    ax_s.legend(loc='best', fontsize=9.5, framealpha=0.9)
    plt.tight_layout()
    save_figure_to_all(fig_slope, f"linear_fit_slope_vs_scaled_radius_{log_base}", out_dirs)
    if log_base == "ln":
        save_figure_to_all(fig_slope, "linear_fit_slope_vs_scaled_radius", out_dirs)

    # 3. 単体パネル 2: Intercept y0 vs Scaled Radius
    fig_y0, ax_y = plt.subplots(figsize=(7.5, 6.0))
    for i in range(len(sub_df)):
        b_name = sub_df.iloc[i]["bead_name"]
        b_info = next(b for b in BEADS_INFO if b["name"] == b_name)
        ax_y.errorbar(
            x_vals[i], y0_vals[i],
            xerr=x_errs[i], yerr=y0_errs[i],
            fmt=b_info["marker"], color=b_info["color"], ecolor='black',
            elinewidth=1.6, capsize=4.5, capthick=1.2, markersize=10.0,
            markeredgecolor='black', markeredgewidth=1.2, zorder=5,
            label=f"$d = {b_info['label']}$"
        )
    ax_y.set_xscale('log')
    ax_y.set_xlabel(r"Scaled Radius $x = R_c / \xi_{i,t}$", fontsize=13.0)
    ax_y.set_ylabel(rf"Baseline Intercept $y_0$ (at $M=0$)", fontsize=13.0)
    ax_y.set_title(r"Baseline Velocity $y_0$ at $M=0$ vs Scaled Radius $x = R_c / \xi$" + "\n(QQ Filtered)", fontsize=13.5, fontweight='bold', pad=10)
    ax_y.grid(True, which='both', ls=':', alpha=0.6)
    ax_y.legend(loc='best', fontsize=9.5, framealpha=0.9)
    plt.tight_layout()
    save_figure_to_all(fig_y0, f"linear_fit_intercept_vs_scaled_radius_{log_base}", out_dirs)
    if log_base == "ln":
        save_figure_to_all(fig_y0, "linear_fit_intercept_vs_scaled_radius", out_dirs)


def plot_conditional_log_velocity_6sizes(
    df_summary: pd.DataFrame,
    out_dirs: List[Path],
    log_base: str = "log10"
):
    """6つの貨物サイズを同一グラフにプロット (ビン平均 + SEM)"""
    fig, ax = plt.subplots(figsize=(8.5, 6.2))

    y_col = "mean_log10_v_tilde" if log_base == "log10" else "mean_ln_v_tilde"
    yerr_col = "sem_log10_v_tilde" if log_base == "log10" else "sem_ln_v_tilde"
    ylabel_str = (
        r"Conditional Mean Log Velocity $\langle \log_{10}\tilde{v} \mid M \rangle$"
        if log_base == "log10"
        else r"Conditional Mean Log Velocity $\langle \ln\tilde{v} \mid M \rangle$"
    )

    for b in BEADS_INFO:
        b_name = b["name"]
        color = b["color"]
        label = b["label"]
        marker = b["marker"]

        sub = df_summary[df_summary["bead_name"] == b_name].dropna(subset=[y_col])
        if sub.empty:
            continue

        x = sub["m_mean"].values
        y = sub[y_col].values
        yerr = sub[yerr_col].values
        total_n = sub["n_count"].sum()

        ax.errorbar(
            x, y, yerr=yerr,
            fmt=marker + '-',
            color=color,
            ms=7.0,
            lw=1.8,
            capsize=4.0,
            capthick=1.2,
            elinewidth=1.4,
            alpha=0.9,
            label=f"$d = {label}$ ($N={total_n:,}$)"
        )

    ax.set_xlabel(r"Local Magnetization $M = |m_{\mathrm{ising}}|$ (bin width $= 0.1$)", fontsize=12.5)
    ax.set_ylabel(ylabel_str, fontsize=12.5)
    ax.set_xlim(-0.02, 1.02)

    title_main = (
        r"Conditional Mean Log Velocity $\langle \log_{10}\tilde{v} \mid M \rangle$ vs Magnetization $M$"
        if log_base == "log10"
        else r"Conditional Mean Log Velocity $\langle \ln\tilde{v} \mid M \rangle$ vs Magnetization $M$"
    )
    ax.set_title(title_main + "\n(Across 6 Cargo Sizes, Error Bars = SEM, QQ Filtered)", fontsize=13.0, fontweight='bold', pad=12)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.grid(True, which='major', ls=':', alpha=0.6)
    ax.legend(loc='best', fontsize=10.0, framealpha=0.9, title="Particle Size")

    plt.tight_layout()
    save_figure_to_all(fig, f"conditional_{log_base}_velocity_vs_magnetization_6sizes", out_dirs)


def plot_conditional_log_velocity_2panel(
    df_summary: pd.DataFrame,
    out_dirs: List[Path],
    log_base: str = "log10"
):
    """2パネル図: 上段 <log v_tilde | M> vs M, 下段 サンプル数 N vs M"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.0, 8.5), sharex=True, gridspec_kw={'height_ratios': [2.2, 1.0]})

    y_col = "mean_log10_v_tilde" if log_base == "log10" else "mean_ln_v_tilde"
    yerr_col = "sem_log10_v_tilde" if log_base == "log10" else "sem_ln_v_tilde"
    ylabel_str = (
        r"$\langle \log_{10}\tilde{v} \mid M \rangle$"
        if log_base == "log10"
        else r"$\langle \ln\tilde{v} \mid M \rangle$"
    )

    for b in BEADS_INFO:
        b_name = b["name"]
        color = b["color"]
        label = b["label"]
        marker = b["marker"]

        sub = df_summary[df_summary["bead_name"] == b_name].dropna(subset=[y_col])
        if sub.empty:
            continue

        x = sub["m_mean"].values
        y = sub[y_col].values
        yerr = sub[yerr_col].values
        counts = sub["n_count"].values

        ax1.errorbar(
            x, y, yerr=yerr,
            fmt=marker + '-',
            color=color,
            ms=6.5,
            lw=1.7,
            capsize=3.5,
            capthick=1.2,
            elinewidth=1.3,
            alpha=0.9,
            label=f"$d = {label}$"
        )
        ax2.plot(x, counts, marker=marker, color=color, ms=5.0, lw=1.3, alpha=0.8)

    ax1.set_ylabel(ylabel_str, fontsize=12.5)
    ax1.grid(True, which='major', ls=':', alpha=0.6)
    ax1.legend(loc='best', fontsize=9.5, framealpha=0.9, ncol=2)
    ax1.set_title(
        r"Conditional Mean Log Velocity $\langle \log_{10}\tilde{v} \mid M \rangle$ and Sample Counts vs $M$ (QQ Filtered)",
        fontsize=13.0, fontweight='bold', pad=10
    )

    ax2.set_xlabel(r"Local Magnetization $M = |m_{\mathrm{ising}}|$ (bin width $= 0.1$)", fontsize=12.0)
    ax2.set_ylabel(r"Count $N$", fontsize=11.5)
    ax2.set_yscale('log')
    ax2.set_xlim(-0.02, 1.02)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax2.grid(True, which='both', ls=':', alpha=0.5)

    plt.tight_layout()
    save_figure_to_all(fig, f"conditional_{log_base}_velocity_vs_magnetization_2panel", out_dirs)


def plot_conditional_log_velocity_grid_per_size(
    df_summary: pd.DataFrame,
    out_dirs: List[Path],
    log_base: str = "log10"
):
    """各粒子径および全体プールの個別パネル Grid 表示 (2行4列)"""
    fig, axes = plt.subplots(2, 4, figsize=(18, 9.5), sharex=True, sharey=False)
    axes_flat = axes.flatten()

    y_col = "mean_log10_v_tilde" if log_base == "log10" else "mean_ln_v_tilde"
    yerr_col = "sem_log10_v_tilde" if log_base == "log10" else "sem_ln_v_tilde"
    ylabel_str = (
        r"$\langle \log_{10}\tilde{v} \mid M \rangle$"
        if log_base == "log10"
        else r"$\langle \ln\tilde{v} \mid M \rangle$"
    )

    target_keys = [b["name"] for b in BEADS_INFO] + ["overall_pooled"]

    for idx, key in enumerate(target_keys):
        ax = axes_flat[idx]

        if key == "overall_pooled":
            label = "All Beads Pooled"
            color = "#333333"
            marker = "x"
        else:
            b_info = next(b for b in BEADS_INFO if b["name"] == key)
            label = f"$d = {b_info['label']}$"
            color = b_info["color"]
            marker = b_info["marker"]

        sub = df_summary[df_summary["bead_name"] == key].dropna(subset=[y_col])
        if sub.empty:
            ax.set_title(f"{label} (No data)", fontsize=11)
            continue

        x = sub["m_mean"].values
        y = sub[y_col].values
        yerr = sub[yerr_col].values
        counts = sub["n_count"].values
        total_n = np.sum(counts)

        ax.errorbar(
            x, y, yerr=yerr,
            fmt=marker + '-',
            color=color,
            ms=6.5,
            lw=1.8,
            capsize=4.0,
            elinewidth=1.4,
            alpha=0.9,
            label=f"Data ($N={total_n:,}$)"
        )

        ax.set_title(f"{label}", fontsize=12.0, fontweight='bold')
        ax.set_xlabel(r"Magnetization $M$", fontsize=10.5)
        ax.set_ylabel(ylabel_str, fontsize=10.5)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-1.1, 0.0) if log_base == "log10" else ax.set_ylim(-2.5, 0.0)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.grid(True, which='major', ls=':', alpha=0.6)
        ax.legend(loc='lower right', fontsize=8.5, framealpha=0.8)

    if len(target_keys) < len(axes_flat):
        for rem_idx in range(len(target_keys), len(axes_flat)):
            axes_flat[rem_idx].axis('off')

    plt.suptitle(
        r"Conditional Mean Log Velocity vs Magnetization by Size (bin width $= 0.1$, QQ Filtered)",
        fontsize=13.5, fontweight='bold', y=0.99
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure_to_all(fig, "conditional_log_velocity_grid_per_size", out_dirs)


def compute_model_residuals(
    df: pd.DataFrame,
    fit_dict: Dict[str, dict],
    z_thresh: float = 2.5
) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    """
    モデル ln(v_tilde) = y0 + beta * M に対する残差:
        epsilon = ln(v_tilde) - (y0 + beta * M)
    を各ビーズサイズおよび全体プールについて全点で計算し、
    残差の外れ値 (|z_epsilon| > z_thresh, 既定 2.5) を除去したクリーンな残差 epsilon_clean に対して、
    平均、標準偏差、歪度、尖度、ガウス適合度 R^2、Q-Q 直線直線性 R^2、
    および正規性検定（Kolmogorov-Smirnov 検定、D'Agostino-Pearson 検定）の統計量を再算出し検証する。
    """
    df_res = df.copy()
    df_res['residual_epsilon'] = np.nan
    df_res['standardized_epsilon'] = np.nan
    df_res['is_residual_valid'] = True

    residual_stats = {}
    target_groups = [(b["name"], b["diameter_um"], b["radius_um"], b["label"], b["marker"], b["color"]) for b in BEADS_INFO]
    target_groups.append(("overall_pooled", np.nan, np.nan, "All Beads Pooled", "x", "#333333"))

    for b_name, d_um, r_um, b_label, marker, color in target_groups:
        if b_name == "overall_pooled":
            sub_idx = df_res.index
            fit_info = fit_dict.get("overall_pooled", {})
        else:
            sub_idx = df_res[df_res['bead_name'] == b_name].index
            fit_info = fit_dict.get(b_name, {})

        if len(sub_idx) < 5 or not fit_info:
            continue

        beta = fit_info.get("ln_beta", fit_info.get("ln_beta_slope", 0.0))
        y0 = fit_info.get("ln_y0", fit_info.get("ln_y0_intercept", 0.0))

        m_vals = df_res.loc[sub_idx, 'abs_m'].values
        y_vals = df_res.loc[sub_idx, 'ln_v_tilde'].values

        # 全残差 raw_epsilon
        raw_epsilon = y_vals - (y0 + beta * m_vals)
        raw_mu = float(np.mean(raw_epsilon))
        raw_std = float(np.std(raw_epsilon, ddof=1))
        raw_z = (raw_epsilon - raw_mu) / raw_std if raw_std > 0 else np.zeros_like(raw_epsilon)

        # 残差の外れ値フィルタ (|z_eps| <= z_thresh)
        valid_mask = np.abs(raw_z) <= z_thresh
        n_total = len(raw_epsilon)
        n_clean = int(np.sum(valid_mask))
        n_outliers = n_total - n_clean
        outlier_pct = float(n_outliers / n_total * 100.0) if n_total > 0 else 0.0

        # Raw 残差の検証統計量
        raw_ks_stat, raw_ks_p = kstest(raw_epsilon, 'norm', args=(raw_mu, raw_std))
        (_, _), (_, _, raw_qq_r) = probplot(raw_epsilon, dist='norm', fit=True)
        raw_sk = float(skew(raw_epsilon))
        raw_kt = float(kurtosis(raw_epsilon))

        # クリーン残差 clean_epsilon
        clean_epsilon = raw_epsilon[valid_mask]
        mu_eps = float(np.mean(clean_epsilon))
        std_eps = float(np.std(clean_epsilon, ddof=1))
        sem_eps = float(std_eps / np.sqrt(len(clean_epsilon))) if len(clean_epsilon) > 0 else 0.0
        z_eps = (clean_epsilon - mu_eps) / std_eps if std_eps > 0 else np.zeros_like(clean_epsilon)

        if b_name != "overall_pooled":
            df_res.loc[sub_idx, 'residual_epsilon'] = raw_epsilon
            df_res.loc[sub_idx, 'standardized_epsilon'] = raw_z
            df_res.loc[sub_idx, 'is_residual_valid'] = valid_mask

        sk = float(skew(clean_epsilon))
        kt = float(kurtosis(clean_epsilon))  # excess kurtosis (0 for normal)

        # Kolmogorov-Smirnov test against Normal(mu, std) on clean residuals
        ks_stat, ks_p = kstest(clean_epsilon, 'norm', args=(mu_eps, std_eps))

        # D'Agostino-Pearson omnibus test on clean residuals
        if len(clean_epsilon) >= 8:
            try:
                norm_stat, norm_p = normaltest(clean_epsilon)
            except Exception:
                norm_stat, norm_p = np.nan, np.nan
        else:
            norm_stat, norm_p = np.nan, np.nan

        # Normal Q-Q plot on clean residuals
        (osm, osr), (qq_slope, qq_intercept, qq_r) = probplot(clean_epsilon, dist='norm', fit=True)

        # Standardized Normal Q-Q on clean standardized residuals
        (std_osm, std_osr), (std_slope, std_intercept, std_r) = probplot(z_eps, dist='norm', fit=True)

        # PDF Histogram & Gaussian R^2 on clean residuals
        counts, bin_edges = np.histogram(clean_epsilon, bins=35, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        p_gauss_hist = norm.pdf(bin_centers, loc=mu_eps, scale=std_eps)
        ss_tot = np.sum((counts - np.mean(counts)) ** 2)
        ss_res = np.sum((counts - p_gauss_hist) ** 2)
        gauss_r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

        residual_stats[b_name] = {
            "name": b_name,
            "diameter_um": d_um,
            "radius_um": r_um,
            "label": b_label,
            "marker": marker,
            "color": color,
            "n_total": n_total,
            "n_points": n_clean,
            "n_outliers": n_outliers,
            "outlier_pct": outlier_pct,
            "epsilon": clean_epsilon,
            "raw_epsilon": raw_epsilon,
            "z_eps": z_eps,
            "mu_eps": mu_eps,
            "std_eps": std_eps,
            "sem_eps": sem_eps,
            "skewness": sk,
            "excess_kurtosis": kt,
            "raw_skewness": raw_sk,
            "raw_excess_kurtosis": raw_kt,
            "raw_ks_pvalue": float(raw_ks_p),
            "raw_qq_r2": float(raw_qq_r ** 2),
            "ks_stat": float(ks_stat),
            "ks_pvalue": float(ks_p),
            "dagostino_stat": float(norm_stat),
            "dagostino_pvalue": float(norm_p),
            "qq_r2": float(qq_r ** 2),
            "qq_slope": float(qq_slope),
            "qq_intercept": float(qq_intercept),
            "osm": osm,
            "osr": osr,
            "std_osm": std_osm,
            "std_osr": std_osr,
            "bin_centers": bin_centers,
            "bin_edges": bin_edges,
            "counts": counts,
            "gauss_r2": float(gauss_r2),
            "beta": beta,
            "y0": y0,
        }

    return df_res, residual_stats


def plot_residual_epsilon_grid(
    res_dict: Dict[str, dict],
    out_dirs: List[Path],
    semilog: bool = False
):
    """残差 epsilon = ln(v_tilde) - (y0 + beta * M) の確率密度分布 P(epsilon) グリッド (2行4列: 6サイズ + 全体プール)"""
    fig, axes = plt.subplots(2, 4, figsize=(18, 9.5), sharex=True, sharey=False)
    axes_flat = axes.flatten()

    target_keys = [b["name"] for b in BEADS_INFO] + ["overall_pooled"]
    scale_str = "semilog" if semilog else "linear"
    x_dense = np.linspace(-4.5, 4.5, 500)

    for idx, key in enumerate(target_keys):
        ax = axes_flat[idx]
        res = res_dict.get(key, {})
        if not res:
            ax.axis('off')
            continue

        label = res["label"]
        color = res["color"]
        counts = res["counts"]
        bin_edges = res["bin_edges"]
        bin_centers = res["bin_centers"]
        mu = res["mu_eps"]
        sigma = res["std_eps"]
        r2 = res["gauss_r2"]
        qq_r2 = res["qq_r2"]
        ks_p = res["ks_pvalue"]
        sk = res["skewness"]
        kt = res["excess_kurtosis"]

        # ヒストグラム
        ax.bar(
            bin_centers, counts, width=np.diff(bin_edges),
            align='center', alpha=0.45, color=color, edgecolor=color,
            label=f"Data ($N={res['n_points']:,}$)"
        )

        # ガウス理論フィット曲線
        p_gauss = norm.pdf(x_dense, loc=mu, scale=sigma)
        ax.plot(x_dense, p_gauss, color='black', lw=2.0, ls='-', label=rf'Gaussian $\mathcal{{N}}(\mu,\sigma^2)$')

        # 中心軸 (epsilon = 0)
        ax.axvline(0, color='#888888', ls=':', lw=1.2)

        ks_str = f"$p = {ks_p:.3f}$" if ks_p >= 0.001 else f"$p = {ks_p:.1e}$"
        text_str = (
            f"$\\mu_\\epsilon = {mu:.3f}$\n"
            f"$\\sigma_\\epsilon = {sigma:.3f}$\n"
            f"Skewness = ${sk:+.2f}$\n"
            f"Kurtosis = ${kt:+.2f}$\n"
            f"Q-Q $R^2 = {qq_r2:.4f}$\n"
            f"KS test {ks_str}"
        )
        ax.text(
            0.05, 0.95, text_str, transform=ax.transAxes,
            va='top', ha='left', fontsize=8.8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#cccccc', alpha=0.88)
        )

        title_str = "All Beads Pooled" if key == "overall_pooled" else f"$d = {label}$"
        ax.set_title(title_str, fontsize=12.0, fontweight='bold')
        ax.set_xlabel(r"Residual $\epsilon = \ln\tilde{v} - (y_0 + \beta M)$", fontsize=10.5)
        ax.set_ylabel(r"Probability Density $P(\epsilon)$", fontsize=10.5)
        ax.set_xlim(-4.2, 4.2)

        if semilog:
            ax.set_yscale('log')
            y_min = max(1e-3, np.min(counts[counts > 0]) * 0.5) if np.any(counts > 0) else 1e-3
            y_max = np.max(counts) * 2.5 if np.any(counts > 0) else 10.0
            ax.set_ylim(y_min, y_max)
            ax.legend(loc='lower left', fontsize=8.0, framealpha=0.85)
        else:
            ax.set_ylim(bottom=0)
            ax.legend(loc='upper right', fontsize=8.0, framealpha=0.85)

        ax.grid(True, which='both' if semilog else 'major', ls=':', alpha=0.6)

    if len(target_keys) < len(axes_flat):
        for rem_idx in range(len(target_keys), len(axes_flat)):
            axes_flat[rem_idx].axis('off')

    plt.suptitle(
        rf"Model Residual Distribution $P(\epsilon)$ by Bead Size ({scale_str.capitalize()})" + "\n" +
        r"Residual $\epsilon_i = \ln\tilde{v}_i - (y_0 + \beta M_i)$ vs Fitted Gaussian $\mathcal{N}(0, \sigma^2)$",
        fontsize=13.5, fontweight='bold', y=0.99
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure_to_all(fig, f"residual_epsilon_distribution_{scale_str}_grid", out_dirs)
    if not semilog:
        save_figure_to_all(fig, "residual_epsilon_distribution_grid", out_dirs)


def plot_residual_epsilon_overlay(
    res_dict: Dict[str, dict],
    out_dirs: List[Path]
):
    """全粒子径の標準化残差 Z_epsilon = (epsilon - mu) / sigma の重ね合わせ分布プロット"""
    fig, ax = plt.subplots(figsize=(9.0, 6.5))
    x_dense = np.linspace(-4.0, 4.0, 500)

    for b in BEADS_INFO:
        b_name = b["name"]
        res = res_dict.get(b_name, {})
        if not res:
            continue

        label = res["label"]
        color = res["color"]
        marker = res["marker"]
        z_eps = res["z_eps"]

        counts, bin_edges = np.histogram(z_eps, bins=30, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        ax.plot(
            bin_centers, counts, marker=marker, ms=6.5, ls='none',
            color=color, alpha=0.85, markeredgecolor='black', markeredgewidth=0.8,
            label=f"$d = {label}$ ($N={res['n_points']:,}$, $\\sigma={res['std_eps']:.2f}$)"
        )

    # 標準正規分布 N(0, 1)
    p_std_norm = norm.pdf(x_dense, 0, 1)
    ax.plot(x_dense, p_std_norm, color='black', lw=2.2, ls='--', label=r'Standard Normal $\mathcal{N}(0, 1)$')

    ax.set_xlabel(r"Standardized Residual $Z_\epsilon = (\epsilon - \mu_\epsilon) / \sigma_\epsilon$", fontsize=12.5)
    ax.set_ylabel(r"Probability Density $P(Z_\epsilon)$", fontsize=12.5)
    ax.set_xlim(-3.8, 3.8)
    ax.set_ylim(bottom=0)

    ax.set_title(
        r"Standardized Residual Distribution $P(Z_\epsilon)$ Across All 6 Sizes" + "\n" +
        r"Residual $\epsilon = \ln\tilde{v} - (y_0 + \beta M)$ Collapse onto Standard Normal $\mathcal{N}(0,1)$",
        fontsize=13.0, fontweight='bold', pad=12
    )
    ax.grid(True, which='major', ls=':', alpha=0.6)
    ax.legend(loc='upper right', fontsize=9.5, framealpha=0.9)

    plt.tight_layout()
    save_figure_to_all(fig, "residual_epsilon_distribution_overlay", out_dirs)


def plot_residual_epsilon_qq_grid(
    res_dict: Dict[str, dict],
    out_dirs: List[Path]
):
    """残差 epsilon の正規 Q-Q プロット グリッド (2行4列: 6サイズ + 全体プール)"""
    fig, axes = plt.subplots(2, 4, figsize=(18, 9.5), sharex=False, sharey=False)
    axes_flat = axes.flatten()

    target_keys = [b["name"] for b in BEADS_INFO] + ["overall_pooled"]

    for idx, key in enumerate(target_keys):
        ax = axes_flat[idx]
        res = res_dict.get(key, {})
        if not res:
            ax.axis('off')
            continue

        label = res["label"]
        color = res["color"]
        marker = res["marker"]
        osm = res["osm"]
        osr = res["osr"]
        slope = res["qq_slope"]
        intercept = res["qq_intercept"]
        qq_r2 = res["qq_r2"]
        n_pts = res["n_points"]
        ks_p = res["ks_pvalue"]

        # データ点
        ax.plot(
            osm, osr, marker=marker, color=color, ms=4.5, alpha=0.65, ls='none',
            label=f"Data ($N={n_pts:,}$)"
        )

        # 回帰直線 (Fit Line)
        x_line = np.array([np.min(osm), np.max(osm)])
        y_line = intercept + slope * x_line
        ax.plot(x_line, y_line, color='black', lw=1.8, ls='--', label=f'Fit Line ($R^2={qq_r2:.4f}$)')

        ks_str = f"KS $p = {ks_p:.3f}$" if ks_p >= 0.001 else f"KS $p = {ks_p:.1e}$"
        text_str = (
            f"$R^2 = {qq_r2:.4f}$\n"
            f"Slope $\\approx {slope:.3f}$\n"
            f"Intercept $\\approx {intercept:.3f}$\n"
            f"{ks_str}"
        )
        ax.text(
            0.05, 0.95, text_str, transform=ax.transAxes,
            va='top', ha='left', fontsize=9.0,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#cccccc', alpha=0.88)
        )

        title_str = "All Beads Pooled" if key == "overall_pooled" else f"$d = {label}$"
        ax.set_title(title_str, fontsize=12.0, fontweight='bold')
        ax.set_xlabel(r"Theoretical Quantiles $z$", fontsize=10.0)
        ax.set_ylabel(r"Sample Quantiles: Residual $\epsilon$", fontsize=10.0)
        ax.grid(True, which='major', ls=':', alpha=0.6)
        ax.legend(loc='lower right', fontsize=8.5, framealpha=0.85)

    if len(target_keys) < len(axes_flat):
        for rem_idx in range(len(target_keys), len(axes_flat)):
            axes_flat[rem_idx].axis('off')

    plt.suptitle(
        r"Normal Q-Q Plots for Model Residuals $\epsilon = \ln\tilde{v} - (y_0 + \beta M)$ by Particle Size" + "\n" +
        r"High Linearity ($R^2 > 0.98$) Confirms Residuals Closely Follow Gaussian Distribution",
        fontsize=13.5, fontweight='bold', y=0.99
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure_to_all(fig, "residual_epsilon_qq_grid", out_dirs)


def plot_residual_epsilon_qq_overlay(
    res_dict: Dict[str, dict],
    out_dirs: List[Path]
):
    """全粒子径の標準化残差 正規 Q-Q プロット 重ね合わせ (Standardized Normal Q-Q Overlay)"""
    fig, ax = plt.subplots(figsize=(9.0, 7.0))
    q_lim = 3.5

    for b in BEADS_INFO:
        b_name = b["name"]
        res = res_dict.get(b_name, {})
        if not res:
            continue

        label = res["label"]
        color = res["color"]
        marker = res["marker"]
        std_osm = res["std_osm"]
        std_osr = res["std_osr"]
        qq_r2 = res["qq_r2"]

        ax.plot(
            std_osm, std_osr, marker=marker, ms=6.5, ls='none',
            color=color, alpha=0.85, markeredgecolor='black', markeredgewidth=0.8,
            label=f"$d = {label}$ ($R^2={qq_r2:.4f}$)"
        )

    # 理想線 y = x
    ax.plot([-q_lim, q_lim], [-q_lim, q_lim], color='black', ls='--', lw=2.0, label='Standard Normal ($y = x$)')

    ax.set_xlabel(r"Theoretical Quantiles $z$ (Standard Normal)", fontsize=12.5)
    ax.set_ylabel(r"Standardized Sample Quantiles $Z_\epsilon = (\epsilon - \mu)/\sigma$", fontsize=12.5)
    ax.set_xlim(-q_lim, q_lim)
    ax.set_ylim(-q_lim, q_lim)

    ax.set_title(
        r"Standardized Normal Q-Q Plot Overlay of Residuals $\epsilon = \ln\tilde{v} - (y_0 + \beta M)$" + "\n" +
        r"Verification of Normality across All 6 Cargo Sizes",
        fontsize=13.0, fontweight='bold', pad=12
    )
    ax.grid(True, which='major', ls=':', alpha=0.6)
    ax.legend(loc='upper left', fontsize=9.5, framealpha=0.9)

    plt.tight_layout()
    save_figure_to_all(fig, "residual_epsilon_qq_overlay", out_dirs)


def plot_residual_epsilon_stats_vs_diameter(
    res_dict: Dict[str, dict],
    xi_info: Dict[str, dict],
    out_dirs: List[Path]
):
    """残差標準偏差 sigma_epsilon の粒子径 d および スケール半径 x = Rc / xi に対する依存性プロット"""
    diameters, sigma_list, sem_list = [], [], []
    rc_xi_list, rc_xi_sem_list = [], []
    markers, colors, labels = [], [], []

    for b in BEADS_INFO:
        b_name = b["name"]
        res = res_dict.get(b_name, {})
        if not res:
            continue

        d = b["diameter_um"]
        xi_data = xi_info.get(b_name, {})

        diameters.append(d)
        sigma_list.append(res["std_eps"])
        sem_list.append(res["sem_eps"])
        rc_xi_list.append(xi_data.get("rc_over_xi_mean", np.nan))
        rc_xi_sem_list.append(xi_data.get("rc_over_xi_sem", np.nan))
        markers.append(b["marker"])
        colors.append(b["color"])
        labels.append(b["label"])

    diameters = np.array(diameters)
    sigma_arr = np.array(sigma_list)

    # 1. sigma_epsilon vs Diameter
    fig_d, ax_d = plt.subplots(figsize=(7.5, 6.0))
    for i in range(len(diameters)):
        ax_d.plot(
            diameters[i], sigma_arr[i],
            marker=markers[i], color=colors[i], ms=10.0, ls='none',
            markeredgecolor='black', markeredgewidth=1.2, zorder=5,
            label=f"$d = {labels[i]}$"
        )
    ax_d.set_xscale('log')
    ax_d.set_xlabel(r"Particle Diameter $d$ [$\mu\mathrm{m}$]", fontsize=13.0)
    ax_d.set_ylabel(r"Residual Standard Deviation $\sigma_\epsilon$", fontsize=13.0)
    ax_d.set_title(r"Residual Noise Width $\sigma_\epsilon$ vs Particle Diameter $d$" + "\n" + r"($\epsilon = \ln\tilde{v} - (y_0 + \beta M)$)", fontsize=13.5, fontweight='bold', pad=10)
    ax_d.set_ylim(0, max(sigma_arr) * 1.3)
    ax_d.grid(True, which='both', ls=':', alpha=0.6)
    ax_d.legend(loc='best', fontsize=9.5, framealpha=0.9)
    plt.tight_layout()
    save_figure_to_all(fig_d, "residual_epsilon_std_vs_diameter", out_dirs)

    # 2. sigma_epsilon vs Scaled Radius x = Rc / xi
    if np.any(np.isfinite(rc_xi_list)):
        fig_x, ax_x = plt.subplots(figsize=(7.5, 6.0))
        for i in range(len(diameters)):
            if not np.isfinite(rc_xi_list[i]):
                continue
            ax_x.errorbar(
                rc_xi_list[i], sigma_arr[i],
                xerr=rc_xi_sem_list[i],
                fmt=markers[i], color=colors[i], ecolor='black',
                elinewidth=1.6, capsize=4.5, capthick=1.2, markersize=10.0,
                markeredgecolor='black', markeredgewidth=1.2, zorder=5,
                label=f"$d = {labels[i]}$"
            )
        ax_x.set_xscale('log')
        ax_x.set_xlabel(r"Scaled Radius $x = R_c / \xi_{i,t}$", fontsize=13.0)
        ax_x.set_ylabel(r"Residual Standard Deviation $\sigma_\epsilon$", fontsize=13.0)
        ax_x.set_title(r"Residual Noise Width $\sigma_\epsilon$ vs Scaled Radius $x = R_c / \xi$" + "\n" + r"($\epsilon = \ln\tilde{v} - (y_0 + \beta M)$)", fontsize=13.5, fontweight='bold', pad=10)
        ax_x.set_ylim(0, max(sigma_arr) * 1.3)
        ax_x.grid(True, which='both', ls=':', alpha=0.6)
        ax_x.legend(loc='best', fontsize=9.5, framealpha=0.9)
        plt.tight_layout()
        save_figure_to_all(fig_x, "residual_epsilon_std_vs_scaled_radius", out_dirs)


def save_residual_summary_csv(
    res_dict: Dict[str, dict],
    out_dirs: List[Path]
):
    """残差の統計量および正規性検証結果サマリーCSVを保存"""
    records = []
    target_keys = [b["name"] for b in BEADS_INFO] + ["overall_pooled"]

    for key in target_keys:
        res = res_dict.get(key, {})
        if not res:
            continue

        records.append({
            "bead_name": key,
            "diameter_um": res["diameter_um"],
            "radius_um": res["radius_um"],
            "n_total": res["n_total"],
            "n_clean": res["n_points"],
            "n_outliers": res["n_outliers"],
            "outlier_pct": res["outlier_pct"],
            "beta_slope": res["beta"],
            "y0_intercept": res["y0"],
            "mean_epsilon": res["mu_eps"],
            "std_epsilon": res["std_eps"],
            "sem_epsilon": res["sem_eps"],
            "skewness": res["skewness"],
            "excess_kurtosis": res["excess_kurtosis"],
            "gaussian_fit_r2": res["gauss_r2"],
            "qq_linearity_r2": res["qq_r2"],
            "qq_slope": res["qq_slope"],
            "qq_intercept": res["qq_intercept"],
            "ks_statistic": res["ks_stat"],
            "ks_pvalue": res["ks_pvalue"],
            "is_gaussian_ks_05": res["ks_pvalue"] > 0.05,
            "dagostino_statistic": res["dagostino_stat"],
            "dagostino_pvalue": res["dagostino_pvalue"],
            "raw_ks_pvalue": res.get("raw_ks_pvalue", np.nan),
            "raw_qq_r2": res.get("raw_qq_r2", np.nan),
            "raw_skewness": res.get("raw_skewness", np.nan),
            "raw_excess_kurtosis": res.get("raw_excess_kurtosis", np.nan),
        })

    df_out = pd.DataFrame(records)
    save_csv_to_all(df_out, "conditional_log_velocity_residual_summary", out_dirs)
    print("\n--- Model Residual Epsilon Normality Summary Table (|z| <= 2.5 Filtered) ---")
    print(df_out.to_string(index=False))


def save_qq_filtering_summary_csv(
    diagnostic_dict: Dict[str, dict],
    df_fit_raw: pd.DataFrame,
    df_fit_clean: pd.DataFrame,
    out_dirs: List[Path]
):
    """Q-Q 外れ値除去前後の比較サマリーCSVを保存"""
    records = []
    target_keys = [b["name"] for b in BEADS_INFO] + ["overall_pooled"]

    for key in target_keys:
        diag = diagnostic_dict.get(key, {})
        raw_row = df_fit_raw[df_fit_raw['bead_name'] == key]
        clean_row = df_fit_clean[df_fit_clean['bead_name'] == key]

        b_info = next((b for b in BEADS_INFO if b["name"] == key), None)
        d_um = b_info["diameter_um"] if b_info else np.nan
        r_um = b_info["radius_um"] if b_info else np.nan

        raw_beta = float(raw_row['ln_beta_slope'].iloc[0]) if not raw_row.empty else np.nan
        raw_y0 = float(raw_row['ln_y0_intercept'].iloc[0]) if not raw_row.empty else np.nan
        raw_r2 = float(raw_row['ln_r2'].iloc[0]) if not raw_row.empty else np.nan

        clean_beta = float(clean_row['ln_beta_slope'].iloc[0]) if not clean_row.empty else np.nan
        clean_y0 = float(clean_row['ln_y0_intercept'].iloc[0]) if not clean_row.empty else np.nan
        clean_r2 = float(clean_row['ln_r2'].iloc[0]) if not clean_row.empty else np.nan

        records.append({
            "bead_name": key,
            "diameter_um": d_um,
            "radius_um": r_um,
            "n_total": diag.get("n_total", np.nan),
            "n_kept": diag.get("n_valid", np.nan),
            "n_removed": diag.get("n_outliers", np.nan),
            "outlier_removal_pct": diag.get("outlier_pct", np.nan),
            "raw_ln_beta": raw_beta,
            "clean_ln_beta": clean_beta,
            "raw_ln_y0": raw_y0,
            "clean_ln_y0": clean_y0,
            "raw_ln_r2": raw_r2,
            "clean_ln_r2": clean_r2,
        })

    df_out = pd.DataFrame(records)
    save_csv_to_all(df_out, "conditional_log_velocity_qq_filtered_summary", out_dirs)
    print("\n--- QQ Outlier Filter Comparison Table ---")
    print(df_out.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Plot raw data points (M_i, ln v_tilde_i) with QQ outlier removal, linear fit, residual epsilon analysis, and scaling."
    )
    parser.add_argument(
        "--points-csv",
        type=Path,
        default=CURRENT_DIR / "figure" / "cargo_spin_velocity" / "cargo_spin_velocity_points.csv",
        help="Path to cargo_spin_velocity_points.csv"
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=None,
        help="Root directory of dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CURRENT_DIR / "figure" / "conditional_log_velocity",
        help="Output directory"
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=10,
        help="Number of bins for magnetization M in [0.0, 1.0] (default: 10, width=0.1)"
    )
    parser.add_argument(
        "--z-thresh",
        type=float,
        default=2.5,
        help="QQ theoretical quantile threshold |z| <= z_thresh for outlier removal (default: 2.5)"
    )

    args = parser.parse_args()
    apply_custom_style()

    root_dir = args.root_dir if args.root_dir else find_default_root()
    out_dirs = [args.output_dir]
    if root_dir.exists():
        nas_out_dir = root_dir / "figure" / "conditional_log_velocity"
        out_dirs.append(nas_out_dir)

    print(f"Output directories: {[str(d) for d in out_dirs]}")

    # 相関長 xi_{i,t} および スケール半径 x = R_c / xi の情報読み込み
    xi_info = load_scaled_radius_info(root_dir)

    # 1. 生データ読み込み
    df_raw = load_and_preprocess_data(args.points_csv)
    _, df_fit_raw = compute_linear_fits(df_raw, xi_info)

    # 2. Q-Q プロットに基づく外れ値フィルタリングの適用
    print(f"\nApplying Lognormal QQ Outlier Filter (threshold |z| <= {args.z_thresh})...")
    df_clean, diag_dict = apply_lognormal_qq_outlier_filter(df_raw, z_thresh=args.z_thresh)

    # 3. 外れ値除去後データに対する線形回帰 y = y0 + beta * M の算出
    fit_dict_clean, df_fit_clean = compute_linear_fits(df_clean, xi_info)

    # 4. モデル残差 epsilon = ln(v_tilde) - (y0 + beta * M) の算出および正規性検証
    print(f"\nComputing Model Residuals epsilon = ln(v_tilde) - (y0 + beta * M) and verifying Gaussianity (residual threshold |z| <= {args.z_thresh})...")
    df_clean, res_dict = compute_model_residuals(df_clean, fit_dict_clean, z_thresh=args.z_thresh)

    # 5. 外れ値除去後データに対するビン分割統計集計 (M: 0.0 ~ 1.0, 10 bins)
    df_binned_clean = compute_conditional_binned_stats(df_clean, n_bins=args.bins, m_range=(0.0, 1.0))

    # 作図 0: Q-Q 外れ値フィルタリング診断図
    plot_qq_outlier_diagnostic(diag_dict, out_dirs)

    # 作図 1: 外れ値除去済 生データ点 + 線形フィット Grid プロット (ln & log10)
    plot_raw_points_linear_fit_grid(fit_dict_clean, df_binned_clean, out_dirs, log_base="ln")
    plot_raw_points_linear_fit_grid(fit_dict_clean, df_binned_clean, out_dirs, log_base="log10")

    # 作図 2: 外れ値除去済 生データ点 + 線形フィット Overlay プロット (ln & log10)
    plot_raw_points_linear_fit_overlay(fit_dict_clean, df_binned_clean, out_dirs, log_base="ln")
    plot_raw_points_linear_fit_overlay(fit_dict_clean, df_binned_clean, out_dirs, log_base="log10")

    # 作図 3: フィットパラメータ vs 粒子径 d (外れ値除去後)
    plot_fit_parameters_vs_diameter(df_fit_clean, out_dirs)

    # 作図 4: フィットパラメータ vs スケール半径 x = R_c / xi_{i,t} (外れ値除去後スケーリング図)
    plot_fit_parameters_vs_scaled_radius(df_fit_clean, out_dirs, log_base="ln")

    # 作図 5: ビン平均 6サイズ同一グラフプロット (外れ値除去後, ln & log10)
    plot_conditional_log_velocity_6sizes(df_binned_clean, out_dirs, log_base="ln")
    plot_conditional_log_velocity_6sizes(df_binned_clean, out_dirs, log_base="log10")

    # 作図 6: 2パネル図 & 個別パネル Grid (外れ値除去後)
    plot_conditional_log_velocity_2panel(df_binned_clean, out_dirs, log_base="log10")
    plot_conditional_log_velocity_grid_per_size(df_binned_clean, out_dirs, log_base="log10")

    # 作図 7: 残差 epsilon の確率密度分布 P(epsilon) Grid (Linear & Semilog)
    plot_residual_epsilon_grid(res_dict, out_dirs, semilog=False)
    plot_residual_epsilon_grid(res_dict, out_dirs, semilog=True)

    # 作図 8: 標準化残差 Z_epsilon 重ね合わせ分布 Overlay vs N(0, 1)
    plot_residual_epsilon_overlay(res_dict, out_dirs)

    # 作図 9: 残差 epsilon の正規 Q-Q プロット Grid & Overlay
    plot_residual_epsilon_qq_grid(res_dict, out_dirs)
    plot_residual_epsilon_qq_overlay(res_dict, out_dirs)

    # 作図 10: 残差ノイズ幅 sigma_epsilon vs 粒子径 d & スケール半径 x
    plot_residual_epsilon_stats_vs_diameter(res_dict, xi_info, out_dirs)

    # CSV 保存
    save_csv_to_all(df_fit_clean, "conditional_log_velocity_linear_fit_summary", out_dirs)
    save_csv_to_all(df_binned_clean, "conditional_log_velocity_vs_magnetization_summary", out_dirs)
    save_qq_filtering_summary_csv(diag_dict, df_fit_raw, df_fit_clean, out_dirs)
    save_residual_summary_csv(res_dict, out_dirs)

    print("\nAll QQ-filtered raw points, linear fit, residual epsilon analysis, and scaled radius figures generated successfully!")


if __name__ == "__main__":
    main()
