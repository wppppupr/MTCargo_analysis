#!/usr/bin/env python3
"""
plot_dimensionless_local_polar.py

notebooks/local_polar.ipynb で行われている局所ポーラーオーダー (Local Polar Order) の解析・プロットを
ベースとし、Window Radius R [μm] を空間配向相関から実験ごとに求めた相関長 xi [μm] で割って
無次元化 (R / xi) してプロット・フィッティングを行うスクリプト。

【仕様と設計方針】
1. 各実験（フォルダー）ごとに空間配向相関から求めた相関長 xi_um を使用
   (bg_angular_correlation_length_per_experiment.csv)
2. 無次元化半径 R_norm = (window_size_px * 0.11 μm/px) / xi_exp
3. 貨物中心 (w/ Cargo):
   - ビーズ半径内部 (R <= R_bead) は微小管が存在しないマスク領域のため、
     物理的に妥当な外側領域 (R > R_bead かつ 非 NaN/inf かつ P > 0.01) のみを有効値としてプロット・フィッティング
4. 3種類のデータ系列に対応:
   - w/ Cargo (local_polar_w.zarr): 粒子中心の局所ポーラーオーダー
   - w/o Cargo (local_polar_bg.zarr): 貨物なし背景領域の局所ポーラーオーダー
   - Control MTs (local_polar_noCargo.zarr): ビーズ非添加コントロール微小管
5. ビーズサイズごとの個別ズームプロット (個別実験曲線 + 条件アンサンブル平均 ± SD)
6. 束ねた全体スケーリングプロット (w/ Cargo, w/o Cargo, Control)
7. 指数減衰モデルによるフィッティング:
   P(R/xi) = (1 - C) * exp(-(R/xi) / xi_norm) + C
"""

import sys
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# ==============================================================================
# 設定・定数
# ==============================================================================
SCALE = 0.11  # μm / pixel

CONDITIONS_INFO = [
    {
        'key': 'beads06um',
        'label': '0.63 μm',
        'diameter_um': 0.63,
        'marker': '^',
        'color_idx': 0,
        'rel_dir': 'beads06um',
    },
    {
        'key': 'beads1um',
        'label': '1.18 μm',
        'diameter_um': 1.18,
        'marker': 'o',
        'color_idx': 1,
        'rel_dir': 'beads1um',
    },
    {
        'key': 'beads3um',
        'label': '3.37 μm',
        'diameter_um': 3.37,
        'marker': 'd',
        'color_idx': 2,
        'rel_dir': 'beads3um',
    },
    {
        'key': 'beads5um',
        'label': '5.00 μm',
        'diameter_um': 5.00,
        'marker': 'p',
        'color_idx': 3,
        'rel_dir': 'beads5um',
    },
    {
        'key': 'beads7um',
        'label': '7.24 μm',
        'diameter_um': 7.24,
        'marker': 'h',
        'color_idx': 4,
        'rel_dir': 'beads7um',
    },
    {
        'key': 'beads20um',
        'label': '20 μm',
        'diameter_um': 20.0,
        'marker': 's',
        'color_idx': 5,
        'rel_dir': 'beads20um',
    },
]

CONTROL_INFO = {
    'key': 'control',
    'label': 'MTs (Control)',
    'diameter_um': np.nan,
    'marker': 'v',
    'color': '#333333',
    'rel_dir': 'control/MTs8uM',
}


def find_data_root() -> Path:
    """データルートディレクトリの探索"""
    candidates = [
        Path("/Volumes/data/Sasaki/MTsingleBeads"),
        Path("/Volumes/data-1/Sasaki/MTsingleBeads"),
        Path("/mnt/NAS-Ebanaru/sasaki/MTsingleBeads"),
        Path("/mnt/NAS-Ebanaru/Sasaki/MTsingleBeads"),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("Data root directory not found in candidates.")


def load_experiment_correlation_lengths(root: Path, workspace: Path) -> Dict[str, float]:
    """
    各実験フォルダーごとの相関長 xi [μm] を CSV から読み込む
    戻り値: dict mapping exp_dir_name -> xi_um
    """
    candidates = [
        root / "figure" / "bg_angular_correlation" / "bg_angular_correlation_length_per_experiment.csv",
        workspace / "figure" / "bg_angular_correlation" / "bg_angular_correlation_length_per_experiment.csv",
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            exp_dict = {}
            for _, row in df.iterrows():
                exp_name = str(row['exp_dir']).strip()
                xi = float(row['xi_um'])
                if pd.notna(xi) and xi > 0:
                    exp_dict[exp_name] = xi
            print(f"[INFO] Loaded {len(exp_dict)} experiment xi values from {p}")
            return exp_dict
    raise FileNotFoundError("bg_angular_correlation_length_per_experiment.csv not found.")


def exp_fit_constrained(x: np.ndarray, xi_star: float, c: float) -> np.ndarray:
    """x = 0 で P(0) = 1 となる指数減衰関数"""
    return (1.0 - c) * np.exp(-x / xi_star) + c


def load_local_polar_for_experiment(
    exp_dir: Path,
    xi_exp: float,
    condition_key: str,
    diameter_um: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    1つの実験ディレクトリから local_polar_w.zarr と local_polar_bg.zarr を読み込み、
    相関長 xi_exp で無次元化した DataFrame を作成する。
    ビーズ内部（マスク領域）や無効な値 (NaN, Inf, 極小値) は除外する。
    """
    p_df = pd.DataFrame()
    bg_df = pd.DataFrame()
    bead_radius_um = diameter_um / 2.0

    # 1. w/ Cargo (local_polar_w.zarr)
    w_path = exp_dir / "local_polar_w.zarr"
    if w_path.exists():
        try:
            ds_w = xr.open_zarr(w_path, consolidated=False)
            p_mean = ds_w.polar_order.mean(dim=['frame', 'particle']).compute().values
            win_size_px = ds_w.coords['window size'].values
            r_um = win_size_px * SCALE
            r_norm = r_um / xi_exp

            # ビーズマスク内部および NaN / Inf / <=0.01 の無効値を除外
            valid_mask = (
                np.isfinite(p_mean) &
                (p_mean > 0.01) &
                (p_mean <= 1.05) &
                (r_um >= bead_radius_um * 0.95)
            )

            if valid_mask.any():
                p_df = pd.DataFrame({
                    'condition': condition_key,
                    'diameter_um': diameter_um,
                    'exp_name': exp_dir.name,
                    'exp_path': str(exp_dir),
                    'xi_um': xi_exp,
                    'window_size_px': win_size_px[valid_mask],
                    'window_radius_um': r_um[valid_mask],
                    'window_radius_norm': r_norm[valid_mask],
                    'polar_order': p_mean[valid_mask],
                    'type': 'w_cargo',
                })
        except Exception as e:
            print(f"[WARN] Failed to read {w_path}: {e}")

    # 2. w/o Cargo (local_polar_bg.zarr)
    bg_path = exp_dir / "local_polar_bg.zarr"
    if bg_path.exists():
        try:
            ds_bg = xr.open_zarr(bg_path, consolidated=False)
            bg_mean = ds_bg.polar_order.mean(dim=['frame']).compute().values
            win_size_px = ds_bg.coords['window size'].values
            r_um = win_size_px * SCALE
            r_norm = r_um / xi_exp

            valid_mask = np.isfinite(bg_mean) & (bg_mean > 0.0) & (bg_mean <= 1.05)
            if valid_mask.any():
                bg_df = pd.DataFrame({
                    'condition': condition_key,
                    'diameter_um': diameter_um,
                    'exp_name': exp_dir.name,
                    'exp_path': str(exp_dir),
                    'xi_um': xi_exp,
                    'window_size_px': win_size_px[valid_mask],
                    'window_radius_um': r_um[valid_mask],
                    'window_radius_norm': r_norm[valid_mask],
                    'polar_order': bg_mean[valid_mask],
                    'type': 'wo_cargo',
                })
        except Exception as e:
            print(f"[WARN] Failed to read {bg_path}: {e}")

    return p_df, bg_df


def load_control_for_experiment(
    exp_dir: Path,
    xi_exp: float,
) -> pd.DataFrame:
    """
    コントロール微小管ディレクトリから local_polar_noCargo.zarr を読み込み、
    xi_exp で無次元化した DataFrame を作成する。
    """
    no_path = exp_dir / "local_polar_noCargo.zarr"
    if not no_path.exists():
        return pd.DataFrame()

    try:
        ds_no = xr.open_zarr(no_path, consolidated=False)
        no_mean = ds_no.polar_order.mean(dim=['frame']).compute().values
        win_size_px = ds_no.coords['window size'].values
        r_um = win_size_px * SCALE
        r_norm = r_um / xi_exp

        valid_mask = np.isfinite(no_mean) & (no_mean > 0.0) & (no_mean <= 1.05)
        if valid_mask.any():
            return pd.DataFrame({
                'condition': 'control',
                'diameter_um': np.nan,
                'exp_name': exp_dir.name,
                'exp_path': str(exp_dir),
                'xi_um': xi_exp,
                'window_size_px': win_size_px[valid_mask],
                'window_radius_um': r_um[valid_mask],
                'window_radius_norm': r_norm[valid_mask],
                'polar_order': no_mean[valid_mask],
                'type': 'control',
            })
        return pd.DataFrame()
    except Exception as e:
        print(f"[WARN] Failed to read {no_path}: {e}")
        return pd.DataFrame()


def load_all_dimensionless_data(
    root: Path,
    workspace: Path,
    exp_xi_dict: Dict[str, float]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """全実験の無次元化局所ポーラーオーダーデータを一括読み込み"""
    p_dfs = []
    bg_dfs = []
    ctrl_dfs = []

    # 1. 各ビーズ条件
    for cond in CONDITIONS_INFO:
        pattern = str(root / cond['rel_dir'] / "*" / "*")
        exp_dirs = [Path(d) for d in glob.glob(pattern) if Path(d).is_dir()]
        for exp_d in exp_dirs:
            exp_name = exp_d.name
            if exp_name not in exp_xi_dict:
                print(f"[WARN] No xi found for experiment '{exp_name}' in {cond['key']}. Skipping.")
                continue
            xi = exp_xi_dict[exp_name]
            p_df, bg_df = load_local_polar_for_experiment(
                exp_d, xi, cond['key'], cond['diameter_um']
            )
            if not p_df.empty:
                p_dfs.append(p_df)
            if not bg_df.empty:
                bg_dfs.append(bg_df)

    # 2. コントロール微小管 (control/MTs8uM)
    ctrl_pattern = str(root / CONTROL_INFO['rel_dir'] / "*" / "*")
    ctrl_dirs = [Path(d) for d in glob.glob(ctrl_pattern) if Path(d).is_dir()]
    for exp_d in ctrl_dirs:
        exp_name = exp_d.name
        if exp_name not in exp_xi_dict:
            print(f"[WARN] No xi found for control experiment '{exp_name}'. Skipping.")
            continue
        xi = exp_xi_dict[exp_name]
        ctrl_df = load_control_for_experiment(exp_d, xi)
        if not ctrl_df.empty:
            ctrl_dfs.append(ctrl_df)

    df_p = pd.concat(p_dfs, ignore_index=True) if p_dfs else pd.DataFrame()
    df_bg = pd.concat(bg_dfs, ignore_index=True) if bg_dfs else pd.DataFrame()
    df_ctrl = pd.concat(ctrl_dfs, ignore_index=True) if ctrl_dfs else pd.DataFrame()

    return df_p, df_bg, df_ctrl


def plot_condition_dimensionless_zoom(
    cond_info: dict,
    df_p_cond: pd.DataFrame,
    df_bg_cond: pd.DataFrame,
    df_ctrl: pd.DataFrame,
    style_colors: list,
    out_dir: Path,
):
    """
    各ビーズ条件ごとの個別ズームプロット (R / xi で無次元化)
    notebooks/local_polar.ipynb の Cell 16-21 に対応する図。
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    c_idx = cond_info['color_idx']
    color = style_colors[c_idx % len(style_colors)]

    # 1. コントロール (Control MTs) の実験別曲線 + 平均
    if not df_ctrl.empty:
        for exp_name, group in df_ctrl.groupby('exp_name'):
            group_sorted = group.sort_values('window_radius_norm')
            ax.plot(
                group_sorted['window_radius_norm'],
                group_sorted['polar_order'],
                color='#888888',
                linestyle=':',
                lw=1.5,
                alpha=0.6,
                zorder=1
            )
        g_ctrl = df_ctrl.groupby('window_size_px')
        ctrl_r_norm_mean = g_ctrl['window_radius_norm'].mean()
        ctrl_p_mean = g_ctrl['polar_order'].mean()
        ctrl_p_std = g_ctrl['polar_order'].std()
        ax.plot(
            ctrl_r_norm_mean,
            ctrl_p_mean,
            label='Control (no bead)',
            color='#333333',
            linestyle='-',
            lw=3.5,
            zorder=2
        )
        ax.fill_between(
            ctrl_r_norm_mean,
            ctrl_p_mean - ctrl_p_std,
            ctrl_p_mean + ctrl_p_std,
            color='#333333',
            alpha=0.15,
            zorder=1
        )

    # 2. w/o Cargo (local_polar_bg)
    if not df_bg_cond.empty:
        for exp_name, group in df_bg_cond.groupby('exp_name'):
            g_s = group.sort_values('window_radius_norm')
            ax.plot(
                g_s['window_radius_norm'],
                g_s['polar_order'],
                color=color,
                linestyle='--',
                lw=1.2,
                alpha=0.4,
                zorder=3
            )
        g_bg = df_bg_cond.groupby('window_size_px')
        bg_r_norm = g_bg['window_radius_norm'].mean()
        bg_p_mean = g_bg['polar_order'].mean()
        bg_p_std = g_bg['polar_order'].std()
        ax.errorbar(
            bg_r_norm,
            bg_p_mean,
            yerr=bg_p_std,
            label=f"{cond_info['label']} w/o Cargo",
            marker='x',
            linestyle='--',
            color=color,
            lw=2.5,
            markersize=8,
            capsize=3,
            zorder=4
        )

    # 3. w/ Cargo (local_polar_w)
    if not df_p_cond.empty:
        for exp_name, group in df_p_cond.groupby('exp_name'):
            g_s = group.sort_values('window_radius_norm')
            ax.plot(
                g_s['window_radius_norm'],
                g_s['polar_order'],
                color=color,
                linestyle='-.',
                lw=1.2,
                alpha=0.4,
                zorder=4
            )
        g_p = df_p_cond.groupby('window_size_px')
        p_r_norm = g_p['window_radius_norm'].mean()
        p_p_mean = g_p['polar_order'].mean()
        p_p_std = g_p['polar_order'].std()
        ax.errorbar(
            p_r_norm,
            p_p_mean,
            yerr=p_p_std,
            label=f"{cond_info['label']} w/ Cargo",
            marker=cond_info['marker'],
            linestyle='-',
            color=color,
            lw=3.0,
            markersize=8,
            capsize=3,
            zorder=5
        )

    ax.vlines(1.0, ymin=0, ymax=1.0, color='#333333', linestyle='--', lw=1.5, zorder=100)
    ax.set_xlabel(r'Dimensionless Window Radius $R/\xi$')
    ax.set_ylabel(r'Polar Order $\Phi(R/\xi)$')
    ax.set_xlim(0, 2.5)
    ax.set_ylim(0, 1.0)
    ax.set_title(f"Local Polar Order: {cond_info['label']} (Scaled by $\\xi$)")
    ax.legend(frameon=True, loc='upper right', fontsize=12)

    fname_base = f"local_polar_dimensionless_{cond_info['key']}"
    for ext in ['png', 'svg', 'pdf']:
        fig.savefig(out_dir / f"{fname_base}.{ext}")
    plt.close(fig)
    print(f"[INFO] Saved {fname_base} plots to {out_dir}")


def plot_dimensionless_master_curves(
    df_p: pd.DataFrame,
    df_bg: pd.DataFrame,
    df_ctrl: pd.DataFrame,
    style_colors: list,
    out_dir: Path,
):
    """
    全ビーズサイズを重ねた無次元化プロット (w/ Cargo および w/o Cargo)
    R / xi を横軸とし、線形軸・対数軸の両方でプロット・フィッティング
    """
    fit_summary = []

    for kind, df_target, kind_title, fname_prefix in [
        ('w_cargo', df_p, 'w/ Cargo', 'local_polar_dimensionless_w_cargo'),
        ('wo_cargo', df_bg, 'w/o Cargo', 'local_polar_dimensionless_wo_cargo'),
    ]:
        if df_target.empty:
            continue

        # -------------------------------------------------------------
        # プロット 1: 線形軸 (0 <= R / xi <= 3.0)
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(9, 6.5))

        # コントロール
        if not df_ctrl.empty:
            g_ctrl = df_ctrl.groupby('window_size_px')
            ctrl_r_norm = g_ctrl['window_radius_norm'].mean()
            ctrl_p_mean = g_ctrl['polar_order'].mean()
            ax.plot(
                ctrl_r_norm,
                ctrl_p_mean,
                label='Control (no bead)',
                color='#333333',
                linestyle='-',
                lw=3.0,
                zorder=2
            )

        for cond in CONDITIONS_INFO:
            df_c = df_target[df_target['condition'] == cond['key']]
            if df_c.empty:
                continue

            g = df_c.groupby('window_size_px')
            r_norm = g['window_radius_norm'].mean()
            p_mean = g['polar_order'].mean()
            p_std = g['polar_order'].std()
            c_idx = cond['color_idx']
            color = style_colors[c_idx % len(style_colors)]

            mean_xi = df_c['xi_um'].unique().mean()
            label_text = f"{cond['label']} ($\\langle\\xi\\rangle$ = {mean_xi:.1f} μm)"

            ax.errorbar(
                r_norm,
                p_mean,
                yerr=p_std,
                label=label_text,
                marker=cond['marker'],
                linestyle='-',
                color=color,
                lw=2.0,
                markersize=7,
                capsize=3,
                zorder=3
            )

            # フィッティング (有効範囲: 0.1 <= R_norm <= 3.0)
            fit_mask = (r_norm >= 0.1) & (r_norm <= 3.0) & np.isfinite(p_mean)
            if fit_mask.sum() >= 3:
                try:
                    popt, pcov = curve_fit(
                        exp_fit_constrained,
                        r_norm[fit_mask].values,
                        p_mean[fit_mask].values,
                        p0=[1.0, 0.4],
                        bounds=([0.05, 0.0], [10.0, 0.85])
                    )
                    xi_norm_fit, c_fit = popt
                    xi_norm_err = np.sqrt(pcov[0, 0]) if pcov is not None else np.nan
                    fit_summary.append({
                        'type': kind,
                        'condition': cond['key'],
                        'label': cond['label'],
                        'diameter_um': cond['diameter_um'],
                        'mean_xi_um': mean_xi,
                        'xi_norm_fit': xi_norm_fit,
                        'xi_norm_err': xi_norm_err,
                        'baseline_C': c_fit,
                    })
                except Exception as e:
                    print(f"[WARN] Fit failed for {kind} {cond['key']}: {e}")

        ax.vlines(1.0, ymin=0, ymax=1.0, color='#333333', linestyle='--', lw=1.5, zorder=100)
        ax.set_xlabel(r'Dimensionless Window Radius $R/\xi$')
        ax.set_ylabel(r'Polar Order $\Phi(R/\xi)$')
        ax.set_xlim(0, 10.0)
        ax.set_ylim(0, 1.0)
        ax.set_title(f"Local Polar Order vs Dimensionless Radius ({kind_title})")
        ax.legend(frameon=True, loc='upper right', fontsize=11)

        for ext in ['png', 'svg', 'pdf']:
            fig.savefig(out_dir / f"{fname_prefix}_linear.{ext}")
        plt.close(fig)

        # -------------------------------------------------------------
        # プロット 2: 対数軸 (Log-x scale: 0.05 <= R / xi <= 50)
        # -------------------------------------------------------------
        fig_log, ax_log = plt.subplots(figsize=(9, 6.5))

        # コントロール
        if not df_ctrl.empty:
            g_ctrl = df_ctrl.groupby('window_size_px')
            ctrl_r_norm = g_ctrl['window_radius_norm'].mean()
            ctrl_p_mean = g_ctrl['polar_order'].mean()
            ax_log.plot(
                ctrl_r_norm,
                ctrl_p_mean,
                label='Control (no bead)',
                color='#333333',
                linestyle='-',
                lw=3.0,
                zorder=2
            )

        for cond in CONDITIONS_INFO:
            df_c = df_target[df_target['condition'] == cond['key']]
            if df_c.empty:
                continue

            g = df_c.groupby('window_size_px')
            r_norm = g['window_radius_norm'].mean()
            p_mean = g['polar_order'].mean()
            p_std = g['polar_order'].std()
            c_idx = cond['color_idx']
            color = style_colors[c_idx % len(style_colors)]

            mean_xi = df_c['xi_um'].unique().mean()
            label_text = f"{cond['label']} ($\\langle\\xi\\rangle$ = {mean_xi:.1f} μm)"

            ax_log.errorbar(
                r_norm,
                p_mean,
                yerr=p_std,
                label=label_text,
                marker=cond['marker'],
                linestyle='-',
                color=color,
                lw=2.0,
                markersize=7,
                capsize=3,
                zorder=3
            )

        ax_log.vlines(1.0, ymin=0, ymax=1.0, color='#333333', linestyle='--', lw=1.5)
        ax_log.set_xlabel(r'Dimensionless Window Radius $R/\xi$')
        ax_log.set_ylabel(r'Polar Order $P$')
        ax_log.set_xscale('log')
        ax_log.set_xlim(0.05, 50.0)
        ax_log.set_ylim(0, 1.0)
        ax_log.set_title(f"Local Polar Order vs Dimensionless Radius ({kind_title})")
        ax_log.legend(frameon=True, loc='upper right', fontsize=11)

        for ext in ['png', 'svg', 'pdf']:
            fig_log.savefig(out_dir / f"{fname_prefix}_log.{ext}")
        plt.close(fig_log)

    # フィッティング結果サマリーの保存
    if fit_summary:
        df_fit = pd.DataFrame(fit_summary)
        csv_path = out_dir / "dimensionless_local_polar_fit_summary.csv"
        df_fit.to_csv(csv_path, index=False)
        print(f"[INFO] Saved fit summary to {csv_path}")
        print("\n=== Exponential Fit Summary P(R/xi) = (1 - C)*exp(-(R/xi)/xi*) + C ===")
        print(df_fit.to_string(index=False))


def plot_side_by_side_comparison(
    df_p: pd.DataFrame,
    df_bg: pd.DataFrame,
    df_ctrl: pd.DataFrame,
    style_colors: list,
    out_dir: Path,
):
    """
    w/ Cargo と w/o Cargo を左右に並べた比較サマリープロット
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), sharey=True)

    configs = [
        (axes[0], df_p, 'w/ Cargo (Particle Center)'),
        (axes[1], df_bg, 'w/o Cargo (Background ROI)'),
    ]

    for ax, df_target, title in configs:
        if df_target.empty:
            continue

        # コントロール
        if not df_ctrl.empty:
            g_ctrl = df_ctrl.groupby('window_size_px')
            ctrl_r_norm = g_ctrl['window_radius_norm'].mean()
            ctrl_p_mean = g_ctrl['polar_order'].mean()
            ax.plot(
                ctrl_r_norm,
                ctrl_p_mean,
                label='Control',
                color='#333333',
                linestyle='-',
                lw=3.0,
                zorder=2
            )

        for cond in CONDITIONS_INFO:
            df_c = df_target[df_target['condition'] == cond['key']]
            if df_c.empty:
                continue

            g = df_c.groupby('window_size_px')
            r_norm = g['window_radius_norm'].mean()
            p_mean = g['polar_order'].mean()
            p_std = g['polar_order'].std()
            c_idx = cond['color_idx']
            color = style_colors[c_idx % len(style_colors)]

            ax.errorbar(
                r_norm,
                p_mean,
                yerr=p_std,
                label=cond['label'],
                marker=cond['marker'],
                linestyle='-',
                color=color,
                lw=2.0,
                markersize=6,
                capsize=2,
                zorder=3
            )

        ax.vlines(1.0, ymin=0, ymax=1.0, color='#333333', linestyle='--', lw=1.5, zorder=100)
        ax.set_xlabel(r'Dimensionless Window Radius $R/\xi$')
        ax.set_xlim(0, 10.0)
        ax.set_ylim(0, 1.0)
        ax.set_title(title)
        ax.legend(frameon=True, loc='upper right', fontsize=11)

    axes[0].set_ylabel(r'Polar Order $P$')

    for ext in ['png', 'svg', 'pdf']:
        fig.savefig(out_dir / f"local_polar_dimensionless_side_by_side.{ext}")
    plt.close(fig)
    print(f"[INFO] Saved side-by-side comparison to {out_dir}")


def main():
    workspace = Path(__file__).resolve().parent
    mplstyle_path = workspace / "libs" / "my_style.mplstyle"
    if mplstyle_path.exists():
        plt.style.use(str(mplstyle_path))
        print(f"[INFO] Loaded mplstyle from {mplstyle_path}")

    style_colors = plt.rcParams.get('axes.prop_cycle', None)
    if style_colors is not None:
        style_colors = style_colors.by_key()['color']
        style_colors = [c if c.startswith('#') else f'#{c}' for c in style_colors]
    else:
        style_colors = ['#882255', '#CC6677', '#DDCC77', '#999933', '#117733', '#44AA99', '#88CCEE', '#332288']

    # 1. データパスと相関長の探索
    root = find_data_root()
    print(f"[INFO] Data root: {root}")

    exp_xi_dict = load_experiment_correlation_lengths(root, workspace)

    # 2. 出力ディレクトリの作成
    out_dirs = [
        workspace / "figure" / "dimensionless_local_polar",
        root / "figure" / "dimensionless_local_polar",
    ]
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)

    # 3. データの読み込み & 実験ごとの無次元化
    print("[INFO] Loading datasets and scaling window sizes by experiment-specific xi...")
    df_p, df_bg, df_ctrl = load_all_dimensionless_data(root, workspace, exp_xi_dict)

    print(f"[INFO] w/ Cargo records: {len(df_p)}, w/o Cargo records: {len(df_bg)}, Control records: {len(df_ctrl)}")

    # 4. 個別ビーズ条件プロット
    for out_d in out_dirs:
        for cond in CONDITIONS_INFO:
            df_p_c = df_p[df_p['condition'] == cond['key']]
            df_bg_c = df_bg[df_bg['condition'] == cond['key']]
            if df_p_c.empty and df_bg_c.empty:
                continue
            plot_condition_dimensionless_zoom(
                cond, df_p_c, df_bg_c, df_ctrl, style_colors, out_d
            )

        # 5. 全体マスタカーブプロット (線形 & 対数)
        plot_dimensionless_master_curves(df_p, df_bg, df_ctrl, style_colors, out_d)

        # 6. Side-by-side 比較プロット
        plot_side_by_side_comparison(df_p, df_bg, df_ctrl, style_colors, out_d)

    # 7. 生データの CSV エクスポート
    csv_out = workspace / "figure" / "dimensionless_local_polar" / "dimensionless_local_polar_data.csv"
    df_all = pd.concat([df_p, df_bg, df_ctrl], ignore_index=True)
    df_all.to_csv(csv_out, index=False)
    print(f"[INFO] Exported full processed dataset to {csv_out}")

    print("\n[SUCCESS] Dimensionless Local Polar Order analysis and plotting completed successfully!")


if __name__ == '__main__':
    main()
