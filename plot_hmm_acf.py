#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_hmm_acf.py
===============
HMM による Run / Tumble 状態別の自己相関関数（VACF, OACF, SACF）の
フィッティング・プロット・探索専用 CLI ツール。

フィッティングモデル（対数回帰、純粋指数非線形最小二乗、オフセット付き指数）、
フィッティング範囲（fit_max_lag_s, fit_min_lag_s）、表示範囲（xlim）、
重み付け（weighted）などを柔軟に指定して、高速にプロット・CSV出力できます。

使用例:
  # デフォルト（対数回帰、max_lag=60s）
  pixi run hmm_acf

  # 非線形最小二乗 (純粋指数) で fit_max_lag=20s に限定
  pixi run hmm_acf --model pure_exp --fit_max_lag_s 20.0 --xlim 30.0

  # オフセット付き指数モデルでフィッティング
  pixi run hmm_acf --model exp_offset --fit_max_lag_s 60.0

  # 特定の粒子径 (beads06um) のみ
  pixi run hmm_acf --beads beads06um
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from libs import hmm_cargo as hc

BEADS_INFO = [
    {"name": "beads06um", "diameter_um": 0.63, "marker": "^"},
    {"name": "beads1um", "diameter_um": 1.18, "marker": "o"},
    {"name": "beads3um", "diameter_um": 3.37, "marker": "d"},
    {"name": "beads5um", "diameter_um": 5.00, "marker": "p"},
    {"name": "beads7um", "diameter_um": 7.24, "marker": "s"},
    {"name": "beads20um", "diameter_um": 20.00, "marker": "h"},
]

POSSIBLE_ROOTS = [
    Path("/Volumes/data-1/Sasaki/MTsingleBeads"),
    Path("/Volumes/data-1/sasaki/MTsingleBeads"),
    Path("/Volumes/data/Sasaki/MTsingleBeads"),
    Path("/Volumes/data/sasaki/MTsingleBeads"),
]

STATE_COLORS = {0: "#e6550d", 1: "#2ca02c"}
STATE_NAMES = {0: "Tumble / Pause", 1: "Run"}


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
    csv_str = str(csv_path)
    if csv_str in _TRACKS_CACHE:
        return _TRACKS_CACHE[csv_str].copy()
    try:
        df = pd.read_csv(csv_path)
        _TRACKS_CACHE[csv_str] = df
        return df.copy()
    except Exception as e:
        print(f"[WARNING] Failed to load {csv_path}: {e}", flush=True)
    return None


def collect_bead_hmm_data(
    exp_dirs: List[Path],
    tau: int = 1,
    scale: float = 0.11,
    frame_interval: float = 4.0,
    epsilon: float = 1e-3,
):
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


def fit_acf_custom(
    df_corr: pd.DataFrame,
    model: str = "log_linear",
    fit_min_lag_s: float = 0.0,
    fit_max_lag_s: float = 60.0,
    min_points: int = 3,
    weighted: bool = True,
) -> dict:
    """
    指定されたモデルとラグ時間範囲で ACF をフィッティングする。

    Parameters
    ----------
    df_corr : pd.DataFrame
        'lag_time_s', 'corr', 'sem' (optional)
    model : str
        'log_linear' : ln C(tau) = -tau / tau_fit (ゼロ切片対数回帰)
        'pure_exp'   : C(tau) = exp(-tau / tau_fit) (非線形最小二乗)
        'exp_offset' : C(tau) = (1 - A) * exp(-tau / tau_fit) + A
    fit_min_lag_s : float
    fit_max_lag_s : float
    min_points : int
    weighted : bool
    """
    if df_corr is None or df_corr.empty or len(df_corr) < min_points:
        return {
            "tau_fit_s": np.nan, "tau_fit_err_s": np.nan,
            "offset_A": np.nan, "offset_A_err": np.nan,
            "r_squared": np.nan, "fit_t": np.array([]), "fit_corr": np.array([]),
            "count": 0, "model": model,
        }

    df_sub = df_corr[(df_corr["lag_time_s"] >= fit_min_lag_s) & (df_corr["lag_time_s"] <= fit_max_lag_s)].copy()
    t_data = df_sub["lag_time_s"].values
    c_data = df_sub["corr"].values

    valid_mask = np.isfinite(c_data) & np.isfinite(t_data)
    if np.sum(valid_mask) < min_points:
        return {
            "tau_fit_s": np.nan, "tau_fit_err_s": np.nan,
            "offset_A": np.nan, "offset_A_err": np.nan,
            "r_squared": np.nan, "fit_t": np.array([]), "fit_corr": np.array([]),
            "count": len(t_data), "model": model,
        }

    t_val = t_data[valid_mask]
    c_val = c_data[valid_mask]

    sigma = None
    if weighted and "sem" in df_sub.columns:
        s_val = df_sub.loc[valid_mask, "sem"].values
        if np.all(s_val > 0) and np.all(np.isfinite(s_val)):
            sigma = np.maximum(s_val, 1e-4)

    tau_fit = np.nan
    tau_err = np.nan
    offset_A = 0.0
    offset_A_err = 0.0
    r2 = np.nan
    fit_t = np.array([])
    fit_corr = np.array([])

    if model == "log_linear":
        # ゼロ切片対数回帰: ln(C) = -(1/tau) * t
        pos_mask = (c_val > 0) & (t_val > 0)
        if np.sum(pos_mask) >= min_points - 1:
            t_pos = t_val[pos_mask]
            c_pos = c_val[pos_mask]

            if sigma is not None:
                weights = 1.0 / np.maximum(sigma[pos_mask], 1e-4)
            else:
                weights = np.sqrt(c_pos)

            log_c = np.log(c_pos)
            denom = np.sum((weights * t_pos) ** 2)
            numer = np.sum((weights ** 2) * t_pos * log_c)
            slope = float(numer / denom) if denom > 0 else -1.0

            if slope < 0:
                tau_fit = float(-1.0 / slope)
                pred_log_c = slope * t_pos
                residuals = log_c - pred_log_c
                df_resid = max(1, len(t_pos) - 1)
                s_sq = np.sum(weights * (residuals ** 2)) / (np.sum(weights) * (df_resid / len(t_pos)) + 1e-12)
                se_slope = np.sqrt(s_sq / (denom + 1e-12))
                tau_err = float((tau_fit ** 2) * se_slope) if np.isfinite(se_slope) else np.nan

                pred_c = np.exp(-t_val / tau_fit)
                ss_res = np.sum((c_val - pred_c) ** 2)
                ss_tot = np.sum((c_val - np.mean(c_val)) ** 2)
                r2 = float(1.0 - ss_res / (ss_tot + 1e-12)) if ss_tot > 0 else np.nan

                fit_t = np.linspace(0, np.max(t_data) * 1.1, 150)
                fit_corr = np.exp(-fit_t / tau_fit)

    elif model == "pure_exp":
        # 非線形最小二乗 C(t) = exp(-t / tau)
        def exp_func(t, tau):
            return np.exp(-t / np.maximum(tau, 1e-6))

        try:
            popt, pcov = curve_fit(
                exp_func,
                t_val,
                c_val,
                p0=[15.0],
                bounds=(0.01, 1000.0),
                sigma=sigma,
                absolute_sigma=False if sigma is not None else False,
                maxfev=5000,
            )
            tau_fit = float(popt[0])
            perr = np.sqrt(np.diag(pcov)) if pcov is not None else [np.nan]
            tau_err = float(perr[0]) if np.isfinite(perr[0]) else np.nan

            pred_c = exp_func(t_val, tau_fit)
            ss_res = np.sum((c_val - pred_c) ** 2)
            ss_tot = np.sum((c_val - np.mean(c_val)) ** 2)
            r2 = float(1.0 - ss_res / (ss_tot + 1e-12)) if ss_tot > 0 else np.nan

            fit_t = np.linspace(0, np.max(t_data) * 1.1, 150)
            fit_corr = exp_func(fit_t, tau_fit)
        except Exception:
            pass

    elif model == "exp_offset":
        # C(t) = (1 - A) * exp(-t / tau) + A
        def exp_off_func(t, tau, A):
            return (1.0 - A) * np.exp(-t / np.maximum(tau, 1e-6)) + A

        tail_len = max(1, len(c_val) // 4)
        a_init = float(np.clip(np.mean(c_val[-tail_len:]), -0.1, 0.6))
        tau_init = 15.0

        try:
            popt, pcov = curve_fit(
                exp_off_func,
                t_val,
                c_val,
                p0=[tau_init, a_init],
                bounds=([0.01, -0.3], [1000.0, 0.95]),
                sigma=sigma,
                absolute_sigma=False if sigma is not None else False,
                maxfev=5000,
            )
            tau_fit = float(popt[0])
            offset_A = float(popt[1])
            perr = np.sqrt(np.diag(pcov)) if pcov is not None else [np.nan, np.nan]
            tau_err = float(perr[0]) if np.isfinite(perr[0]) else np.nan
            offset_A_err = float(perr[1]) if np.isfinite(perr[1]) else np.nan

            pred_c = exp_off_func(t_val, tau_fit, offset_A)
            ss_res = np.sum((c_val - pred_c) ** 2)
            ss_tot = np.sum((c_val - np.mean(c_val)) ** 2)
            r2 = float(1.0 - ss_res / (ss_tot + 1e-12)) if ss_tot > 0 else np.nan

            fit_t = np.linspace(0, np.max(t_data) * 1.1, 150)
            fit_corr = exp_off_func(fit_t, tau_fit, offset_A)
        except Exception:
            pass

    # 積分相関時間の計算
    int_res = hc.calc_integral_correlation_time(
        df_corr, max_lag_s=fit_max_lag_s, offset_A=offset_A if model == "exp_offset" else 0.0
    )

    return {
        "tau_fit_s": tau_fit,
        "tau_fit_err_s": tau_err,
        "offset_A": offset_A,
        "offset_A_err": offset_A_err,
        "r_squared": r2,
        "fit_t": fit_t,
        "fit_corr": fit_corr,
        "count": len(t_data),
        "tau_int_zero_s": int_res.get("tau_int_zero_s", np.nan),
        "tau_int_window_s": int_res.get("tau_int_window_s", np.nan),
        "t_zero_crossing_s": int_res.get("t_zero_crossing_s", np.nan),
        "model": model,
    }


def plot_acf_grid(
    fitted_results: Dict[str, dict],
    output_path: Path,
    xlim: float = 60.0,
    model_name: str = "log_linear",
    show_integral: bool = True,
):
    fig, axes = plt.subplots(3, 6, figsize=(22, 11.0), sharex=True)
    corr_types = [
        ("vacf", r"VACF $\langle \mathbf{v}(t)\cdot\mathbf{v}(t+\tau) \rangle / \langle v^2 \rangle$", "Velocity Vector"),
        ("oacf", r"OACF $\langle \hat{\mathbf{e}}(t)\cdot\hat{\mathbf{e}}(t+\tau) \rangle$", "Orientation Unit Vector"),
        ("sacf", r"SACF $\langle \delta v(t)\delta v(t+\tau) \rangle / \langle \delta v^2 \rangle$", "Speed Fluctuation"),
    ]

    for col_idx, binfo in enumerate(BEADS_INFO):
        bname = binfo["name"]
        dia = binfo["diameter_um"]

        if bname not in fitted_results or "autocorrelations" not in fitted_results[bname]:
            for row_idx in range(3):
                axes[row_idx, col_idx].set_visible(False)
            continue

        ac_dict = fitted_results[bname]["autocorrelations"]
        fits_dict = fitted_results[bname].get("autocorr_fits", {})

        for row_idx, (ctype, ctitle, clbl) in enumerate(corr_types):
            ax = axes[row_idx, col_idx]
            ax.axhline(0.0, color="gray", linestyle=":", lw=1.0, alpha=0.7)

            for s in [0, 1]:
                if s not in ac_dict or ctype not in ac_dict[s]:
                    continue
                df_c = ac_dict[s][ctype]
                if df_c.empty:
                    continue

                sub_c = df_c[df_c["lag_time_s"] <= xlim]
                s_lbl = STATE_NAMES.get(s, f"State {s}")
                col = STATE_COLORS.get(s, f"C{s}")
                mrk = "o" if s == 0 else "s"
                lsty = "--" if s == 0 else "-"

                fit_res = fits_dict.get(s, {}).get(ctype, {})
                tau_fit = fit_res.get("tau_fit_s", np.nan)
                tau_err = fit_res.get("tau_fit_err_s", np.nan)
                tau_int = fit_res.get("tau_int_zero_s", np.nan)
                r2 = fit_res.get("r_squared", np.nan)

                label_parts = [s_lbl]
                if not np.isnan(tau_fit):
                    if not np.isnan(tau_err) and tau_err < 100.0:
                        fit_str = f"$\\tau_{{\\mathrm{{fit}}}}={tau_fit:.1f}\\pm{tau_err:.1f}\\,\\mathrm{{s}}$"
                    else:
                        fit_str = f"$\\tau_{{\\mathrm{{fit}}}}={tau_fit:.1f}\\,\\mathrm{{s}}$"
                    label_parts.append(fit_str)
                if show_integral and not np.isnan(tau_int):
                    label_parts.append(f"$\\tau_{{\\mathrm{{int}}}}={tau_int:.1f}\\,\\mathrm{{s}}$")

                label_str = " (" + ", ".join(label_parts[1:]) + ")" if len(label_parts) > 1 else label_parts[0]
                label_str = f"{s_lbl}{label_str}"

                # データ点
                if "sem" in sub_c.columns and not sub_c["sem"].isna().all():
                    ax.errorbar(
                        sub_c["lag_time_s"], sub_c["corr"], yerr=sub_c["sem"],
                        fmt=mrk, color=col, ecolor=col,
                        markersize=4.0, capsize=2, label=label_str, zorder=3, alpha=0.85
                    )
                else:
                    ax.plot(
                        sub_c["lag_time_s"], sub_c["corr"],
                        marker=mrk, color=col, linestyle="",
                        markersize=4.0, label=label_str, zorder=3, alpha=0.85
                    )

                # フィッティング曲線
                fit_t = fit_res.get("fit_t", np.array([]))
                fit_corr = fit_res.get("fit_corr", np.array([]))
                if len(fit_t) > 0 and len(fit_corr) > 0 and not np.isnan(tau_fit):
                    mask_t = fit_t <= xlim
                    ax.plot(
                        fit_t[mask_t], fit_corr[mask_t],
                        color=col, linestyle=lsty, lw=1.8, alpha=0.9, zorder=4
                    )

            ax.grid(True, linestyle="--", alpha=0.4)
            ax.set_ylim(-0.25, 1.05)
            ax.set_xlim(0, xlim)

            if row_idx == 0:
                ax.set_title(f"$d = {dia:.2f}\\,\\mu\\mathrm{{m}}$", fontsize=12, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(ctitle, fontsize=10, fontweight="bold")
            if row_idx == 2:
                ax.set_xlabel(r"Lag Time $\tau$ [s]", fontsize=11)

            ax.legend(loc="upper right", fontsize=6.8, frameon=True, framealpha=0.92)

    model_title_dict = {
        "log_linear": r"Log-Linear Zero-Intercept Fit: $\ln C(\tau) = -\tau / \tau_{\mathrm{fit}}$",
        "pure_exp": r"Nonlinear Pure Exponential Fit: $C(\tau) = \exp(-\tau / \tau_{\mathrm{fit}})$",
        "exp_offset": r"Exponential with Offset Fit: $C(\tau) = (1-A)\exp(-\tau / \tau_{\mathrm{fit}}) + A$",
    }
    fig.suptitle(
        f"State-Dependent Autocorrelation Functions & {model_title_dict.get(model_name, model_name)}",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    png_path = output_path.with_suffix(".png")
    if png_path != output_path:
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {output_path} (and {png_path})", flush=True)


def plot_timescales_vs_diameter(
    df_autocorr_summary: pd.DataFrame,
    df_state_summary: pd.DataFrame,
    output_path: Path,
):
    if df_autocorr_summary.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    mode_colors = {"vacf": "#1f77b4", "oacf": "#2ca02c", "sacf": "#9467bd"}
    mode_labels = {"vacf": "Velocity (VACF)", "oacf": "Orientation (OACF)", "sacf": "Speed (SACF)"}
    mode_markers = {"vacf": "o", "oacf": "^", "sacf": "s"}

    # (a) Run
    ax0 = axes[0]
    for mode in ["vacf", "oacf", "sacf"]:
        sub = df_autocorr_summary[(df_autocorr_summary["state"] == 1) & (df_autocorr_summary["mode"] == mode)].sort_values(by="diameter_um")
        if sub.empty:
            continue
        y_val = sub["tau_fit_s"].values
        y_err = sub["tau_fit_err_s"].values if "tau_fit_err_s" in sub.columns else None
        if y_err is not None:
            y_err = np.where((y_err > 5.0 * y_val) | (y_err > 50.0) | np.isnan(y_err), np.nan, y_err)
        ax0.errorbar(
            sub["diameter_um"], y_val, yerr=y_err,
            marker=mode_markers[mode], color=mode_colors[mode], lw=2.2,
            markersize=7.0, capsize=3.5, label=f"{mode_labels[mode]} $\\tau_{{\\mathrm{{fit}}}}$", zorder=4
        )
        if "tau_int_zero_s" in sub.columns:
            ax0.plot(
                sub["diameter_um"], sub["tau_int_zero_s"],
                marker=mode_markers[mode], markerfacecolor="none", color=mode_colors[mode],
                linestyle=":", lw=1.5, markersize=6.5, label=f"{mode_labels[mode]} $\\tau_{{\\mathrm{{int}}}}$", zorder=3
            )

    if not df_state_summary.empty:
        df_run_state = df_state_summary[df_state_summary["state"] == 1].sort_values(by="diameter_um")
        if not df_run_state.empty and "tau_ccdf_s" in df_run_state.columns:
            ax0.errorbar(
                df_run_state["diameter_um"], df_run_state["tau_ccdf_s"], yerr=df_run_state.get("tau_ccdf_err_s", None),
                fmt="--d", color="#e41a1c", lw=2.0, capsize=4, markersize=7,
                label=r"Run Dwell $\tau_{\mathrm{Run}}^{\mathrm{dwell}}$", zorder=5
            )

    ax0.set_xscale("log")
    ax0.set_yscale("log")
    ax0.set_xlabel(r"Cargo Particle Diameter $d$ [$\mu\mathrm{m}$]", fontsize=11)
    ax0.set_ylabel(r"Relaxation Time $\tau$ [s]", fontsize=11)
    ax0.set_title(r"(a) Run State Correlation Timescales", fontsize=12, fontweight="bold")
    ax0.set_xticks([0.63, 1.18, 3.37, 5.0, 7.24, 20.0])
    ax0.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax0.grid(True, which="both", linestyle="--", alpha=0.4)
    ax0.legend(loc="best", fontsize=7.8, frameon=True, framealpha=0.92)

    # (b) Tumble
    ax1 = axes[1]
    for mode in ["vacf", "oacf", "sacf"]:
        sub = df_autocorr_summary[(df_autocorr_summary["state"] == 0) & (df_autocorr_summary["mode"] == mode)].sort_values(by="diameter_um")
        if sub.empty:
            continue
        y_val = sub["tau_fit_s"].values
        y_err = sub["tau_fit_err_s"].values if "tau_fit_err_s" in sub.columns else None
        if y_err is not None:
            y_err = np.where((y_err > 5.0 * y_val) | (y_err > 50.0) | np.isnan(y_err), np.nan, y_err)
        ax1.errorbar(
            sub["diameter_um"], y_val, yerr=y_err,
            marker=mode_markers[mode], color=mode_colors[mode], lw=2.2,
            markersize=7.0, capsize=3.5, label=f"{mode_labels[mode]} $\\tau_{{\\mathrm{{fit}}}}$", zorder=4
        )
        if "tau_int_zero_s" in sub.columns:
            ax1.plot(
                sub["diameter_um"], sub["tau_int_zero_s"],
                marker=mode_markers[mode], markerfacecolor="none", color=mode_colors[mode],
                linestyle=":", lw=1.5, markersize=6.5, label=f"{mode_labels[mode]} $\\tau_{{\\mathrm{{int}}}}$", zorder=3
            )

    if not df_state_summary.empty:
        df_tum_state = df_state_summary[df_state_summary["state"] == 0].sort_values(by="diameter_um")
        if not df_tum_state.empty and "tau_ccdf_s" in df_tum_state.columns:
            ax1.errorbar(
                df_tum_state["diameter_um"], df_tum_state["tau_ccdf_s"], yerr=df_tum_state.get("tau_ccdf_err_s", None),
                fmt="--d", color="#e41a1c", lw=2.0, capsize=4, markersize=7,
                label=r"Tumble Dwell $\tau_{\mathrm{Tumble}}^{\mathrm{dwell}}$", zorder=5
            )

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"Cargo Particle Diameter $d$ [$\mu\mathrm{m}$]", fontsize=11)
    ax1.set_ylabel(r"Relaxation Time $\tau$ [s]", fontsize=11)
    ax1.set_title(r"(b) Tumble State Correlation Timescales", fontsize=12, fontweight="bold")
    ax1.set_xticks([0.63, 1.18, 3.37, 5.0, 7.24, 20.0])
    ax1.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax1.grid(True, which="both", linestyle="--", alpha=0.4)
    ax1.legend(loc="best", fontsize=7.8, frameon=True, framealpha=0.92)

    # (c) Ratio
    ax2 = axes[2]
    sub_run_v = df_autocorr_summary[(df_autocorr_summary["state"] == 1) & (df_autocorr_summary["mode"] == "vacf")].set_index("bead_name")
    sub_tum_v = df_autocorr_summary[(df_autocorr_summary["state"] == 0) & (df_autocorr_summary["mode"] == "vacf")].set_index("bead_name")
    sub_run_o = df_autocorr_summary[(df_autocorr_summary["state"] == 1) & (df_autocorr_summary["mode"] == "oacf")].set_index("bead_name")
    sub_tum_o = df_autocorr_summary[(df_autocorr_summary["state"] == 0) & (df_autocorr_summary["mode"] == "oacf")].set_index("bead_name")

    dias = []
    ratio_vacf_fit = []
    ratio_oacf_fit = []
    ratio_vacf_int = []
    ratio_oacf_int = []
    for binfo in BEADS_INFO:
        bn = binfo["name"]
        if bn in sub_run_v.index and bn in sub_tum_v.index:
            tv_r = sub_run_v.loc[bn, "tau_fit_s"]
            tv_t = sub_tum_v.loc[bn, "tau_fit_s"]
            to_r = sub_run_o.loc[bn, "tau_fit_s"] if bn in sub_run_o.index else np.nan
            to_t = sub_tum_o.loc[bn, "tau_fit_s"] if bn in sub_tum_o.index else np.nan

            tv_r_int = sub_run_v.loc[bn, "tau_int_zero_s"] if "tau_int_zero_s" in sub_run_v.columns else np.nan
            tv_t_int = sub_tum_v.loc[bn, "tau_int_zero_s"] if "tau_int_zero_s" in sub_tum_v.columns else np.nan
            to_r_int = sub_run_o.loc[bn, "tau_int_zero_s"] if (bn in sub_run_o.index and "tau_int_zero_s" in sub_run_o.columns) else np.nan
            to_t_int = sub_tum_o.loc[bn, "tau_int_zero_s"] if (bn in sub_tum_o.index and "tau_int_zero_s" in sub_tum_o.columns) else np.nan

            dias.append(binfo["diameter_um"])
            ratio_vacf_fit.append(tv_r / (tv_t + 1e-12) if not np.isnan(tv_r) and not np.isnan(tv_t) else np.nan)
            ratio_oacf_fit.append(to_r / (to_t + 1e-12) if not np.isnan(to_r) and not np.isnan(to_t) else np.nan)
            ratio_vacf_int.append(tv_r_int / (tv_t_int + 1e-12) if not np.isnan(tv_r_int) and not np.isnan(tv_t_int) else np.nan)
            ratio_oacf_int.append(to_r_int / (to_t_int + 1e-12) if not np.isnan(to_r_int) and not np.isnan(to_t_int) else np.nan)

    ax2.axhline(1.0, color="gray", linestyle=":", lw=1.2, label="Equal Ratio (1.0)")
    if dias:
        ax2.plot(dias, ratio_vacf_int, marker="o", color="#1f77b4", lw=2.0, markersize=7, label=r"VACF Integral Ratio $\tau_{\mathrm{int}}^{\mathrm{Run}} / \tau_{\mathrm{int}}^{\mathrm{Tumble}}$")
        ax2.plot(dias, ratio_oacf_int, marker="^", color="#2ca02c", lw=2.0, markersize=7, label=r"OACF Integral Ratio $\tau_{\mathrm{int}}^{\mathrm{Run}} / \tau_{\mathrm{int}}^{\mathrm{Tumble}}$")
        ax2.plot(dias, ratio_vacf_fit, marker="o", markerfacecolor="none", color="#1f77b4", linestyle="--", lw=1.5, markersize=6.5, label=r"VACF Fit Ratio $\tau_{\mathrm{fit}}^{\mathrm{Run}} / \tau_{\mathrm{fit}}^{\mathrm{Tumble}}$")
        ax2.plot(dias, ratio_oacf_fit, marker="^", markerfacecolor="none", color="#2ca02c", linestyle="--", lw=1.5, markersize=6.5, label=r"OACF Fit Ratio $\tau_{\mathrm{fit}}^{\mathrm{Run}} / \tau_{\mathrm{fit}}^{\mathrm{Tumble}}$")

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"Cargo Particle Diameter $d$ [$\mu\mathrm{m}$]", fontsize=11)
    ax2.set_ylabel(r"Correlation Time Ratio $\tau_{\mathrm{Run}} / \tau_{\mathrm{Tumble}}$", fontsize=11)
    ax2.set_title(r"(c) Run / Tumble Persistence Ratio", fontsize=12, fontweight="bold")
    ax2.set_xticks([0.63, 1.18, 3.37, 5.0, 7.24, 20.0])
    ax2.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax2.grid(True, which="both", linestyle="--", alpha=0.4)
    ax2.legend(loc="best", fontsize=7.5, frameon=True, framealpha=0.92)

    fig.suptitle(
        r"Characteristic Autocorrelation Timescales $\tau_{\mathrm{fit}}$ & $\tau_{\mathrm{int}}$ Across Cargo Diameters",
        fontsize=13.5,
        fontweight="bold",
    )
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    png_path = output_path.with_suffix(".png")
    if png_path != output_path:
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {output_path} (and {png_path})", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Fast Run & Tumble Autocorrelation Function (VACF, OACF, SACF) Fitting & Exploration CLI"
    )
    parser.add_argument("--root_dir", type=str, default=None, help="Root directory containing bead experiments")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: figure/hmm_acf or figure/hmm_1d)")
    parser.add_argument("--beads", type=str, default="all", help="Target beads (all, beads06um, beads1um, ...)")
    parser.add_argument(
        "--model",
        type=str,
        default="log_linear",
        choices=["log_linear", "pure_exp", "exp_offset"],
        help="Fitting model: 'log_linear' (zero-intercept log regression), 'pure_exp' (nonlinear curve_fit C=exp(-t/tau)), 'exp_offset' (C=(1-A)exp(-t/tau)+A)",
    )
    parser.add_argument("--fit_min_lag_s", type=float, default=0.0, help="Minimum lag time for fitting in seconds (default: 0.0)")
    parser.add_argument("--fit_max_lag_s", type=float, default=60.0, help="Maximum lag time for fitting in seconds (default: 60.0)")
    parser.add_argument("--xlim", type=float, default=60.0, help="Maximum lag time to display on plots in seconds (default: 60.0)")
    parser.add_argument("--weighted", action="store_true", default=True, help="Use 1/SEM^2 weighting (default: True)")
    parser.add_argument("--no-weighted", dest="weighted", action="store_false", help="Disable SEM weighting")
    parser.add_argument("--show_integral", action="store_true", default=True, help="Show integral correlation time in legend (default: True)")
    parser.add_argument("--no_show_integral", dest="show_integral", action="store_false", help="Hide integral correlation time in legend")
    parser.add_argument("--save_csv", action="store_true", default=True, help="Save summary CSV")

    args = parser.parse_args()

    root_dir = Path(args.root_dir) if args.root_dir else find_default_root()
    output_dir = Path(args.output_dir) if args.output_dir else (root_dir / "figure" / "hmm_1d")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("      Run & Tumble Autocorrelation Fitting & Exploration CLI     ")
    print("=================================================================")
    print(f"Root dir:         {root_dir}")
    print(f"Output dir:       {output_dir}")
    print(f"Target beads:     {args.beads}")
    print(f"Fitting model:    {args.model}")
    print(f"Fit lag range:    [{args.fit_min_lag_s:.1f} s, {args.fit_max_lag_s:.1f} s]")
    print(f"Plot xlim:        [0, {args.xlim:.1f} s]")
    print(f"Weighted fit:     {args.weighted}")
    print("=================================================================\n")

    if args.beads == "all":
        target_bead_infos = BEADS_INFO
    else:
        target_bead_infos = [b for b in BEADS_INFO if b["name"] == args.beads]
        if not target_bead_infos:
            print(f"[ERROR] Unknown bead name '{args.beads}'. Available: {[b['name'] for b in BEADS_INFO]}")
            return

    fitted_results = {}
    all_autocorr_records = []

    for binfo in target_bead_infos:
        bname = binfo["name"]
        dia = binfo["diameter_um"]
        print(f"--- Loading HMM tracks for {bname} (d={dia:.2f} um) ---")

        edirs = find_experiment_dirs(root_dir, bname)
        if not edirs:
            print(f"  [WARNING] No experiment dirs found for {bname}. Skipping.")
            continue

        X, lengths, df_obs = collect_bead_hmm_data(
            edirs,
            tau=1,
            scale=0.11,
            frame_interval=4.0,
            epsilon=1e-3,
        )
        if len(X) < 20:
            continue

        # 高速 HMM 予測 (2状態)
        init_means = np.array([[-3.0], [0.0]]) if bname == "beads5um" else np.array([[-3.5], [0.0]])
        init_covars = np.array([[[1.0]], [[0.1]]]) if bname == "beads5um" else None
        init_startprob = np.array([0.8, 0.2]) if bname == "beads5um" else None
        init_transmat = np.array([[0.95, 0.05], [0.1, 0.9]]) if bname == "beads5um" else None

        hmm_model = hc.CargoGaussianHMM(
            n_components=2,
            covariance_type="full",
            epsilon=1e-3,
            random_state=42,
            init_means=init_means,
            init_covars=init_covars,
            init_transmat=init_transmat,
            init_startprob=init_startprob,
        )
        hmm_model.fit(X, lengths=lengths)
        raw_pred_states = hmm_model.predict(X, lengths=lengths)
        pred_states = hc.filter_state_glitches(raw_pred_states, lengths, min_duration_frames=2)
        df_obs["pred_state"] = pred_states

        # 自己相関データの算出
        autocorr_data = hc.calc_state_dependent_autocorrelations(
            df_obs,
            n_components=2,
            frame_interval=4.0,
            max_lag_frames=int(np.ceil(args.xlim / 4.0)) + 2,
        )

        autocorr_fits = {}
        for s in [0, 1]:
            autocorr_fits[s] = {}
            s_label = STATE_NAMES.get(s, f"State {s}")
            for ctype in ["vacf", "oacf", "sacf"]:
                if s in autocorr_data and ctype in autocorr_data[s]:
                    df_c = autocorr_data[s][ctype]
                    f_res = fit_acf_custom(
                        df_c,
                        model=args.model,
                        fit_min_lag_s=args.fit_min_lag_s,
                        fit_max_lag_s=args.fit_max_lag_s,
                        weighted=args.weighted,
                    )
                    autocorr_fits[s][ctype] = f_res
                    all_autocorr_records.append({
                        "bead_name": bname,
                        "diameter_um": dia,
                        "state": s,
                        "state_label": s_label,
                        "mode": ctype,
                        "fit_model": args.model,
                        "fit_min_lag_s": args.fit_min_lag_s,
                        "fit_max_lag_s": args.fit_max_lag_s,
                        "tau_fit_s": f_res.get("tau_fit_s", np.nan),
                        "tau_fit_err_s": f_res.get("tau_fit_err_s", np.nan),
                        "offset_A": f_res.get("offset_A", np.nan),
                        "offset_A_err": f_res.get("offset_A_err", np.nan),
                        "r_squared": f_res.get("r_squared", np.nan),
                        "tau_int_zero_s": f_res.get("tau_int_zero_s", np.nan),
                        "tau_int_window_s": f_res.get("tau_int_window_s", np.nan),
                        "t_zero_crossing_s": f_res.get("t_zero_crossing_s", np.nan),
                        "count": f_res.get("count", 0),
                    })
                    tau_val = f_res.get("tau_fit_s", np.nan)
                    tau_err = f_res.get("tau_fit_err_s", np.nan)
                    tau_int = f_res.get("tau_int_zero_s", np.nan)
                    print(f"    [{s_label}] {ctype.upper()}: tau_fit = {tau_val:.2f} ± {tau_err:.2f} s (tau_int = {tau_int:.2f} s, R2 = {f_res.get('r_squared', np.nan):.3f})")

        fitted_results[bname] = {
            "autocorrelations": autocorr_data,
            "autocorr_fits": autocorr_fits,
        }

    # プロット生成
    df_autocorr_summary = pd.DataFrame(all_autocorr_records)
    tag = f"{args.model}_fit{int(args.fit_max_lag_s)}s"

    fig_grid_path = output_dir / f"hmm_acf_grid_{tag}.svg"
    plot_acf_grid(
        fitted_results,
        fig_grid_path,
        xlim=args.xlim,
        model_name=args.model,
        show_integral=args.show_integral,
    )

    if len(fitted_results) > 1:
        fig_tau_path = output_dir / f"hmm_acf_timescales_vs_diameter_{tag}.svg"
        df_state_dummy = pd.DataFrame()
        plot_timescales_vs_diameter(df_autocorr_summary, df_state_dummy, fig_tau_path)

    # CSV 出力
    if args.save_csv and not df_autocorr_summary.empty:
        csv_path = output_dir / f"hmm_acf_summary_{tag}.csv"
        df_autocorr_summary.to_csv(csv_path, index=False)
        print(f"[SAVED] {csv_path}")

    print("\n=================================================================")
    print("        ACF Fitting & Plotting Completed Successfully!           ")
    print("=================================================================")


if __name__ == "__main__":
    main()
