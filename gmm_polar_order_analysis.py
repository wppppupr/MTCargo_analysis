#!/usr/bin/env python3
"""
gmm_polar_order_analysis.py

小貨物（0.63 µm, 1.18 µm）について、ln(v(t) / u_inf(t)) の 1D GMM で分類した
Run（高速状態）と Tumble（低速状態）それぞれにおける「貨物をマスクした近傍ポーラーオーダー」を
算出・比較・可視化するスクリプトです。

使用する2種類の貨物マスク付きポーラーオーダー（他コードと共通定義）:
  1. 足印アニュラス (RAD_MAP[bead] < r <= BEST_MAP[bead], 貨物内部 r <= R_mask を除外):
     - beads06um: 3 < r <= 5 px (0.33 - 0.55 um)
     - beads1um:  6 < r <= 10 px (0.66 - 1.10 um)
     (scratch/extract_footprint_heterogeneity.py と同一定義)
  2. 窓半径依存ポーラーオーダー Phi_masked(R) (local_polar_w.zarr, particle_radius=3/6 px マスク済み,
     R > particle_radius の有効窓サイズ、および背景 local_polar_bg.zarr との比較)

出力先:
  - figure/gmm_1d/
  - /mnt/NAS-Ebanaru/sasaki/MTsingleBeads/figure/gmm_1d/
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from libs import gmm_cargo as gc
from libs.experiment_cutoff import load_strict_active_particles
from gmm_cargo_analysis import (
    SMALL_BEADS_INFO,
    collect_bead_gmm_data,
    find_default_root,
    find_experiment_dirs,
    save_csv_to_dirs,
    save_fig_to_dirs,
)


def summarize_footprint_by_state(
    df_merged: pd.DataFrame,
    bead_name: str,
    diameter_um: float,
    n_boot: int = 2000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run / Tumble / All ごとの足印指標サマリーと貨物単位ペア比較テーブルを返す。"""
    metrics = ["phi_t", "f_opp", "u_fp_norm", "eta", "cos_psi", "v_norm"]
    summary_rows = []
    for st_val, st_lbl in [(0, "Tumble"), (1, "Run"), (-1, "All")]:
        sub = df_merged if st_val == -1 else df_merged[df_merged["pred_state"] == st_val]
        for col in metrics:
            st_stats = gc.hierarchical_bootstrap_ci(sub, col, n_boot=n_boot, seed=42)
            summary_rows.append({
                "bead_name": bead_name,
                "diameter_um": diameter_um,
                "state": st_val,
                "state_label": st_lbl,
                "metric": col,
                **st_stats,
            })

    # Per-cargo paired table
    p_col = "orig_particle" if "orig_particle" in df_merged.columns else "particle"
    cargo_rows = []
    for (exp_dir, pid), grp in df_merged.groupby(["exp_dir", p_col], sort=False):
        g_tum = grp[grp["pred_state"] == 0]
        g_run = grp[grp["pred_state"] == 1]
        phi_tum = float(g_tum["phi_t"].mean()) if not g_tum.empty else np.nan
        phi_run = float(g_run["phi_t"].mean()) if not g_run.empty else np.nan
        cargo_rows.append({
            "bead_name": bead_name,
            "diameter_um": diameter_um,
            "exp_dir": exp_dir,
            "particle": int(pid),
            "n_total": int(len(grp)),
            "n_tumble": int(len(g_tum)),
            "n_run": int(len(g_run)),
            "phi_fp_all": float(grp["phi_t"].mean()),
            "phi_fp_tumble": phi_tum,
            "phi_fp_run": phi_run,
            "delta_phi_run_minus_tumble": (phi_run - phi_tum) if (np.isfinite(phi_run) and np.isfinite(phi_tum)) else np.nan,
            "f_opp_tumble": float(g_tum["f_opp"].mean()) if not g_tum.empty else np.nan,
            "f_opp_run": float(g_run["f_opp"].mean()) if not g_run.empty else np.nan,
            "u_fp_norm_tumble": float(g_tum["u_fp_norm"].mean()) if not g_tum.empty else np.nan,
            "u_fp_norm_run": float(g_run["u_fp_norm"].mean()) if not g_run.empty else np.nan,
            "cos_psi_tumble": float(g_tum["cos_psi"].mean()) if not g_tum.empty else np.nan,
            "cos_psi_run": float(g_run["cos_psi"].mean()) if not g_run.empty else np.nan,
        })

    return pd.DataFrame(summary_rows), pd.DataFrame(cargo_rows)


def summarize_radial_curves(
    df_rad: pd.DataFrame,
    df_bg: pd.DataFrame,
    bead_name: str,
    diameter_um: float,
    n_boot: int = 1000,
) -> pd.DataFrame:
    """窓半径 R ごとの Run / Tumble / All / Background ポーラーオーダー曲線を集計する。"""
    if df_rad.empty:
        return pd.DataFrame()

    rows = []
    for (w_px, w_um), w_grp in df_rad.groupby(["window_px", "window_um"], sort=True):
        for st_val, st_lbl in [(0, "Tumble"), (1, "Run"), (-1, "All")]:
            sub = w_grp if st_val == -1 else w_grp[w_grp["pred_state"] == st_val]
            ci = gc.hierarchical_bootstrap_ci(sub, "phi", n_boot=n_boot, seed=42)
            rows.append({
                "bead_name": bead_name,
                "diameter_um": diameter_um,
                "window_px": int(w_px),
                "window_um": float(w_um),
                "mode": st_lbl,
                **ci,
            })

        # Paired difference per window
        piv = w_grp.groupby(["exp_dir", "orig_particle", "pred_state"])["phi"].mean().unstack()
        if 0 in piv.columns and 1 in piv.columns:
            diff = (piv[1] - piv[0]).dropna()
            if len(diff) >= 2:
                d_mean = float(diff.mean())
                d_sem = float(diff.std(ddof=1) / np.sqrt(len(diff)))
                rng = np.random.default_rng(42)
                b_means = [float(rng.choice(diff.to_numpy(), size=len(diff), replace=True).mean()) for _ in range(n_boot)]
                rows.append({
                    "bead_name": bead_name,
                    "diameter_um": diameter_um,
                    "window_px": int(w_px),
                    "window_um": float(w_um),
                    "mode": "Delta_Run_minus_Tumble",
                    "mean": d_mean,
                    "median": float(diff.median()),
                    "std": float(diff.std(ddof=1)),
                    "sem": d_sem,
                    "ci_lo": float(np.percentile(b_means, 2.5)),
                    "ci_hi": float(np.percentile(b_means, 97.5)),
                    "n_obs": int(len(diff)),
                    "n_cargos": int(len(diff)),
                    "n_movies": int(w_grp["exp_dir"].nunique()),
                })

    if not df_bg.empty:
        for (w_px, w_um), grp_bg in df_bg.groupby(["window_px", "window_um"], sort=True):
            vals = grp_bg["phi_mean"].to_numpy(dtype=float)
            w_cnt = grp_bg["n_frames"].to_numpy(dtype=float)
            bg_mean = float(np.average(vals, weights=w_cnt))
            bg_sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            rows.append({
                "bead_name": bead_name,
                "diameter_um": diameter_um,
                "window_px": int(w_px),
                "window_um": float(w_um),
                "mode": "Background",
                "mean": bg_mean,
                "median": float(np.median(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "sem": bg_sem,
                "ci_lo": bg_mean - 1.96 * bg_sem,
                "ci_hi": bg_mean + 1.96 * bg_sem,
                "n_obs": int(w_cnt.sum()),
                "n_cargos": 0,
                "n_movies": int(len(vals)),
            })

    return pd.DataFrame(rows)


def compute_binned_polar_vs_run(
    merged_by_bead: Dict[str, pd.DataFrame],
    beads_info: List[dict],
    n_bins: int = 8,
    n_boot: int = 1000,
) -> pd.DataFrame:
    """1 - Phi_fp の分位ビンごとに Run 占有率 P(Run) と規格化速度中央値 v/u_inf(t) を算出する。"""
    rows = []
    rng = np.random.default_rng(42)
    for binfo in beads_info:
        bname = binfo["name"]
        dia = binfo["diameter_um"]
        df = merged_by_bead.get(bname)
        if df is None or df.empty:
            continue

        work = df.copy()
        work["one_minus_phi"] = np.clip(1.0 - work["phi_t"], 1e-4, 1.0)
        work["is_run"] = (work["pred_state"] == 1).astype(float)
        # Rank-based quantile binning to handle ties near Phi=1
        ranks = work["one_minus_phi"].rank(method="first")
        work["bin_id"] = pd.qcut(ranks, q=n_bins, labels=False)

        for bid, grp in work.groupby("bin_id", sort=True):
            p_ids = grp.groupby(["exp_dir", "orig_particle"])
            c_keys = list(p_ids.groups.keys())
            n_c = len(c_keys)
            x_mean = float(grp["one_minus_phi"].mean())
            phi_mean = float(grp["phi_t"].mean())
            p_run = float(grp["is_run"].mean())
            v_med = float(grp["v_norm"].median())

            if n_c >= 2:
                c_runs = np.array([p_ids.get_group(k)["is_run"].sum() for k in c_keys], dtype=float)
                c_cnts = np.array([len(p_ids.get_group(k)) for k in c_keys], dtype=float)
                boot_p = []
                for _ in range(n_boot):
                    idx = rng.integers(0, n_c, size=n_c)
                    boot_p.append(c_runs[idx].sum() / c_cnts[idx].sum())
                p_lo, p_hi = [float(v) for v in np.percentile(boot_p, [2.5, 97.5])]
            else:
                p_lo = p_hi = p_run

            rows.append({
                "bead_name": bname,
                "diameter_um": dia,
                "bin_id": int(bid),
                "n_obs": int(len(grp)),
                "n_cargos": n_c,
                "mean_one_minus_phi": x_mean,
                "mean_phi_fp": phi_mean,
                "p_run": p_run,
                "p_run_ci_lo": p_lo,
                "p_run_ci_hi": p_hi,
                "median_v_norm": v_med,
            })
    return pd.DataFrame(rows)


def plot_footprint_run_vs_tumble(
    merged_by_bead: Dict[str, pd.DataFrame],
    df_sum: pd.DataFrame,
    df_cargo: pd.DataFrame,
    beads_info: List[dict],
    out_dirs: List[Path],
):
    """足印アニュラス（貨物マスク済み）における Run vs Tumble の比較図（4パネル）。"""
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.6))
    bnames = [b["name"] for b in beads_info if b["name"] in merged_by_bead]
    x_pos = np.arange(len(bnames))
    x_labels = [rf"${next(b['diameter_um'] for b in beads_info if b['name'] == bn):.2f}\,\mu\mathrm{{m}}$" for bn in bnames]

    # (A) Footprint Phi_fp distribution (boxplot + mean marker)
    ax = axes[0, 0]
    w = 0.34
    for i, bn in enumerate(bnames):
        df = merged_by_bead[bn]
        v_tum = df.loc[df["pred_state"] == 0, "phi_t"].to_numpy()
        v_run = df.loc[df["pred_state"] == 1, "phi_t"].to_numpy()
        bp0 = ax.boxplot(v_tum, positions=[i - w / 2], widths=0.26, patch_artist=True, showfliers=False)
        bp1 = ax.boxplot(v_run, positions=[i + w / 2], widths=0.26, patch_artist=True, showfliers=False)
        for box in bp0["boxes"]:
            box.set(facecolor=gc.STATE_COLORS[0], alpha=0.55)
        for box in bp1["boxes"]:
            box.set(facecolor=gc.STATE_COLORS[1], alpha=0.55)
        for med in bp0["medians"] + bp1["medians"]:
            med.set(color="#111111", lw=1.8)
        ax.scatter([i - w / 2], [np.mean(v_tum)], color=gc.STATE_COLORS[0], edgecolor="black", s=55, zorder=5, label="Tumble mean" if i == 0 else None)
        ax.scatter([i + w / 2], [np.mean(v_run)], color=gc.STATE_COLORS[1], edgecolor="black", s=55, zorder=5, label="Run mean" if i == 0 else None)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0.45, 1.03)
    ax.set_ylabel(r"Footprint Polar Order $\Phi_{\mathrm{fp}}$", fontsize=11)
    ax.set_title(r"(A) $\Phi_{\mathrm{fp}}$ Distribution (Tumble vs Run)", fontweight="bold")
    ax.legend(loc="lower left", fontsize=8.8, framealpha=0.9)
    ax.grid(True, axis="y", ls="--", alpha=0.35)

    # (B) Hierarchical Mean ± 95% CI & Paired Per-Cargo Lines
    ax = axes[0, 1]
    for i, bn in enumerate(bnames):
        c_sub = df_cargo[df_cargo["bead_name"] == bn].dropna(subset=["phi_fp_tumble", "phi_fp_run"])
        for _, r in c_sub.iterrows():
            col_line = "#555555" if r["delta_phi_run_minus_tumble"] >= 0 else "#cc3333"
            ax.plot([i - w / 2, i + w / 2], [r["phi_fp_tumble"], r["phi_fp_run"]], color=col_line, alpha=0.35, lw=1.1, zorder=2)

        s_tum = df_sum[(df_sum["bead_name"] == bn) & (df_sum["metric"] == "phi_t") & (df_sum["state"] == 0)].iloc[0]
        s_run = df_sum[(df_sum["bead_name"] == bn) & (df_sum["metric"] == "phi_t") & (df_sum["state"] == 1)].iloc[0]
        ax.errorbar([i - w / 2], [s_tum["mean"]], yerr=[[s_tum["mean"] - s_tum["ci_lo"]], [s_tum["ci_hi"] - s_tum["mean"]]],
                    fmt="s", color=gc.STATE_COLORS[0], ms=8, markeredgecolor="black", capsize=4, lw=1.8, zorder=5, label="Tumble (95% CI)" if i == 0 else None)
        ax.errorbar([i + w / 2], [s_run["mean"]], yerr=[[s_run["mean"] - s_run["ci_lo"]], [s_run["ci_hi"] - s_run["mean"]]],
                    fmt="o", color=gc.STATE_COLORS[1], ms=8, markeredgecolor="black", capsize=4, lw=1.8, zorder=5, label="Run (95% CI)" if i == 0 else None)

        if len(c_sub) >= 2:
            _, pval = stats.wilcoxon(c_sub["phi_fp_run"] - c_sub["phi_fp_tumble"])
            d_mean = float((c_sub["phi_fp_run"] - c_sub["phi_fp_tumble"]).mean())
            n_pos = int((c_sub["delta_phi_run_minus_tumble"] > 0).sum())
            ax.text(i, 1.005, rf"$\Delta\Phi={d_mean:+.3f}$ ({n_pos}/{len(c_sub)}, $p={pval:.1e}$)",
                    ha="center", va="bottom", fontsize=8.3, bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="#cccccc"))

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0.75, 1.045)
    ax.set_ylabel(r"Mean $\Phi_{\mathrm{fp}}$ (95% CI)", fontsize=11)
    ax.set_title(r"(B) Paired Cargo: $\Phi_{\mathrm{fp,Tumble}} \to \Phi_{\mathrm{fp,Run}}$", fontweight="bold")
    ax.legend(loc="lower left", fontsize=8.8, framealpha=0.9)
    ax.grid(True, axis="y", ls="--", alpha=0.35)

    # (C) Normalized footprint flow speed |<u>_fp| / u_inf(t) & f_opp
    ax = axes[1, 0]
    for st_val, st_lbl, offset, col in [(0, "Tumble", -w / 2, gc.STATE_COLORS[0]), (1, "Run", +w / 2, gc.STATE_COLORS[1])]:
        means, lo_errs, hi_errs = [], [], []
        for bn in bnames:
            row = df_sum[(df_sum["bead_name"] == bn) & (df_sum["metric"] == "u_fp_norm") & (df_sum["state"] == st_val)].iloc[0]
            means.append(row["mean"])
            lo_errs.append(row["mean"] - row["ci_lo"])
            hi_errs.append(row["ci_hi"] - row["mean"])
        ax.bar(x_pos + offset, means, width=w, color=col, alpha=0.85, yerr=[lo_errs, hi_errs], capsize=4, label=st_lbl)
    ax.axhline(1.0, color="#3366aa", ls="--", lw=1.2, label=r"$|\langle \mathbf{u} \rangle_{\mathrm{fp}}| = u_\infty(t)$")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel(r"$|\langle \mathbf{u} \rangle_{\mathrm{fp}}| \,/\, u_\infty(t)$", fontsize=11)
    ax.set_title("(C) Masked Footprint Flow Speed by State", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8.6, framealpha=0.9)
    ax.grid(True, axis="y", ls="--", alpha=0.35)

    # (D) Alignment cos psi & coupling efficiency eta by state
    ax = axes[1, 1]
    w4 = 0.18
    configs = [
        ("cos_psi", 0, -1.5 * w4, gc.STATE_COLORS[0], 0.55, r"$\cos\psi$ (Tumble)"),
        ("cos_psi", 1, -0.5 * w4, gc.STATE_COLORS[1], 0.55, r"$\cos\psi$ (Run)"),
        ("eta", 0, +0.5 * w4, gc.STATE_COLORS[0], 0.92, r"$\eta$ (Tumble)"),
        ("eta", 1, +1.5 * w4, gc.STATE_COLORS[1], 0.92, r"$\eta$ (Run)"),
    ]
    for metric, st_val, offset, col, alp, lbl in configs:
        means, lo_errs, hi_errs = [], [], []
        for bn in bnames:
            row = df_sum[(df_sum["bead_name"] == bn) & (df_sum["metric"] == metric) & (df_sum["state"] == st_val)].iloc[0]
            means.append(row["mean"])
            lo_errs.append(max(0.0, row["mean"] - row["ci_lo"]))
            hi_errs.append(max(0.0, row["ci_hi"] - row["mean"]))
        hatch = "//" if metric == "cos_psi" else None
        ax.bar(x_pos + offset, means, width=w4, color=col, alpha=alp, hatch=hatch, edgecolor="#222222", lw=0.6, yerr=[lo_errs, hi_errs], capsize=3, label=lbl)
    ax.axhline(0.0, color="#666666", ls=":", lw=1.0)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(-0.05, 1.15)
    ax.set_ylabel(r"$\cos\psi$ and $\eta$", fontsize=11)
    ax.set_title("(D) Alignment & Coupling Efficiency by State", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8.0, ncol=2, framealpha=0.9)
    ax.grid(True, axis="y", ls="--", alpha=0.35)

    fig.suptitle(r"Cargo-Masked Footprint Polar Order & Local Flow Coupling (Run vs Tumble)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_fig_to_dirs(fig, out_dirs, "gmm_polar_footprint_run_vs_tumble_k2")


def plot_radial_polar_curves(
    df_rad_sum: pd.DataFrame,
    beads_info: List[dict],
    out_dirs: List[Path],
    max_window_um: float = 30.0,
):
    """貨物マスク付き窓半径依存ポーラーオーダー Phi_masked(R) の Run vs Tumble 比較（2x2）。"""
    if df_rad_sum.empty:
        return

    bnames = [b["name"] for b in beads_info if b["name"] in df_rad_sum["bead_name"].unique()]
    n_b = len(bnames)
    fig, axes = plt.subplots(2, n_b, figsize=(6.4 * n_b, 8.2), sharex="col")
    if n_b == 1:
        axes = np.expand_dims(axes, axis=1)

    mode_styles = [
        ("Run", gc.STATE_COLORS[1], "-", "o", 2.2),
        ("Tumble", gc.STATE_COLORS[0], "-", "s", 2.2),
        ("All", "#222222", "--", None, 1.6),
        ("Background", "#7570b3", ":", None, 1.8),
    ]

    for col_idx, bn in enumerate(bnames):
        dia = next(b["diameter_um"] for b in beads_info if b["name"] == bn)
        sub_b = df_rad_sum[(df_rad_sum["bead_name"] == bn) & (df_rad_sum["window_um"] <= max_window_um)]

        ax_top = axes[0, col_idx]
        for mode, col, ls, mk, lw in mode_styles:
            sm = sub_b[sub_b["mode"] == mode].sort_values("window_um")
            if sm.empty:
                continue
            ax_top.plot(sm["window_um"], sm["mean"], color=col, ls=ls, marker=mk, ms=4, lw=lw, label=mode)
            if mode in ("Run", "Tumble"):
                ax_top.fill_between(sm["window_um"], sm["ci_lo"], sm["ci_hi"], color=col, alpha=0.18)

        ax_top.set_xscale("log")
        ax_top.set_ylim(0.15, 1.02)
        ax_top.set_title(rf"$d = {dia:.2f}\,\mu\mathrm{{m}}$ (Cargo-Masked Radial Window)", fontsize=11.5, fontweight="bold")
        ax_top.grid(True, which="both", ls="--", alpha=0.35)
        ax_top.legend(loc="lower left", fontsize=8.8, framealpha=0.9)

        ax_bot = axes[1, col_idx]
        sd = sub_b[sub_b["mode"] == "Delta_Run_minus_Tumble"].sort_values("window_um")
        if not sd.empty:
            ax_bot.plot(sd["window_um"], sd["mean"], "o-", color="#332288", ms=4.5, lw=2.0, label=r"Paired $\Delta\Phi = \Phi_{\mathrm{Run}} - \Phi_{\mathrm{Tumble}}$")
            ax_bot.fill_between(sd["window_um"], sd["ci_lo"], sd["ci_hi"], color="#332288", alpha=0.20, label="95% Bootstrap CI")
        ax_bot.axhline(0.0, color="#555555", ls="--", lw=1.2)
        ax_bot.set_xscale("log")
        ax_bot.set_xlabel(r"Window Radius $R$ [$\mu\mathrm{m}$]", fontsize=10.5)
        ax_bot.grid(True, which="both", ls="--", alpha=0.35)
        ax_bot.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    axes[0, 0].set_ylabel(r"Cargo-Masked Polar Order $\Phi_{\mathrm{masked}}(R)$", fontsize=10.5)
    axes[1, 0].set_ylabel(r"$\Delta\Phi(R) = \Phi_{\mathrm{Run}}(R) - \Phi_{\mathrm{Tumble}}(R)$", fontsize=10.5)
    fig.suptitle("Radial Cargo-Masked Local Polar Order: Run vs Tumble", fontsize=13.5, fontweight="bold")
    fig.tight_layout()
    save_fig_to_dirs(fig, out_dirs, "gmm_polar_radial_run_vs_tumble_k2")


def plot_polar_vs_run_fraction(
    df_binned: pd.DataFrame,
    beads_info: List[dict],
    out_dirs: List[Path],
):
    """1 - Phi_fp に対する Run 占有率と規格化速度のプロット（1x2）。"""
    if df_binned.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.0))
    for binfo in beads_info:
        bn = binfo["name"]
        sub = df_binned[df_binned["bead_name"] == bn].sort_values("mean_one_minus_phi")
        if sub.empty:
            continue
        lbl = rf"${binfo['diameter_um']:.2f}\,\mu\mathrm{{m}}$"
        axes[0].errorbar(
            sub["mean_one_minus_phi"],
            sub["p_run"],
            yerr=[sub["p_run"] - sub["p_run_ci_lo"], sub["p_run_ci_hi"] - sub["p_run"]],
            fmt=binfo["marker"] + "-",
            color=binfo["color"],
            ms=6.5,
            capsize=3,
            lw=1.8,
            label=lbl,
        )
        axes[1].plot(
            sub["mean_one_minus_phi"],
            sub["median_v_norm"],
            binfo["marker"] + "-",
            color=binfo["color"],
            ms=6.5,
            lw=1.8,
            label=lbl,
        )

    axes[0].set_xscale("log")
    axes[0].set_ylim(0, 0.8)
    axes[0].set_xlabel(r"$1 - \Phi_{\mathrm{fp}}$ (left: aligned, right: disordered)")
    axes[0].set_ylabel(r"$P(\mathrm{Run} \mid \Phi_{\mathrm{fp}})$")
    axes[0].set_title(r"(A) Run Probability vs $1 - \Phi_{\mathrm{fp}}$", fontweight="bold")
    axes[0].grid(True, which="both", ls="--", alpha=0.35)
    axes[0].legend(fontsize=9, framealpha=0.9)

    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"$1 - \Phi_{\mathrm{fp}}$ (left: aligned, right: disordered)")
    axes[1].set_ylabel(r"Median $v(t)\,/\,u_\infty(t)$")
    axes[1].set_title(r"(B) Normalized Speed vs $1 - \Phi_{\mathrm{fp}}$", fontweight="bold")
    axes[1].grid(True, which="both", ls="--", alpha=0.35)
    axes[1].legend(fontsize=9, framealpha=0.9)

    fig.suptitle(r"Dependence of Run State & Speed on Cargo-Masked Polar Order $\Phi_{\mathrm{fp}}$", fontsize=13.5, fontweight="bold")
    fig.tight_layout()
    save_fig_to_dirs(fig, out_dirs, "gmm_polar_vs_run_fraction_k2")


def main():
    parser = argparse.ArgumentParser(description="Cargo-masked local polar order in GMM Run vs Tumble states")
    parser.add_argument("--root_dir", type=str, default=None)
    parser.add_argument("--uinf_cache", type=str, default=str(CURRENT_DIR / "scratch" / "u_inf_per_frame_cache.pkl"))
    parser.add_argument("--fp_cache", type=str, default=str(CURRENT_DIR / "scratch" / "footprint_heterogeneity_cache.pkl"))
    parser.add_argument("--recompute_footprint", action="store_true", help="Recompute footprint annulus polar order directly from GFP_flows.h5")
    parser.add_argument("--beads", nargs="+", default=["beads06um", "beads1um"])
    parser.add_argument("--n_boot", type=int, default=2000)
    args = parser.parse_args()

    root_dir = Path(args.root_dir) if args.root_dir else find_default_root()
    out_dirs = [CURRENT_DIR / "figure" / "gmm_1d"]
    if root_dir != CURRENT_DIR and root_dir.exists():
        out_dirs.append(root_dir / "figure" / "gmm_1d")

    df_uinf = pd.read_pickle(args.uinf_cache)
    strict_particles = load_strict_active_particles()
    target_beads = [b for b in SMALL_BEADS_INFO if b["name"] in args.beads]

    merged_by_bead: Dict[str, pd.DataFrame] = {}
    fp_sum_list, cargo_list, rad_sum_list = [], [], []

    print("=================================================================")
    print("  Cargo-Masked Polar Order during GMM Run vs Tumble States       ")
    print("=================================================================")

    for binfo in target_beads:
        bname = binfo["name"]
        dia = binfo["diameter_um"]
        edirs = find_experiment_dirs(root_dir, bname)
        X, _, df_obs = collect_bead_gmm_data(edirs, bname, df_uinf, strict_particles)
        if len(X) < 20:
            continue

        gmm = gc.CargoGaussianGMM(n_components=2, random_state=42).fit(X)
        df_obs["pred_state"] = gmm.predict(X)
        df_obs["state_label"] = [gc.get_state_label(int(s), 2) for s in df_obs["pred_state"]]

        df_merged = gc.load_or_compute_masked_annulus_data(
            df_obs,
            bead_name=bname,
            exp_dirs=edirs,
            cache_path=Path(args.fp_cache),
            recompute=args.recompute_footprint,
        )
        merged_by_bead[bname] = df_merged

        df_s, df_c = summarize_footprint_by_state(df_merged, bname, dia, n_boot=args.n_boot)
        fp_sum_list.append(df_s)
        cargo_list.append(df_c)

        df_rad_raw, df_bg_raw = gc.extract_masked_radial_polar_order(edirs, df_obs)
        df_rad_s = summarize_radial_curves(df_rad_raw, df_bg_raw, bname, dia, n_boot=min(args.n_boot, 1000))
        if not df_rad_s.empty:
            rad_sum_list.append(df_rad_s)

        print(f"--- {bname} (d = {dia:.2f} um, N = {len(df_merged)}) ---")
        for col in ["phi_t", "f_opp", "u_fp_norm", "eta", "cos_psi"]:
            r_t = df_s[(df_s["metric"] == col) & (df_s["state"] == 0)].iloc[0]
            r_r = df_s[(df_s["metric"] == col) & (df_s["state"] == 1)].iloc[0]
            print(f"  {col:10s}: Tumble={r_t['mean']:.4f} [{r_t['ci_lo']:.4f}, {r_t['ci_hi']:.4f}] | "
                  f"Run={r_r['mean']:.4f} [{r_r['ci_lo']:.4f}, {r_r['ci_hi']:.4f}]")
        valid_d = df_c["delta_phi_run_minus_tumble"].dropna()
        _, pval = stats.wilcoxon(valid_d) if len(valid_d) >= 2 else (np.nan, np.nan)
        print(f"  Paired dPhi (Run - Tumble): mean={valid_d.mean():+.4f}, median={valid_d.median():+.4f}, "
              f"pos={(valid_d > 0).sum()}/{len(valid_d)}, Wilcoxon p={pval:.4g}\n")

    df_fp_sum = pd.concat(fp_sum_list, ignore_index=True) if fp_sum_list else pd.DataFrame()
    df_cargo_all = pd.concat(cargo_list, ignore_index=True) if cargo_list else pd.DataFrame()
    df_rad_all = pd.concat(rad_sum_list, ignore_index=True) if rad_sum_list else pd.DataFrame()
    df_binned = compute_binned_polar_vs_run(merged_by_bead, target_beads, n_bins=8, n_boot=min(args.n_boot, 1000))

    plot_footprint_run_vs_tumble(merged_by_bead, df_fp_sum, df_cargo_all, target_beads, out_dirs)
    plot_radial_polar_curves(df_rad_all, target_beads, out_dirs, max_window_um=30.0)
    plot_polar_vs_run_fraction(df_binned, target_beads, out_dirs)

    save_csv_to_dirs(df_fp_sum, out_dirs, "gmm_polar_footprint_summary_k2.csv")
    save_csv_to_dirs(df_cargo_all, out_dirs, "gmm_polar_footprint_per_cargo_k2.csv")
    save_csv_to_dirs(df_rad_all, out_dirs, "gmm_polar_radial_curves_k2.csv")
    save_csv_to_dirs(df_binned, out_dirs, "gmm_polar_vs_run_binned_k2.csv")

    print("=== GMM Cargo-Masked Polar Order Analysis Completed ===")


if __name__ == "__main__":
    main()
