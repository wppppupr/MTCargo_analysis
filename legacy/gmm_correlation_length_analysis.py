#!/usr/bin/env python3
"""
gmm_correlation_length_analysis.py

小貨物（0.63 µm, 1.18 µm）について、ln(v(t) / u_inf(t)) の 1D GMM で分類した
Run（高速状態）と Tumble（低速状態）それぞれの空間配向相関 C(r) から相関長 xi を求め、
Run と Tumble それぞれの実効的な R_c / xi を算出・比較・可視化するスクリプトです。

算出方法:
  1. 主解析 (angular_correlation_w.zarr & angular_correlation_bg.zarr):
     他のコード（plot_bg_angular_correlation.py / libs.hmm_flow_correlation.py）と同一の
     重み付き対数線形フィット (ln C = ln a - r / xi, cutoff_ratio = 1/e, max_fit_dist = 20 um)
     により xi_Tumble, xi_Run, xi_All, xi_BG および実効的 R_c / xi を算出し、
     実験動画 -> 貨物粒子の階層ブートストラップで 95% CI を推定する。
  2. クロスチェック (貨物マスク付き足印基準 C_fp(r), GFP_flows.h5):
     貨物中心ピクセルを完全に使わず、貨物マスク外側の足印アニュラス方向 u_hat_fp を基準に
     空間配向相関 C_fp(r) を算出して xi と R_c / xi の頑健性を確認する。

出力先:
  - figure/gmm_1d/
  - /mnt/NAS-Ebanaru/sasaki/MTsingleBeads/figure/gmm_1d/
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from libs import gmm_cargo as gc
from libs import hmm_flow_correlation as hfc
from libs.experiment_cutoff import load_strict_active_particles
from gmm_cargo_analysis import (
    SMALL_BEADS_INFO,
    collect_bead_gmm_data,
    find_default_root,
    find_experiment_dirs,
    save_csv_to_dirs,
    save_fig_to_dirs,
)


def plot_spatial_corr_and_fits(
    zarr_curves: Dict[str, dict],
    fp_curves: Dict[str, dict],
    beads_info: List[dict],
    out_dirs: List[Path],
):
    """Run vs Tumble の空間配向相関 C(r) と指数フィット曲線の描画（2x2）。"""
    bnames = [b["name"] for b in beads_info if b["name"] in zarr_curves]
    n_b = len(bnames)
    fig, axes = plt.subplots(2, n_b, figsize=(6.5 * n_b, 8.4), sharex="col")
    if n_b == 1:
        axes = np.expand_dims(axes, axis=1)

    styles = [
        ("Run", gc.STATE_COLORS[1], "-", "o"),
        ("Tumble", gc.STATE_COLORS[0], "-", "s"),
        ("All", "#222222", "--", "^"),
        ("Background", "#7570b3", ":", "d"),
    ]

    for c_idx, bn in enumerate(bnames):
        dia = next(b["diameter_um"] for b in beads_info if b["name"] == bn)
        rc_um = next(b["radius_um"] for b in beads_info if b["name"] == bn)

        # Top row: angular_correlation_w.zarr
        ax_top = axes[0, c_idx]
        z_data = zarr_curves[bn]
        for mode, col, ls, mk in styles:
            if mode not in z_data:
                continue
            dfc, summ = z_data[mode]
            if dfc.empty:
                continue
            sub = dfc[dfc["distance_um"] <= 20.0]
            xi_val = summ.get("xi_um", np.nan)
            rc_xi = summ.get("rc_over_xi", np.nan)
            lbl = rf"{mode}: $\xi={xi_val:.2f}\,\mu\mathrm{{m}}$ ($R_c/\xi={rc_xi:.3f}$)"
            ax_top.errorbar(
                sub["distance_um"],
                sub["mean_correlation"],
                yerr=sub["sem_correlation"],
                fmt=mk,
                ms=3.8,
                color=col,
                ls="none",
                alpha=0.8,
                capsize=2,
                label=lbl,
            )
            fit_r = summ.get("fit_r", np.array([]))
            fit_c = summ.get("fit_c", np.array([]))
            if len(fit_r) > 0:
                m_f = (fit_r <= 20.0) & np.isfinite(fit_c) & (fit_c > 0.01)
                ax_top.plot(fit_r[m_f], fit_c[m_f], color=col, ls=ls, lw=2.0)

        ax_top.set_yscale("log")
        ax_top.set_ylim(0.03, 1.15)
        ax_top.set_xlim(0.0, 20.5)
        ax_top.set_title(rf"$d={dia:.2f}\,\mu\mathrm{{m}}$ ($R_c={rc_um:.3f}\,\mu\mathrm{{m}}$): `angular_correlation_w`", fontsize=11, fontweight="bold")
        ax_top.grid(True, which="both", ls="--", alpha=0.35)
        ax_top.legend(loc="lower left", fontsize=8.0, framealpha=0.9)

        # Bottom row: pure cargo-masked C_fp(r) cross-check
        ax_bot = axes[1, c_idx]
        f_data = fp_curves.get(bn, {})
        for mode, col, ls, mk in [("Run", gc.STATE_COLORS[1], "-", "o"), ("Tumble", gc.STATE_COLORS[0], "-", "s")]:
            if mode not in f_data:
                continue
            dfc, summ = f_data[mode]
            if dfc.empty:
                continue
            sub = dfc[dfc["distance_um"] <= 20.0]
            xi_val = summ.get("xi_um", np.nan)
            rc_xi = summ.get("rc_over_xi", np.nan)
            lbl = rf"{mode}: $\xi={xi_val:.2f}\,\mu\mathrm{{m}}$ ($R_c/\xi={rc_xi:.3f}$)"
            ax_bot.errorbar(
                sub["distance_um"],
                sub["mean_correlation"],
                yerr=sub["sem_correlation"],
                fmt=mk,
                ms=3.8,
                color=col,
                ls="none",
                alpha=0.8,
                capsize=2,
                label=lbl,
            )
            fit_r = summ.get("fit_r", np.array([]))
            fit_c = summ.get("fit_c", np.array([]))
            if len(fit_r) > 0:
                m_f = (fit_r <= 20.0) & np.isfinite(fit_c) & (fit_c > 0.01)
                ax_bot.plot(fit_r[m_f], fit_c[m_f], color=col, ls=ls, lw=2.0)

        ax_bot.set_yscale("log")
        ax_bot.set_ylim(0.03, 1.15)
        ax_bot.set_xlim(0.0, 20.5)
        ax_bot.set_xlabel(r"Distance $r$ from Cargo Center [$\mu\mathrm{m}$]", fontsize=10.5)
        ax_bot.set_title(rf"$d={dia:.2f}\,\mu\mathrm{{m}}$: Cargo-Masked Footprint $C_{{\mathrm{{fp}}}}(r)$", fontsize=11, fontweight="bold")
        ax_bot.grid(True, which="both", ls="--", alpha=0.35)
        if f_data:
            ax_bot.legend(loc="lower left", fontsize=8.3, framealpha=0.9)

    axes[0, 0].set_ylabel(r"Spatial Orientation Correlation $C(r)$", fontsize=10.5)
    axes[1, 0].set_ylabel(r"Masked Footprint Correlation $C_{\mathrm{fp}}(r)$", fontsize=10.5)
    fig.suptitle(r"Spatial Orientation Correlation Decay & Correlation Length $\xi$ (Run vs Tumble)", fontsize=13.5, fontweight="bold")
    fig.tight_layout()
    save_fig_to_dirs(fig, out_dirs, "gmm_spatial_corr_xi_run_vs_tumble_k2")


def plot_effective_rc_over_xi(
    df_sum: pd.DataFrame,
    df_per_exp: pd.DataFrame,
    beads_info: List[dict],
    out_dirs: List[Path],
):
    """Run と Tumble それぞれの xi と実効的な R_c / xi の比較図（1x2）。"""
    if df_sum.empty:
        return

    bnames = [b["name"] for b in beads_info if b["name"] in df_sum["bead_name"].unique()]
    x_pos = np.arange(len(bnames))
    x_labels = [rf"${next(b['diameter_um'] for b in beads_info if b['name'] == bn):.2f}\,\mu\mathrm{{m}}$" for bn in bnames]

    modes = [
        ("Tumble", -0.27, gc.STATE_COLORS[0]),
        ("Run", -0.09, gc.STATE_COLORS[1]),
        ("All", +0.09, "#444444"),
        ("Background", +0.27, "#7570b3"),
    ]
    w = 0.17

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    z_sub = df_sum[df_sum["source"] == "angular_correlation_w"]
    fp_sub = df_sum[df_sum["source"] == "masked_footprint_C_fp"]

    for ax, col_val, col_lo, col_hi, ylabel, title in [
        (axes[0], "xi_um", "xi_ci_lo", "xi_ci_hi", r"Correlation Length $\xi$ [$\mu\mathrm{m}$]", r"(A) Correlation Length $\xi$ (95% Hierarchical CI)"),
        (axes[1], "rc_over_xi", "rc_xi_ci_lo", "rc_xi_ci_hi", r"Effective Scaled Radius $R_c / \xi$", r"(B) Effective $R_c / \xi$ (95% Hierarchical CI)"),
    ]:
        for mode, offset, col in modes:
            vals, lo_e, hi_e = [], [], []
            for bn in bnames:
                row = z_sub[(z_sub["bead_name"] == bn) & (z_sub["mode"] == mode)]
                if row.empty:
                    vals.append(np.nan); lo_e.append(0.0); hi_e.append(0.0)
                    continue
                r0 = row.iloc[0]
                v = float(r0[col_val])
                vals.append(v)
                lo_e.append(max(0.0, v - float(r0[col_lo])))
                hi_e.append(max(0.0, float(r0[col_hi]) - v))
            ax.bar(x_pos + offset, vals, width=w, color=col, alpha=0.85, yerr=[lo_e, hi_e], capsize=3.5, label=mode, zorder=3)

            # Overlay masked C_fp(r) cross-check diamond for Run and Tumble
            if mode in ("Tumble", "Run"):
                fp_vals = []
                for bn in bnames:
                    rf = fp_sub[(fp_sub["bead_name"] == bn) & (fp_sub["mode"] == mode)]
                    fp_vals.append(float(rf[col_val].iloc[0]) if not rf.empty else np.nan)
                ax.scatter(
                    x_pos + offset,
                    fp_vals,
                    marker="D",
                    s=42,
                    color="white",
                    edgecolor="black",
                    lw=1.3,
                    zorder=5,
                    label=r"Masked $C_{\mathrm{fp}}(r)$ check" if mode == "Tumble" else None,
                )

            # Per-experiment scatter for Run / Tumble
            if mode in ("Tumble", "Run") and not df_per_exp.empty:
                for xi_i, bn in enumerate(bnames):
                    pe = df_per_exp[(df_per_exp["bead_name"] == bn) & (df_per_exp["mode"] == mode)]
                    ax.scatter(
                        np.full(len(pe), xi_i + offset),
                        pe[col_val],
                        s=16,
                        color="#111111",
                        alpha=0.45,
                        zorder=4,
                    )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, axis="y", ls="--", alpha=0.35)
        ax.legend(loc="upper right" if col_val == "xi_um" else "upper left", fontsize=8.3, framealpha=0.92)

    fig.suptitle(r"Effective Correlation Length $\xi$ and Dimensionless Size $R_c / \xi$ in Run vs Tumble", fontsize=13.5, fontweight="bold")
    fig.tight_layout()
    save_fig_to_dirs(fig, out_dirs, "gmm_effective_rc_over_xi_run_vs_tumble_k2")


def main():
    parser = argparse.ArgumentParser(description="Compute spatial correlation length xi and effective R_c/xi for GMM Run vs Tumble")
    parser.add_argument("--root_dir", type=str, default=None)
    parser.add_argument("--uinf_cache", type=str, default=str(CURRENT_DIR / "scratch" / "u_inf_per_frame_cache.pkl"))
    parser.add_argument("--beads", nargs="+", default=["beads06um", "beads1um"])
    parser.add_argument("--max_fit_dist", type=float, default=20.0)
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--skip_fp", action="store_true", help="Skip slow GFP_flows.h5 C_fp(r) computation")
    args = parser.parse_args()

    root_dir = Path(args.root_dir) if args.root_dir else find_default_root()
    out_dirs = [CURRENT_DIR / "figure" / "gmm_1d"]
    if root_dir != CURRENT_DIR and root_dir.exists():
        out_dirs.append(root_dir / "figure" / "gmm_1d")

    df_uinf = pd.read_pickle(args.uinf_cache)
    strict_particles = load_strict_active_particles()
    target_beads = [b for b in SMALL_BEADS_INFO if b["name"] in args.beads]

    summary_rows = []
    per_exp_rows = []
    zarr_curve_rows = []
    fp_curve_rows = []
    zarr_curves_for_plot: Dict[str, dict] = {}
    fp_curves_for_plot: Dict[str, dict] = {}

    print("=================================================================")
    print("  Spatial Correlation Length xi & Effective R_c/xi (Run vs Tumble)")
    print("=================================================================")

    for binfo in target_beads:
        bname = binfo["name"]
        dia = binfo["diameter_um"]
        rc_um = binfo["radius_um"]
        edirs = find_experiment_dirs(root_dir, bname)

        X, _, df_obs = collect_bead_gmm_data(edirs, bname, df_uinf, strict_particles)
        if len(X) < 20:
            continue

        gmm = gc.CargoGaussianGMM(n_components=2, random_state=42).fit(X)
        df_obs["pred_state"] = gmm.predict(X)

        # 1. Primary: angular_correlation_w.zarr & angular_correlation_bg.zarr
        d_um, cargo_by_st, df_bg = gc.extract_spatial_correlation_curves(edirs, df_obs)
        zarr_curves_for_plot[bname] = {}

        print(f"\n--- {bname} (d = {dia:.2f} um, R_c = {rc_um:.3f} um) ---")
        for st, mode_lbl in [(0, "Tumble"), (1, "Run"), (-1, "All")]:
            dfc, summ = gc.fit_xi_with_hierarchical_bootstrap(
                cargo_by_st.get(st, {}),
                d_um=d_um,
                radius_um=rc_um,
                min_fit_dist=0.0,
                max_fit_dist=args.max_fit_dist,
                min_corr_threshold=0.01,
                cutoff_ratio=1.0 / np.e,
                n_boot=args.n_boot,
                seed=42,
            )
            zarr_curves_for_plot[bname][mode_lbl] = (dfc, summ)
            if not dfc.empty:
                tmp = dfc.copy()
                tmp["bead_name"] = bname
                tmp["diameter_um"] = dia
                tmp["mode"] = mode_lbl
                zarr_curve_rows.append(tmp)

            summary_rows.append({
                "source": "angular_correlation_w",
                "bead_name": bname,
                "diameter_um": dia,
                "radius_um": rc_um,
                "mode": mode_lbl,
                **{k: v for k, v in summ.items() if k not in ("fit_r", "fit_c")},
            })
            print(
                f"  [zarr] {mode_lbl:10s}: xi = {summ['xi_um']:.3f} [{summ['xi_ci_lo']:.3f}, {summ['xi_ci_hi']:.3f}] um "
                f"(R2_log={summ['r2_log']:.3f}) | R_c/xi = {summ['rc_over_xi']:.4f} [{summ['rc_xi_ci_lo']:.4f}, {summ['rc_xi_ci_hi']:.4f}]"
            )

            # Per-experiment xi
            for exp_name, items in sorted(cargo_by_st.get(st, {}).items()):
                num = np.zeros(len(d_um), dtype=float)
                den = np.zeros(len(d_um), dtype=float)
                sq = np.zeros(len(d_um), dtype=float)
                for cm, cnt in items:
                    v = np.isfinite(cm) & (cnt > 0)
                    num[v] += cm[v] * cnt[v]
                    sq[v] += (cm[v] ** 2) * cnt[v]
                    den[v] += cnt[v]
                ok = den >= 3
                mc = np.where(ok, num / np.maximum(den, 1.0), np.nan)
                vc = np.where(ok, np.maximum(0.0, sq / np.maximum(den, 1.0) - mc ** 2), 0.0)
                sc = np.where(ok, np.sqrt(vc / np.maximum(den, 1.0)), np.nan)
                df_pe = pd.DataFrame({"distance_um": d_um[ok], "mean_correlation": mc[ok], "sem_correlation": sc[ok]})
                res_pe = hfc.fit_flow_correlation_length(df_pe, min_fit_dist=0.0, max_fit_dist=args.max_fit_dist, min_corr_threshold=0.01, cutoff_ratio=1.0 / np.e)
                xi_pe = float(res_pe.get("xi_um", np.nan))
                per_exp_rows.append({
                    "bead_name": bname,
                    "diameter_um": dia,
                    "radius_um": rc_um,
                    "exp_dir": exp_name,
                    "mode": mode_lbl,
                    "n_cargos": len(items),
                    "xi_um": xi_pe,
                    "xi_err_um": float(res_pe.get("xi_err_um", np.nan)),
                    "rc_over_xi": float(rc_um / xi_pe) if (np.isfinite(xi_pe) and xi_pe > 0) else np.nan,
                    "r2_log": float(res_pe.get("r2_log", np.nan)),
                })

        if not df_bg.empty:
            res_bg = hfc.fit_flow_correlation_length(
                df_bg, min_fit_dist=0.0, max_fit_dist=args.max_fit_dist, min_corr_threshold=0.01, cutoff_ratio=1.0 / np.e
            )
            xi_bg = float(res_bg.get("xi_um", np.nan))
            xi_bg_err = float(res_bg.get("xi_err_um", np.nan))
            rc_bg = float(rc_um / xi_bg) if (np.isfinite(xi_bg) and xi_bg > 0) else np.nan
            rc_bg_lo = float(rc_um / (xi_bg + 1.96 * xi_bg_err)) if np.isfinite(xi_bg_err) else rc_bg
            rc_bg_hi = float(rc_um / max(xi_bg - 1.96 * xi_bg_err, 1e-6)) if np.isfinite(xi_bg_err) else rc_bg
            summ_bg = {
                "xi_um": xi_bg,
                "xi_fit_err_um": xi_bg_err,
                "xi_ci_lo": xi_bg - 1.96 * xi_bg_err if np.isfinite(xi_bg_err) else xi_bg,
                "xi_ci_hi": xi_bg + 1.96 * xi_bg_err if np.isfinite(xi_bg_err) else xi_bg,
                "rc_over_xi": rc_bg,
                "rc_xi_ci_lo": rc_bg_lo,
                "rc_xi_ci_hi": rc_bg_hi,
                "r2_log": float(res_bg.get("r2_log", np.nan)),
                "r2": float(res_bg.get("r2", np.nan)),
                "amplitude": float(res_bg.get("amplitude", np.nan)),
                "r_fit_min_um": float(res_bg.get("r_fit_min_um", np.nan)),
                "r_fit_max_um": float(res_bg.get("r_fit_max_um", np.nan)),
                "n_cargos": 0,
                "n_movies": len(edirs),
                "fit_r": res_bg.get("fit_r", np.array([])),
                "fit_c": res_bg.get("fit_c", np.array([])),
            }
            zarr_curves_for_plot[bname]["Background"] = (df_bg, summ_bg)
            tmp_bg = df_bg.copy()
            tmp_bg["bead_name"] = bname
            tmp_bg["diameter_um"] = dia
            tmp_bg["mode"] = "Background"
            zarr_curve_rows.append(tmp_bg)
            summary_rows.append({
                "source": "angular_correlation_w",
                "bead_name": bname,
                "diameter_um": dia,
                "radius_um": rc_um,
                "mode": "Background",
                **{k: v for k, v in summ_bg.items() if k not in ("fit_r", "fit_c")},
            })
            print(f"  [zarr] {'Background':10s}: xi = {xi_bg:.3f}±{xi_bg_err:.3f} um (R2_log={summ_bg['r2_log']:.3f}) | R_c/xi = {rc_bg:.4f}")

        # 2. Cross-check: pure cargo-masked footprint-referenced C_fp(r) from GFP_flows.h5
        fp_curves_for_plot[bname] = {}
        if not args.skip_fp:
            r_fp_um, fp_by_st = gc.extract_masked_footprint_spatial_correlation(edirs, df_obs, bead_name=bname, r_max_um=args.max_fit_dist + 0.5)
            for st, mode_lbl in [(0, "Tumble"), (1, "Run"), (-1, "All")]:
                dfc_fp, summ_fp = gc.fit_xi_with_hierarchical_bootstrap(
                    fp_by_st.get(st, {}),
                    d_um=r_fp_um,
                    radius_um=rc_um,
                    min_fit_dist=0.0,
                    max_fit_dist=args.max_fit_dist,
                    min_corr_threshold=0.01,
                    cutoff_ratio=1.0 / np.e,
                    n_boot=args.n_boot,
                    seed=42,
                )
                fp_curves_for_plot[bname][mode_lbl] = (dfc_fp, summ_fp)
                if not dfc_fp.empty:
                    tmp = dfc_fp.copy()
                    tmp["bead_name"] = bname
                    tmp["diameter_um"] = dia
                    tmp["mode"] = mode_lbl
                    fp_curve_rows.append(tmp)
                summary_rows.append({
                    "source": "masked_footprint_C_fp",
                    "bead_name": bname,
                    "diameter_um": dia,
                    "radius_um": rc_um,
                    "mode": mode_lbl,
                    **{k: v for k, v in summ_fp.items() if k not in ("fit_r", "fit_c")},
                })
                print(
                    f"  [C_fp] {mode_lbl:10s}: xi = {summ_fp['xi_um']:.3f} [{summ_fp['xi_ci_lo']:.3f}, {summ_fp['xi_ci_hi']:.3f}] um "
                    f"(R2_log={summ_fp['r2_log']:.3f}) | R_c/xi = {summ_fp['rc_over_xi']:.4f} [{summ_fp['rc_xi_ci_lo']:.4f}, {summ_fp['rc_xi_ci_hi']:.4f}]"
                )

    df_sum = pd.DataFrame(summary_rows)
    df_pe = pd.DataFrame(per_exp_rows)
    df_zc = pd.concat(zarr_curve_rows, ignore_index=True) if zarr_curve_rows else pd.DataFrame()
    df_fc = pd.concat(fp_curve_rows, ignore_index=True) if fp_curve_rows else pd.DataFrame()

    plot_spatial_corr_and_fits(zarr_curves_for_plot, fp_curves_for_plot, target_beads, out_dirs)
    plot_effective_rc_over_xi(df_sum, df_pe, target_beads, out_dirs)

    save_csv_to_dirs(df_sum, out_dirs, "gmm_xi_and_rc_over_xi_summary_k2.csv")
    save_csv_to_dirs(df_pe, out_dirs, "gmm_xi_and_rc_over_xi_per_experiment_k2.csv")
    save_csv_to_dirs(df_zc, out_dirs, "gmm_spatial_corr_curves_k2.csv")
    save_csv_to_dirs(df_fc, out_dirs, "gmm_masked_footprint_corr_curves_k2.csv")

    print("\n=== GMM Correlation Length & Effective R_c/xi Analysis Completed ===")


if __name__ == "__main__":
    main()
