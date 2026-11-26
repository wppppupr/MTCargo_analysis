#!/usr/bin/env python3
"""
plot_c_potential_du_plus1_and_k1.py

c = v_hat . u_hat の有効ポテンシャル U_tilde_eff(c) および角度分布 P(theta) から抽出した:
  1. 過剰レール捕捉深さ Delta U_tilde_+1 vs R_c / xi
  2. 1次モーメント一致 (I_1(kappa_1)/I_0(kappa_1) = m_1) による大域整列パラメータ kappa_1 vs R_c / xi
  3. 1次モーメント <cos theta> = m_1 vs R_c / xi
の個別グラフを作成する。
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import verify_cos_decomposition as vcd
from calc_and_plot_kl_decomposition import solve_kappa1_from_m1

warnings.filterwarnings("ignore")

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = vcd.ROOT_DIR
OUT_DIRS = [
    CURRENT_DIR / "figure" / "per_point_v_dot_u",
    ROOT_DIR / "figure" / "per_point_v_dot_u",
]
HIST_CACHE = CURRENT_DIR / "scratch" / "per_point_v_dot_u_hist_cache.pkl"


def plot_single_metric(
    df: pd.DataFrame,
    y_col: str,
    lo_col: str,
    hi_col: str,
    ylabel: str,
    title: str,
    out_name: str,
    ylim: tuple,
    offsets: dict,
):
    fig, ax = plt.subplots(figsize=(7.6, 5.6))

    X_LOCK = 45.0 * np.pi / 512.0  # ~0.28 (Higher-order collapse)
    X_BULK = 45.0 * np.pi / 128.0  # ~1.10 (Bulk alignment decay)

    ax.axhline(0.0, color="#888888", linestyle="-", linewidth=1.0, zorder=1)
    ax.axvline(X_LOCK, color="#d95f02", linestyle="--", linewidth=1.6, alpha=0.85, zorder=1,
               label=r"$x_{\mathrm{lock}} = \frac{45\pi}{512} \approx 0.28$ (Higher-order collapse)")
    ax.axvline(X_BULK, color="#222222", linestyle=":", linewidth=1.7, alpha=0.85, zorder=1,
               label=r"$x_{\mathrm{bulk}} = \frac{45\pi}{128} \approx 1.10$ (Bulk alignment decay)")

    # 接続線
    ax.plot(df["rc_over_xi"], df[y_col], color="#444444", linestyle="-", linewidth=1.8, alpha=0.65, zorder=2)

    for b_item in vcd.BEADS_INFO:
        bname = b_item["name"]
        sub = df[df["bead_name"] == bname]
        if sub.empty:
            continue
        r = sub.iloc[0]
        y_val = float(r[y_col])
        yerr_lo = max(0.0, y_val - float(r[lo_col]))
        yerr_hi = max(0.0, float(r[hi_col]) - y_val)
        x_val = float(r["rc_over_xi"])
        x_err = float(r["rc_over_xi_sem"]) if np.isfinite(r["rc_over_xi_sem"]) else 0.0

        ax.errorbar(
            x_val, y_val,
            xerr=x_err,
            yerr=[[yerr_lo], [yerr_hi]],
            fmt=b_item["marker"], color=b_item["color"],
            markersize=9.5, markeredgecolor="#222222", markeredgewidth=0.9,
            ecolor=b_item["color"], elinewidth=1.8, capsize=4.0, alpha=0.92,
            zorder=4,
        )

        d_um = float(r["diameter_um"])
        dx, dy, ha = offsets.get(bname, (8, 8, "left"))
        ax.annotate(
            rf"{d_um:.2f} $\mu$m" if d_um < 10 else rf"{d_um:.1f} $\mu$m",
            xy=(x_val, y_val),
            xytext=(dx, dy), textcoords="offset points",
            ha=ha, va="center",
            fontsize=10.5, fontweight="bold", color=b_item["color"],
            zorder=5,
        )

    ax.set_xscale("log")
    ax.set_xlim(0.045, 3.2)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"Dimensionless Cargo Radius $R_c / \xi$", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=13.0, fontweight="bold", pad=10)
    ax.grid(True, which="both", linestyle=":", alpha=0.55)
    ax.legend(loc="upper right", fontsize=9.3, framealpha=0.92)
    fig.tight_layout()

    vcd.save_figure(fig, out_name, OUT_DIRS)
    plt.close(fig)


def compute_moment_matched_k1_and_m1(n_boot: int = 1000, seed: int = 42) -> pd.DataFrame:
    """P(theta) の1次モーメント m_1 = <cos theta> と I_1(kappa_1)/I_0(kappa_1) = m_1 を満たす kappa_1 (+ 95% CI) を算出"""
    with open(HIST_CACHE, "rb") as f:
        df_rec = pd.DataFrame(pickle.load(f))

    rng = np.random.default_rng(seed)
    rows = []
    for b_item in vcd.BEADS_INFO:
        bname = b_item["name"]
        sub = df_rec[df_rec["bead_name"] == bname]
        if sub.empty:
            continue

        n_p = len(sub)
        rc_xi = float(sub["rc_over_xi"].mean())
        rc_xi_sem = float(sub["rc_over_xi"].sem()) if n_p > 1 else 0.0

        sumc_arr = sub["sum_c"].values.astype(float)
        npts_arr = sub["n_pts"].values.astype(float)

        m1_obs = float(sumc_arr.sum() / npts_arr.sum())
        k1_obs = solve_kappa1_from_m1(m1_obs)

        boot_m1, boot_k1 = [], []
        for _ in range(n_boot):
            idx = rng.integers(0, n_p, size=n_p)
            tot_n = float(npts_arr[idx].sum())
            if tot_n <= 0:
                continue
            m1_b = float(sumc_arr[idx].sum() / tot_n)
            k1_b = solve_kappa1_from_m1(m1_b)
            boot_m1.append(m1_b)
            boot_k1.append(k1_b)

        rows.append({
            "bead_name": bname,
            "diameter_um": float(b_item["diameter_um"]),
            "rc_over_xi": rc_xi,
            "rc_over_xi_sem": rc_xi_sem,
            "n_particles": n_p,
            "m1_cos": m1_obs,
            "m1_cos_ci_low": float(np.percentile(boot_m1, 2.5)),
            "m1_cos_ci_high": float(np.percentile(boot_m1, 97.5)),
            "kappa_1_vm": k1_obs,
            "kappa_1_vm_ci_low": float(np.percentile(boot_k1, 2.5)),
            "kappa_1_vm_ci_high": float(np.percentile(boot_k1, 97.5)),
        })

    return pd.DataFrame(rows).sort_values("rc_over_xi").reset_index(drop=True)


def main():
    vcd.setup_matplotlib()
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Delta U_+1
    csv_path = CURRENT_DIR / "figure" / "per_point_v_dot_u" / "k1_and_excess_pinning_summary.csv"
    df_du = pd.read_csv(csv_path).sort_values("rc_over_xi")

    offsets_du = {
        "beads06um": (10, 4, "left"),
        "beads1um":  (10, 4, "left"),
        "beads3um":  (-10, -8, "right"),
        "beads7um":  (10, -10, "left"),
        "beads5um":  (-10, 10, "right"),
        "beads20um": (0, -24, "center"),
    }
    plot_single_metric(
        df_du,
        y_col="delta_U_plus1",
        lo_col="delta_U_plus1_ci_low",
        hi_col="delta_U_plus1_ci_high",
        ylabel=r"Excess Rail Pinning Depth $\Delta\tilde{U}_{+1}$",
        title=r"Excess Rail Pinning $\Delta\tilde{U}_{+1}$ vs $R_c / \xi$ (Error bars: 95% Bootstrap CI)",
        out_name="delta_U_plus1_vs_rc_xi",
        ylim=(-0.15, 1.45),
        offsets=offsets_du,
    )

    # 2. 1次モーメント一致による kappa_1 と <cos theta>
    df_vm = compute_moment_matched_k1_and_m1(n_boot=1000, seed=42)
    for d in OUT_DIRS:
        df_vm.to_csv(d / "m1_and_kappa1_vm_summary.csv", index=False)

    print("\n=== 1st-Moment Matched von Mises kappa_1 & <cos theta> ===")
    print(df_vm[["bead_name", "diameter_um", "rc_over_xi", "m1_cos", "m1_cos_ci_low", "m1_cos_ci_high",
                 "kappa_1_vm", "kappa_1_vm_ci_low", "kappa_1_vm_ci_high"]].to_string(index=False))

    offsets_k1 = {
        "beads06um": (10, 6, "left"),
        "beads1um":  (-10, -8, "right"),
        "beads3um":  (10, 8, "left"),
        "beads7um":  (10, 6, "left"),
        "beads5um":  (-10, -10, "right"),
        "beads20um": (0, 18, "center"),
    }
    plot_single_metric(
        df_vm,
        y_col="kappa_1_vm",
        lo_col="kappa_1_vm_ci_low",
        hi_col="kappa_1_vm_ci_high",
        ylabel=r"Moment-Matched Coupling $\kappa_1$ ($I_1/I_0 = \langle\cos\theta\rangle$)",
        title=r"Moment-Matched $\kappa_1$ vs $R_c / \xi$ (Error bars: 95% Bootstrap CI)",
        out_name="kappa1_vs_rc_xi",
        ylim=(-0.05, 1.05),
        offsets=offsets_k1,
    )

    offsets_m1 = {
        "beads06um": (10, 6, "left"),
        "beads1um":  (-10, -8, "right"),
        "beads3um":  (10, 8, "left"),
        "beads7um":  (10, 6, "left"),
        "beads5um":  (-10, -10, "right"),
        "beads20um": (0, 18, "center"),
    }
    plot_single_metric(
        df_vm,
        y_col="m1_cos",
        lo_col="m1_cos_ci_low",
        hi_col="m1_cos_ci_high",
        ylabel=r"Mean Alignment $\langle \cos\theta \rangle = \int P(\theta)\cos\theta\,d\theta$",
        title=r"Mean Alignment $\langle \cos\theta \rangle$ vs $R_c / \xi$ (Error bars: 95% Bootstrap CI)",
        out_name="cos_theta_mean_vs_rc_xi",
        ylim=(-0.20, 0.48),
        offsets=offsets_m1,
    )
    print("\nSaved delta_U_plus1_vs_rc_xi, kappa1_vs_rc_xi, and cos_theta_mean_vs_rc_xi (.png and .svg)")


if __name__ == "__main__":
    main()
