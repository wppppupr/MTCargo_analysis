#!/usr/bin/env python3
"""
plot_per_point_delta_u_vs_rc_xi.py

scratch/per_point_v_dot_u_hist_cache.pkl から各ビーズサイズの
井戸の深さ Delta U_tilde_eff = U_tilde_eff(pi) - U_tilde_eff(0)
と粒子単位クラスターブートストラップ (B=1000) による 95% CI を算出し、
横軸 R_c / xi に対してプロットする。
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import verify_cos_decomposition as vcd

warnings.filterwarnings("ignore")

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = vcd.ROOT_DIR
OUT_DIRS = [
    CURRENT_DIR / "figure" / "per_point_v_dot_u",
    ROOT_DIR / "figure" / "per_point_v_dot_u",
]
HIST_CACHE = CURRENT_DIR / "scratch" / "per_point_v_dot_u_hist_cache.pkl"

BINS_TH = np.linspace(-np.pi, np.pi, 37)


def exp_decay(x, u0, lam):
    return u0 * np.exp(-x / lam)


def main():
    vcd.setup_matplotlib()
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)

    with open(HIST_CACHE, "rb") as f:
        records = pickle.load(f)
    df_rec = pd.DataFrame(records)

    th_centers = 0.5 * (BINS_TH[:-1] + BINS_TH[1:])
    dth = BINS_TH[1] - BINS_TH[0]
    idx_zero = int(np.argmin(np.abs(th_centers)))
    # pi と -pi は周期境界で同一なので両端2ビンの平均を U(pi) とする
    pi_indices = [0, len(th_centers) - 1]

    B_BOOT = 1000
    rng = np.random.default_rng(42)
    rows = []

    for b_item in vcd.BEADS_INFO:
        bname = b_item["name"]
        sub = df_rec[df_rec["bead_name"] == bname]
        if sub.empty:
            continue

        n_p = len(sub)
        rc_xi = float(sub["rc_over_xi"].mean())
        rc_xi_sem = float(sub["rc_over_xi"].sem()) if n_p > 1 else 0.0

        ht_arr = np.vstack(sub["hist_th"].values)
        ht_tot = ht_arr.sum(axis=0).astype(float)
        P_th = ht_tot / (ht_tot.sum() * dth)

        with np.errstate(divide="ignore", invalid="ignore"):
            U_th = np.where(P_th > 0, -np.log(2.0 * np.pi * P_th), np.nan)
        u_pi = float(np.nanmean(U_th[pi_indices]))
        u_0 = float(U_th[idx_zero])
        delta_u = u_pi - u_0

        boot_du = []
        for _ in range(B_BOOT):
            idx = rng.integers(0, n_p, size=n_p)
            ht_b = ht_arr[idx].sum(axis=0).astype(float)
            pth_b = ht_b / (ht_b.sum() * dth)
            with np.errstate(divide="ignore", invalid="ignore"):
                uth_b = np.where(pth_b > 0, -np.log(2.0 * np.pi * pth_b), np.nan)
            up_b = np.nanmean(uth_b[pi_indices])
            u0_b = uth_b[idx_zero]
            if np.isfinite(up_b) and np.isfinite(u0_b):
                boot_du.append(up_b - u0_b)

        ci_lo = float(np.percentile(boot_du, 2.5))
        ci_hi = float(np.percentile(boot_du, 97.5))
        sem_du = float(np.std(boot_du))

        rows.append({
            "bead_name": bname,
            "diameter_um": b_item["diameter_um"],
            "label": b_item["label"],
            "color": b_item["color"],
            "marker": b_item["marker"],
            "rc_over_xi": rc_xi,
            "rc_over_xi_sem": rc_xi_sem,
            "n_particles": n_p,
            "delta_U_eff": delta_u,
            "delta_U_sem": sem_du,
            "delta_U_ci_low": ci_lo,
            "delta_U_ci_high": ci_hi,
        })

    df_out = pd.DataFrame(rows).sort_values("rc_over_xi").reset_index(drop=True)
    for d in OUT_DIRS:
        df_out.drop(columns=["color", "marker"]).to_csv(d / "per_point_delta_U_vs_rc_xi.csv", index=False)

    print("=== Per-Point Well Depth Delta U_eff = U(pi) - U(0) vs R_c / xi ===")
    print(df_out[["bead_name", "diameter_um", "rc_over_xi", "delta_U_eff", "delta_U_ci_low", "delta_U_ci_high"]].to_string(index=False))

    # 指数減衰フィット
    x_data = df_out["rc_over_xi"].values
    y_data = df_out["delta_U_eff"].values
    popt, _ = curve_fit(exp_decay, x_data, y_data, p0=[2.0, 0.5], bounds=([0.1, 0.05], [10.0, 5.0]))
    x_fit = np.logspace(np.log10(0.04), np.log10(3.2), 200)
    y_fit = exp_decay(x_fit, *popt)

    # 2パネル: (A) 片対数 (log x), (B) 両対数 (log-log)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    for ax, use_logy in [(ax1, False), (ax2, True)]:
        ax.plot(x_fit, y_fit, color="#333333", linestyle="--", linewidth=1.8, alpha=0.85,
                label=rf"Exp fit: ${popt[0]:.2f}\,\exp(-R_c/({popt[1]:.2f}\xi))$", zorder=2)
        for _, r in df_out.iterrows():
            yerr = [[r["delta_U_eff"] - r["delta_U_ci_low"]], [r["delta_U_ci_high"] - r["delta_U_eff"]]]
            ax.errorbar(r["rc_over_xi"], r["delta_U_eff"], yerr=yerr, xerr=r["rc_over_xi_sem"],
                        fmt=r["marker"], color=r["color"], markersize=9.5, capsize=4.5, elinewidth=1.8,
                        markeredgecolor="k", markeredgewidth=0.8, label=r["label"], zorder=4)

        ax.axhline(1.0, color="#d62728", linestyle=":", linewidth=1.4, alpha=0.85,
                   label=r"$\Delta\tilde{U}_{\mathrm{eff}} = 1$ ($k_B T_{\mathrm{eff}}$)")
        ax.axvline(1.0, color="black", linestyle="-.", linewidth=1.2, alpha=0.6, label=r"$R_c = \xi$")
        ax.set_xscale("log")
        if use_logy:
            ax.set_yscale("log")
            ax.set_ylim(0.06, 4.0)
            ax.set_title(r"(B) Log-Log Scale", fontsize=13, fontweight="bold")
        else:
            ax.axhline(0.0, color="#888888", linestyle="-", linewidth=0.9)
            ax.set_ylim(-0.1, 2.7)
            ax.set_title(r"(A) Semi-Log Scale", fontsize=13, fontweight="bold")

        ax.set_xlim(0.04, 3.2)
        ax.set_xlabel(r"Dimensionless Cargo Radius $R_c / \xi$", fontsize=13)
        ax.set_ylabel(r"Well Depth $\Delta\tilde{U}_{\mathrm{eff}} = \tilde{U}_{\mathrm{eff}}(\pi) - \tilde{U}_{\mathrm{eff}}(0)$", fontsize=12.5)
        ax.grid(True, which="both", linestyle=":", alpha=0.5)

    ax1.legend(loc="upper right", fontsize=8.8, framealpha=0.92, ncol=2)
    fig.suptitle(r"Felt Potential Well Depth $\Delta\tilde{U}_{\mathrm{eff}}$ vs $R_c/\xi$ (Per-Point $\hat{\mathbf{v}}_i \cdot \hat{\mathbf{u}}$)",
                 fontsize=14.5, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    vcd.save_figure(fig, "per_point_delta_U_vs_rc_xi", OUT_DIRS)
    plt.close(fig)


if __name__ == "__main__":
    main()
