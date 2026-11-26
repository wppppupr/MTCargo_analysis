from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
CSV_CANDIDATES = [
    ROOT / "figure" / "k1_and_excess_pinning_summary.csv",
    ROOT / "k1_and_excess_pinning_summary.csv",
]


def draw(fig):
    """Draw raw two-stage crossover: kappa_1 and Delta U_+1 vs Rc / xi."""
    ax = fig.add_subplot(111)
    csv_path = next((p for p in CSV_CANDIDATES if p.exists()), None)
    if csv_path is None:
        ax.text(0.5, 0.5, "Missing:\nk1_and_excess_pinning_summary.csv\nRun analysis/calc_k1_and_excess_pinning.py",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    df_out = pd.read_csv(csv_path).sort_values("rc_over_xi")
    ax.plot(df_out["rc_over_xi"], df_out["delta_U_plus1"], "-", color="#d62728", alpha=0.8,
            label=r"Excess rail pinning $\Delta\tilde{U}_{+1}$", zorder=3)
    ax.plot(df_out["rc_over_xi"], df_out["kappa_1"], "--", color="#1f77b4", alpha=0.85,
            label=r"Bulk alignment slope $\kappa_1$", zorder=3)

    markers = ["o", "s", "v", "D", "P", "*"]
    for i, (_, r) in enumerate(df_out.iterrows()):
        m = markers[i % len(markers)]
        yerr_du = [[r["delta_U_plus1"] - r["delta_U_plus1_ci_low"]], [r["delta_U_plus1_ci_high"] - r["delta_U_plus1"]]]
        yerr_k1 = [[r["kappa_1"] - r["kappa_1_ci_low"]], [r["kappa_1_ci_high"] - r["kappa_1"]]]
        ax.errorbar(r["rc_over_xi"], r["delta_U_plus1"], yerr=yerr_du, xerr=r["rc_over_xi_sem"],
                    fmt=m, color="#d62728", markersize=4.5, capsize=3.0, markeredgecolor="k", zorder=5)
        ax.errorbar(r["rc_over_xi"], r["kappa_1"], yerr=yerr_k1, xerr=r["rc_over_xi_sem"],
                    fmt=m, color="#1f77b4", markersize=4.5, capsize=3.0, markeredgecolor="k", zorder=4)

    x_lock = 45.0 * np.pi / 512.0
    x_bulk = 45.0 * np.pi / 128.0
    ax.axvline(x_lock, color="#d95f02", linestyle="--", alpha=0.85, label=r"$x_{\mathrm{lock}} \approx 0.28$")
    ax.axvline(x_bulk, color="#222222", linestyle=":", alpha=0.85, label=r"$x_{\mathrm{bulk}} \approx 1.10$")
    ax.axhline(0.0, color="#888888", linestyle="-")
    ax.set_xscale("log")
    ax.set_xlim(0.04, 3.0)
    ax.set_ylim(-0.08, 1.18)
    ax.set_xlabel(r"Dimensionless Cargo Radius $R_c / \xi$")
    ax.set_ylabel(r"Coupling / Pinning Depth ($\kappa_1,\ \Delta\tilde{U}_{+1}$)")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.92)
