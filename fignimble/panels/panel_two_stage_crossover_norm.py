from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
CSV_CANDIDATES = [
    ROOT / "figure" / "k1_and_excess_pinning_summary.csv",
    ROOT / "k1_and_excess_pinning_summary.csv",
]


def draw(fig):
    """Draw normalized two-stage crossover: kappa_1 and Delta U_+1 relative to smallest cargo."""
    ax = fig.add_subplot(111)
    csv_path = next((p for p in CSV_CANDIDATES if p.exists()), None)
    if csv_path is None:
        ax.text(0.5, 0.5, "Missing:\nk1_and_excess_pinning_summary.csv\nRun analysis/calc_k1_and_excess_pinning.py",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    df_out = pd.read_csv(csv_path).sort_values("rc_over_xi")
    du0 = float(df_out["delta_U_plus1"].iloc[0])
    k10 = float(df_out["kappa_1"].iloc[0])

    ax.plot(df_out["rc_over_xi"], df_out["delta_U_plus1"] / du0, "o-", color="#d62728",
            markersize=4.5, markeredgecolor="k", label=r"$\Delta\tilde{U}_{+1} / \Delta\tilde{U}_{+1}^{(0)}$")
    ax.plot(df_out["rc_over_xi"], df_out["kappa_1"] / k10, "s--", color="#1f77b4",
            markersize=4.2, markeredgecolor="k", label=r"$\kappa_1 / \kappa_1^{(0)}$")

    for _, r in df_out.iterrows():
        ax.annotate(f"{r['diameter_um']:.2g} µm", xy=(r["rc_over_xi"], r["kappa_1"] / k10),
                    xytext=(0, 7), textcoords="offset points", ha="center", color="#1f77b4")

    x_lock = 45.0 * np.pi / 512.0
    x_bulk = 45.0 * np.pi / 128.0
    ax.axvline(x_lock, color="#d95f02", linestyle="--", alpha=0.85, label=r"$x_{\mathrm{lock}} \approx 0.28$")
    ax.axvline(x_bulk, color="#222222", linestyle=":", alpha=0.85, label=r"$x_{\mathrm{bulk}} \approx 1.10$")
    ax.axhline(0.0, color="#888888", linestyle="-")
    ax.set_xscale("log")
    ax.set_xlim(0.04, 3.0)
    ax.set_ylim(-0.08, 1.30)
    ax.set_xlabel(r"Dimensionless Cargo Radius $R_c / \xi$")
    ax.set_ylabel("Normalized Strength")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.92)
