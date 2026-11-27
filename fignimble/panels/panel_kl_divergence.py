from pathlib import Path
import numpy as np
import pandas as pd
from style import COLOR_CARGO, COLOR_POLAR, COLOR_NEMATIC, COLOR_RESIDUAL

ROOT = Path(__file__).resolve().parent.parent.parent
CSV_CANDIDATES = [
    ROOT / "figure" / "per_point_v_dot_u" / "gvm_moments_and_potential_summary.csv",
    ROOT / "figure" / "gvm_moments_and_potential_summary.csv",
    Path("/Volumes/data/Sasaki/MTsingleBeads/figure/per_point_v_dot_u/gvm_moments_and_potential_summary.csv"),
]


def draw(fig):
    """Draw 3-term KL divergence decomposition (D_KL_total, D_KL_1, D_KL_2, Delta_D_ge3) vs Rc / xi."""
    ax = fig.add_subplot(111)
    csv_path = next((p for p in CSV_CANDIDATES if p.exists()), None)
    if csv_path is None:
        ax.text(0.5, 0.5, "Missing:\ngvm_moments_and_potential_summary.csv",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    df = pd.read_csv(csv_path).sort_values("rc_over_xi")
    x = df["rc_over_xi"].to_numpy(dtype=float)

    series = [
        ("D_KL_total", "D_KL_total_ci_low", "D_KL_total_ci_high", COLOR_CARGO, "o", "-", r"$D_{\mathrm{KL}}^{\mathrm{total}}$"),
        ("D_KL_1_polar", "D_KL_1_ci_low", "D_KL_1_ci_high", COLOR_POLAR, "s", "--", r"$D_{\mathrm{KL}}^{(1)}$ (polar)"),
        ("D_KL_2_nematic", "D_KL_2_ci_low", "D_KL_2_ci_high", COLOR_NEMATIC, "^", "-.", r"$D_{\mathrm{KL}}^{(2)}$ (nematic)"),
        ("Delta_D_ge3_residual", "Delta_D_ge3_ci_low", "Delta_D_ge3_ci_high", COLOR_RESIDUAL, "D", ":", r"$\Delta D_{\geq 3}$ (residual)"),
    ]

    x_lock = 45.0 * np.pi / 512.0
    x_bulk = 45.0 * np.pi / 128.0
    ax.axvline(x_lock, color="#888888", linestyle=":", alpha=0.7, zorder=1, label=r"$x_{\mathrm{lock}} \approx 0.28$")
    ax.axvline(x_bulk, color="#888888", linestyle="-.", alpha=0.7, zorder=1, label=r"$x_{\mathrm{bulk}} \approx 1.10$")

    for col_val, col_lo, col_hi, color, marker, ls, label in series:
        y = df[col_val].to_numpy(dtype=float)
        y_lo = df[col_lo].to_numpy(dtype=float)
        y_hi = df[col_hi].to_numpy(dtype=float)
        yerr = [np.clip(y - y_lo, 0.0, None), np.clip(y_hi - y, 0.0, None)]
        ax.errorbar(x, y, yerr=yerr, marker=marker, linestyle=ls, markersize=3.5,
                    color=color, ecolor=color, capsize=2.2, label=label, zorder=4)

    ax.set_xscale("log")
    #ax.set_yscale("log")
    ax.set_xlim(0.045, 3.2)
    ax.set_ylim(0, 0.25)
    ax.set_xlabel(r"Dimensionless Cargo Radius $R_c / \xi$")
    ax.set_ylabel("KL Divergence [nats]")
    #ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", framealpha=0.88, markerscale=0.85, handlelength=1.4, facecolor="black")
