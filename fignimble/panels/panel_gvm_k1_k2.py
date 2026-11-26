from pathlib import Path
import numpy as np
import pandas as pd
from style import ACTIVE_PALETTE

ROOT = Path(__file__).resolve().parent.parent.parent
CSV_CANDIDATES = [
    ROOT / "figure" / "per_point_v_dot_u" / "gvm_moments_and_potential_summary.csv",
    ROOT / "figure" / "gvm_moments_and_potential_summary.csv",
    Path("/Volumes/data/Sasaki/MTsingleBeads/figure/per_point_v_dot_u/gvm_moments_and_potential_summary.csv"),
]


def draw(fig):
    """Draw dual-axis GvM coupling parameters (kappa_1, kappa_2) vs Rc / xi without legend."""
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()

    csv_path = next((p for p in CSV_CANDIDATES if p.exists()), None)
    if csv_path is None:
        ax1.text(0.5, 0.5, "Missing:\ngvm_moments_and_potential_summary.csv",
                 ha="center", va="center", transform=ax1.transAxes)
        ax1.set_axis_off()
        ax2.set_axis_off()
        return

    df_out = pd.read_csv(csv_path).sort_values("rc_over_xi")
    x = df_out["rc_over_xi"].to_numpy(dtype=float)

    c1 = ACTIVE_PALETTE[0]
    c2 = ACTIVE_PALETTE[-1]

    k1 = df_out["kappa_1_gvm"].to_numpy(dtype=float)
    k1_lo = df_out["kappa_1_ci_low"].to_numpy(dtype=float)
    k1_hi = df_out["kappa_1_ci_high"].to_numpy(dtype=float)

    k2 = df_out["kappa_2_gvm"].to_numpy(dtype=float)
    k2_lo = df_out["kappa_2_ci_low"].to_numpy(dtype=float)
    k2_hi = df_out["kappa_2_ci_high"].to_numpy(dtype=float)

    x_lock = 45.0 * np.pi / 512.0
    x_bulk = 45.0 * np.pi / 128.0
    ax1.axhline(0.0, color="#666666", linestyle="--", alpha=0.7, zorder=1)
    ax1.axvline(x_lock, color="#888888", linestyle=":", alpha=0.7, zorder=1)
    ax1.axvline(x_bulk, color="#888888", linestyle="-.", alpha=0.7, zorder=1)

    ax1.fill_between(x, k1_lo, k1_hi, color=c1, alpha=0.18, edgecolor="none", zorder=2)
    ax1.plot(x, k1, marker="o", linestyle="-", markersize=3.0, color=c1, zorder=4)

    ax2.fill_between(x, k2_lo, k2_hi, color=c2, alpha=0.18, edgecolor="none", zorder=2)
    ax2.plot(x, k2, marker="s", linestyle="--", markersize=3.0, color=c2, zorder=4)

    ax1.set_xscale("log")
    ax1.set_xlim(0.045, 3.2)
    ax1.set_xlabel(r"Dimensionless Cargo Radius $R_c / \xi$")
    ax1.set_ylabel(r"Polar coupling $\kappa_1$", color=c1)
    ax1.tick_params(axis="y", labelcolor=c1)

    ax2.set_ylabel(r"Nematic coupling $\kappa_2$", color=c2)
    ax2.tick_params(axis="y", labelcolor=c2)

    #ax1.grid(True, which="both", linestyle="--", alpha=0.35)
