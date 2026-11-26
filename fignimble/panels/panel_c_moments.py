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
    """Draw dual-axis 1st and 2nd moments (m1 = <cos theta>, m2 = <cos 2theta>) vs Rc / xi without legend."""
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()

    csv_path = next((p for p in CSV_CANDIDATES if p.exists()), None)
    if csv_path is None:
        ax1.text(0.5, 0.5, "Missing:\ngvm_moments_and_potential_summary.csv",
                 ha="center", va="center", transform=ax1.transAxes)
        ax1.set_axis_off()
        ax2.set_axis_off()
        return

    df = pd.read_csv(csv_path).sort_values("rc_over_xi")
    x = df["rc_over_xi"].to_numpy(dtype=float)

    c1 = ACTIVE_PALETTE[0]
    c2 = ACTIVE_PALETTE[-1]

    m1 = df["m1_cos_theta"].to_numpy(dtype=float)
    m1_lo = df["m1_ci_low"].to_numpy(dtype=float)
    m1_hi = df["m1_ci_high"].to_numpy(dtype=float)

    m2 = df["m2_cos_2theta"].to_numpy(dtype=float)
    m2_lo = df["m2_ci_low"].to_numpy(dtype=float)
    m2_hi = df["m2_ci_high"].to_numpy(dtype=float)

    x_lock = 45.0 * np.pi / 512.0
    x_bulk = 45.0 * np.pi / 128.0
    ax1.axhline(0.0, color="#666666", linestyle="--", alpha=0.7, zorder=1)
    ax1.axvline(x_lock, color="#888888", linestyle=":", alpha=0.7, zorder=1)
    ax1.axvline(x_bulk, color="#888888", linestyle="-.", alpha=0.7, zorder=1)

    ax1.fill_between(x, m1_lo, m1_hi, color=c1, alpha=0.18, edgecolor="none", zorder=2)
    ax1.plot(x, m1, marker="o", linestyle="-", markersize=3.0, color=c1, zorder=4)

    ax2.fill_between(x, m2_lo, m2_hi, color=c2, alpha=0.18, edgecolor="none", zorder=2)
    ax2.plot(x, m2, marker="s", linestyle="--", markersize=3.0, color=c2, zorder=4)

    ax1.set_xscale("log")
    ax1.set_xlim(0.045, 3.2)
    ax1.set_xlabel(r"Dimensionless Cargo Radius $R_c / \xi$")
    ax1.set_ylabel(r"1st moment $\langle \cos\theta \rangle$", color=c1)
    ax1.tick_params(axis="y", labelcolor=c1)

    ax2.set_ylabel(r"2nd moment $\langle \cos 2\theta \rangle$", color=c2)
    ax2.tick_params(axis="y", labelcolor=c2)

    #ax1.grid(True, which="both", linestyle="--", alpha=0.35)
