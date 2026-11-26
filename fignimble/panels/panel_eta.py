from pathlib import Path
import numpy as np
import pandas as pd
from style import ACTIVE_PALETTE

ROOT = Path(__file__).resolve().parent.parent.parent
CSV_CANDIDATES = [
    ROOT / "figure" / "per_point_v_dot_u" / "data_processing_inequality_summary.csv",
    ROOT / "figure" / "data_processing_inequality_summary.csv",
    ROOT / "data_processing_inequality_summary.csv",
    Path("/Volumes/data/Sasaki/MTsingleBeads/figure/per_point_v_dot_u/data_processing_inequality_summary.csv"),
]


def draw(fig):
    """Draw information efficiency decomposition (eta_info, eta_geom, eta_mech) vs Rc / xi."""
    ax = fig.add_subplot(111)
    csv_path = next((p for p in CSV_CANDIDATES if p.exists()), None)
    if csv_path is None:
        ax.text(0.5, 0.5, "Missing:\ndata_processing_inequality_summary.csv",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    df = pd.read_csv(csv_path).sort_values("rc_over_xi")
    x = df["rc_over_xi"].to_numpy(dtype=float)

    series = [
        ("info_efficiency_D_KL_over_D_MT", "eta_info_ci_low", "eta_info_ci_high", ACTIVE_PALETTE[0], "s", "-", r"$\eta = D_{\mathrm{KL}} / D_{\mathrm{MT}}$"),
        ("eta_geom", "eta_geom_ci_low", "eta_geom_ci_high", ACTIVE_PALETTE[2], "D", "-.", r"$\eta_{\mathrm{geom}}$"),
        ("eta_mech", "eta_mech_ci_low", "eta_mech_ci_high", ACTIVE_PALETTE[-1], "o", "--", r"$\eta_{\mathrm{mech}}$"),
    ]

    ax.axhline(1.0, color="#666666", linestyle=":", alpha=0.7, zorder=1)
    ax.axvline(1.0, color="#888888", linestyle="-.", alpha=0.7, zorder=1, label=r"$R_c / \xi = 1$")

    for col_val, col_lo, col_hi, color, marker, ls, label in series:
        if col_val not in df.columns:
            continue
        y = df[col_val].to_numpy(dtype=float)
        y_lo = df[col_lo].to_numpy(dtype=float)
        y_hi = df[col_hi].to_numpy(dtype=float)
        ax.fill_between(x, np.clip(y_lo, 0.0, None), y_hi, color=color, alpha=0.18, edgecolor="none", zorder=2)
        ax.plot(x, y, marker=marker, linestyle=ls, markersize=3.0, color=color, label=label, zorder=4)

    ax.set_xscale("log")
    ax.set_xlim(0.045, 3.2)
    ax.set_ylim(0.0, 1.5)
    ax.set_xlabel(r"Dimensionless Cargo Radius $R_c / \xi$")
    ax.set_ylabel(r"Efficiency $\eta$")
    #ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", framealpha=0.88, markerscale=0.85, handlelength=1.4)
