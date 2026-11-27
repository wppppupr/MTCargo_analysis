from pathlib import Path
import numpy as np
import pandas as pd
from style import COLOR_MT, COLOR_CARGO, COLOR_TOTAL

ROOT = Path(__file__).resolve().parent.parent.parent
CSV_CANDIDATES = [
    ROOT / "figure" / "msd" / "msd_kl_relaxation_reconstruction_summary.csv",
    Path("/Volumes/data/Sasaki/MTsingleBeads/figure/msd/msd_kl_relaxation_reconstruction_summary.csv"),
    ROOT / "figure" / "lagrangian_flow_autocorr" / "lagrangian_u_bar_autocorr_summary.csv",
    Path("/Volumes/data/Sasaki/MTsingleBeads/figure/lagrangian_flow_autocorr/lagrangian_u_bar_autocorr_summary.csv"),
]


def draw(fig):
    """Draw relaxation times (tau_u_Lag, tau_v_slow, tau_GK) vs Rc / xi."""
    ax = fig.add_subplot(111)
    csv_path = next((p for p in CSV_CANDIDATES if p.exists()), None)
    if csv_path is None:
        ax.text(0.5, 0.5, "Missing relaxation summary CSV", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    df = pd.read_csv(csv_path).sort_values("rc_over_xi")
    x = df["rc_over_xi"].to_numpy(dtype=float)

    candidates = [
        ("tau_u_lag_s", COLOR_MT, "o", "-", r"$\tau_{\bar{\mathbf{u}}}^{\mathrm{Lag}}$ (flow)"),
        ("tau_v_slow_s", COLOR_CARGO, "s", "--", r"$\tau_{\mathbf{v}}^{\mathrm{slow}}$ (cargo)"),
        ("tau_gk_s", COLOR_TOTAL, "^", "-.", r"$\tau_{\mathrm{GK}}$ (memory)"),
        ("tau_1e_s", COLOR_MT, "o", "-", r"$\tau_{1/e}$"),
        ("tau_slow_fit_s", COLOR_CARGO, "s", "--", r"$\tau_{\mathrm{slow}}$"),
    ]

    plotted = False
    for col, color, marker, ls, label in candidates:
        if col in df.columns and df[col].notna().any():
            ax.plot(x, df[col].to_numpy(dtype=float), marker=marker, linestyle=ls, markersize=3.5,
                    color=color, label=label, zorder=4)
            plotted = True

    if not plotted:
        ax.text(0.5, 0.5, "No relaxation time columns found", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.045, 3.2)
    ax.set_xlabel(r"Dimensionless Cargo Radius $R_c / \xi$")
    ax.set_ylabel(r"Relaxation Time $\tau$ [s]")
    ax.legend(loc="best", framealpha=0.88, markerscale=0.85, handlelength=1.4)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
