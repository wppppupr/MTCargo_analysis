from pathlib import Path
import numpy as np
import pandas as pd
from style import ACTIVE_PALETTE

ROOT = Path(__file__).resolve().parent.parent.parent
CSV_CANDIDATES = [
    ROOT / "figure" / "msd" / "msd_kl_relaxation_reconstruction_summary.csv",
    Path("/Volumes/data/Sasaki/MTsingleBeads/figure/msd/msd_kl_relaxation_reconstruction_summary.csv"),
]


def draw(fig):
    """Compare observed MSD(240s) with KL + relaxation-time theoretical predictions."""
    ax = fig.add_subplot(111)
    csv_path = next((p for p in CSV_CANDIDATES if p.exists()), None)
    if csv_path is None:
        ax.text(0.5, 0.5, "Missing:\nmsd_kl_relaxation_reconstruction_summary.csv",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    df = pd.read_csv(csv_path).sort_values("diameter_um")
    d = df["diameter_um"].to_numpy(dtype=float)

    ax.plot(d, df["msd_obs_240s"].to_numpy(dtype=float), marker="o", linestyle="-", markersize=4.0,
            color=ACTIVE_PALETTE[0], label=r"Observed $\mathrm{MSD}(240\,\mathrm{s})$", zorder=4)
    ax.plot(d, df["msd_kl_240s"].to_numpy(dtype=float), marker="s", linestyle="--", markersize=3.5,
            color=ACTIVE_PALETTE[-1], label=r"Theory: $D_{\mathrm{KL}} + \tau_{\mathbf{v}}^{\mathrm{slow}}$", zorder=4)
    ax.plot(d, df["msd_kl_flow_240s"].to_numpy(dtype=float), marker="^", linestyle="-.", markersize=3.5,
            color=ACTIVE_PALETTE[2], label=r"Theory: $D_{\mathrm{KL}} + \tau_{\bar{\mathbf{u}}}^{\mathrm{Lag}}$", zorder=3)
    if "msd_brown_240s" in df.columns:
        ax.plot(d, df["msd_brown_240s"].to_numpy(dtype=float), marker="None", linestyle=":",
                color="#777777", label=r"Uncorrelated $\mathrm{MSD}_{\mathrm{zero}}$", zorder=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.45, 28.0)
    ax.set_xlabel(r"Cargo Diameter $2R_c$ [$\mu\mathrm{m}$]")
    ax.set_ylabel(r"$\mathrm{MSD}(\Delta t = 240\,\mathrm{s})$ [$\mu\mathrm{m}^2$]")
    ax.legend(loc="lower left", framealpha=0.88, markerscale=0.85, handlelength=1.4)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
