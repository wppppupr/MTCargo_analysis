from pathlib import Path
import numpy as np
import pandas as pd
from style import get_bead_style

try:
    from fignimble import is_dark
except ImportError:
    is_dark = lambda: False

ROOT = Path(__file__).resolve().parent.parent.parent
CSV_CANDIDATES = [
    ROOT / "figure" / "msd" / "msd_kl_relaxation_curves.csv",
    Path("/Volumes/data/Sasaki/MTsingleBeads/figure/msd/msd_kl_relaxation_curves.csv"),
]


def draw_msd_kl(fig, bead_name: str, show_xlabel: bool = True, show_ylabel: bool = True):
    """Draw observed MSD(dt) markers vs zero-parameter theoretical MSD(dt) for one cargo size."""
    ax = fig.add_subplot(111)
    csv_path = next((p for p in CSV_CANDIDATES if p.exists()), None)
    if csv_path is None:
        ax.text(0.5, 0.5, "Missing:\nmsd_kl_relaxation_curves.csv\nRun analysis/calc_msd_from_kl_and_relaxation.py",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    df = pd.read_csv(csv_path)
    sub = df[df["bead_name"] == bead_name].sort_values("lag_s")
    if sub.empty:
        ax.text(0.5, 0.5, f"No data for {bead_name}", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    d_um = float(sub["diameter_um"].iloc[0])
    color, marker = get_bead_style(d_um=d_um, bead_name=bead_name)

    lags_s = sub["lag_s"].to_numpy(dtype=float)
    msd_obs = sub["msd_obs"].to_numpy(dtype=float)
    msd_ci_lo = sub["msd_ci_lo"].to_numpy(dtype=float)
    msd_ci_hi = sub["msd_ci_hi"].to_numpy(dtype=float)
    msd_kl = sub["msd_kl"].to_numpy(dtype=float)

    ax.fill_between(lags_s, np.clip(msd_ci_lo, 1e-4, None), msd_ci_hi,
                    color=color, alpha=0.18, edgecolor="none", zorder=2)
    ax.plot(lags_s, msd_obs, marker=marker, linestyle="none", markersize=3.0, color=color,
            label=f"Obs. {d_um:.2f} µm", zorder=5)
    ax.plot(lags_s, msd_kl, linestyle="-", color="white" if is_dark() else "#000000",
            label=r"$D_{\mathrm{KL}}+\tau_{\mathbf{v}}^{\mathrm{slow}}$", zorder=6)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(4.0, 320.0)
    ax.set_ylim(0.08, 1200.0)
    if show_xlabel:
        ax.set_xlabel(r"Lag time $\Delta t$ [s]")
    if show_ylabel:
        ax.set_ylabel(r"MSD [$\mu\mathrm{m}^2$]")
    #ax.legend(loc="lower right", framealpha=0.88, markerscale=0.85, handlelength=1.4)
    #ax.grid(True, which="both", linestyle="--", alpha=0.35)
