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
    ROOT / "figure" / "ngp" / "NGP_summary_2d.csv",
    ROOT / "figure" / "NGP_summary_2d.csv",
    ROOT / "NGP_summary_2d.csv",
    Path("/Volumes/data/Sasaki/MTsingleBeads/figure/ngp/NGP_summary_2d.csv"),
]


def draw(fig):
    """Draw 2D Non-Gaussian Parameter (NGP) vs lag time (semilog-x)."""
    ax = fig.add_subplot(111)
    csv_path = next((p for p in CSV_CANDIDATES if p.exists()), None)
    if csv_path is None:
        ax.text(0.5, 0.5, "Missing:\nNGP_summary_2d.csv", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    df = pd.read_csv(csv_path)
    for d_um, grp in df.groupby("diameter_um", sort=True):
        grp = grp.sort_values("lag_time_s")
        if grp.empty:
            continue

        bname = str(grp["bead_name"].iloc[0]) if "bead_name" in grp.columns else None
        color, marker = get_bead_style(d_um=d_um, bead_name=bname)
        lags = grp["lag_time_s"].to_numpy(dtype=float)
        ngp = grp["ngp_pooled"].to_numpy(dtype=float)
        ci_low = grp["ngp_pooled_boot_ci_low"].to_numpy(dtype=float)
        ci_high = grp["ngp_pooled_boot_ci_high"].to_numpy(dtype=float)
        n_exp = int(grp["n_experiments"].iloc[0]) if "n_experiments" in grp.columns else 0

        ax.fill_between(lags, ci_low, ci_high, color=color, alpha=0.18, edgecolor="none")

        n_pts = len(lags)
        if n_pts > 20:
            target_lags = np.geomspace(lags[0], lags[-1], num=18)
            sub_idx = np.unique([np.argmin(np.abs(lags - tg)) for tg in target_lags])
        else:
            sub_idx = np.arange(n_pts)

        yerr_rep = [
            np.clip(ngp[sub_idx] - ci_low[sub_idx], 0, None),
            np.clip(ci_high[sub_idx] - ngp[sub_idx], 0, None),
        ]
        ax.errorbar(lags[sub_idx], ngp[sub_idx], yerr=yerr_rep, fmt="none", ecolor=color, capsize=2.5, alpha=0.85, zorder=3)

        label = f"{d_um:.2f} µm ($N={n_exp}$)" if n_exp > 0 else f"{d_um:.2f} µm"
        ax.plot(lags, ngp, marker=marker, linestyle="-", markersize=3.5, color=color, label=label, zorder=4)

    ref_color = "white" if is_dark() else "#333333"
    ax.axhline(0.0, color=ref_color, linestyle="--", alpha=0.7, zorder=2)
    ax.set_xscale("log")
    ax.set_xlim(4.0, 200.0)
    ax.set_xlabel(r"Lag time $\Delta t$ [s]")
    ax.set_ylabel(r"Non-Gaussian parameter $\alpha_2(\Delta t)$")
    #ax.legend(loc="upper right", framealpha=0.88, markerscale=0.8, handlelength=1.4)
    #ax.grid(True, which="both", linestyle="--", alpha=0.35)
