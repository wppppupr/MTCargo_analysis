from pathlib import Path
import numpy as np
import pandas as pd
from style import get_bead_style

try:
    from fignimble import is_dark
except ImportError:
    is_dark = lambda: False

ROOT = Path(__file__).resolve().parent.parent.parent
CSV_PATH = ROOT / "figure" / "msad" / "msad_pooled_mean_iqr_summary.csv"


def draw(fig):
    """Draw pooled ensemble-mean MSAD (rad^2) with bootstrap 95% CI."""
    ax = fig.add_subplot(111)
    if not CSV_PATH.exists():
        ax.text(0.5, 0.5, f"Missing:\n{CSV_PATH.name}\nRun analysis/calc_msad.py",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    df = pd.read_csv(CSV_PATH)
    for d_um, grp in df.groupby("diameter_um", sort=True):
        grp = grp[grp["n_active_particles"] >= 2].sort_values("lag_time_s")
        if grp.empty:
            continue

        color, marker = get_bead_style(d_um=d_um)
        lags = grp["lag_time_s"].to_numpy(dtype=float)
        mean_v = grp["mean_msad_rad2"].to_numpy(dtype=float)
        ci_low = grp["boot_ci_low_msad_rad2"].to_numpy(dtype=float)
        ci_high = grp["boot_ci_high_msad_rad2"].to_numpy(dtype=float)
        n_filt = int(grp["n_filtered_particles"].iloc[0])

        ax.fill_between(lags, np.clip(ci_low, 1e-4, None), ci_high, color=color, alpha=0.18, edgecolor="none")

        n_pts = len(lags)
        if n_pts > 20:
            target_lags = np.geomspace(lags[0], lags[-1], num=18)
            sub_idx = np.unique([np.argmin(np.abs(lags - tg)) for tg in target_lags])
        else:
            sub_idx = np.arange(n_pts)

        yerr_rep = [
            np.clip(mean_v[sub_idx] - ci_low[sub_idx], 0, None),
            np.clip(ci_high[sub_idx] - mean_v[sub_idx], 0, None),
        ]
        ax.errorbar(lags[sub_idx], mean_v[sub_idx], yerr=yerr_rep, fmt="none", ecolor=color, capsize=2.5, alpha=0.85, zorder=3)
        ax.plot(lags, mean_v, marker=marker, linestyle="-", markersize=3.5, label=f"{d_um:.2f} µm ($N={n_filt}$)", color=color, zorder=4)

    guide_color = "white" if is_dark() else "#333333"
    t_g1, t_g2 = 50.0, 200.0
    ax.plot([t_g1, t_g2], [18.0 * (t_g1 / 50.0) ** 1.0, 18.0 * (t_g2 / 50.0) ** 1.0], color=guide_color, linestyle="--")
    ax.text(140, 30, r"$\propto \Delta t^{1.0}$", color=guide_color)

    ax.legend(loc="upper left", framealpha=0.88, markerscale=0.8, handlelength=1.4)
    ax.set(xlim=(4.0, 1000.0), ylim=(1.0, 2e3), xscale="log", yscale="log",
           xlabel=r"Lag time $\Delta t$ [s]", ylabel=r"MSAD $\langle (\Delta \theta)^2 \rangle$ [$\mathrm{rad}^2$]")
