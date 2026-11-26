from pathlib import Path
import numpy as np
import pandas as pd
from style import get_bead_style

try:
    from fignimble import is_dark
except ImportError:
    is_dark = lambda: False

ROOT = Path(__file__).resolve().parent.parent.parent
CSV_PATH = ROOT / "figure" / "msd" / "msd_dimensionless_summary.csv"


def draw(fig):
    """Draw dimensionless ensemble-mean MSD with bootstrap 95% CI."""
    ax = fig.add_subplot(111)
    if not CSV_PATH.exists():
        ax.text(0.5, 0.5, f"Missing:\n{CSV_PATH.name}\nRun analysis/calc_msd.py",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    df = pd.read_csv(CSV_PATH)
    for d_um, grp in df.groupby("diameter_um", sort=True):
        grp = grp[grp["n_active_particles"] >= 2].sort_values("t_tilde")
        if grp.empty:
            continue

        color, marker = get_bead_style(d_um=d_um)
        t_vals = grp["t_tilde"].to_numpy(dtype=float)
        mean_v = grp["mean_msd_tilde"].to_numpy(dtype=float)
        ci_low = grp["boot_ci_low_msd_tilde"].to_numpy(dtype=float)
        ci_high = grp["boot_ci_high_msd_tilde"].to_numpy(dtype=float)
        n_part = int(grp["n_active_particles"].max())

        ax.fill_between(
            t_vals,
            np.clip(ci_low, 1e-5, None),
            ci_high,
            color=color,
            alpha=0.18,
            edgecolor="none",
        )

        n_pts = len(t_vals)
        if n_pts > 20:
            target_t = np.geomspace(t_vals[0], t_vals[-1], num=16)
            sub_idx = np.unique([np.argmin(np.abs(t_vals - tg)) for tg in target_t])
        else:
            sub_idx = np.arange(n_pts)

        yerr_rep = [
            np.clip(mean_v[sub_idx] - ci_low[sub_idx], 0, None),
            np.clip(ci_high[sub_idx] - mean_v[sub_idx], 0, None),
        ]
        ax.errorbar(
            t_vals[sub_idx],
            mean_v[sub_idx],
            yerr=yerr_rep,
            fmt="none",
            ecolor=color,
            capsize=2.5,
            alpha=0.85,
            zorder=3,
        )

        ax.plot(
            t_vals,
            mean_v,
            marker=marker,
            linestyle="-",
            markersize=3.5,
            label=f"{d_um:.2f} µm ($N={n_part}$)",
            color=color,
            zorder=4,
        )

    guide_color = "white" if is_dark() else "#333333"
    tg1, tg2 = 3.0, 15.0
    ax.plot([tg1, tg2], [0.06 * (tg1 / 3.0) ** 1.0, 0.06 * (tg2 / 3.0) ** 1.0],
            color=guide_color, linestyle="--")
    ax.text(8.5, 0.09, r"$\propto \tilde{t}^{1.0}$", color=guide_color)
    tg3, tg4 = 0.5, 2.5
    ax.plot([tg3, tg4], [0.30 * (tg3 / 0.5) ** 2.0, 0.30 * (tg4 / 0.5) ** 2.0],
            color=guide_color, linestyle="--")
    ax.text(0.8, 2.2, r"$\propto \tilde{t}^{2.0}$", color=guide_color)

    ax.legend(loc="upper left", framealpha=0.88, markerscale=0.8, handlelength=1.4)
    ax.set(
        xlim=(0.06, 60.0),
        ylim=(1e-4, 3e2),
        xscale="log",
        yscale="log",
        xlabel=r"Dimensionless lag time $\tilde{t} \equiv u_\infty t / \xi$",
        ylabel=r"$\widetilde{\mathrm{MSD}}(\tilde{t}) \equiv \mathrm{MSD}(t) / \xi^2$",
    )
