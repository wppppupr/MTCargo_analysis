from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker

try:
    from fignimble import is_dark
except ImportError:
    is_dark = lambda: False

ROOT = Path(__file__).resolve().parent.parent.parent
CSV_PATH = ROOT / "figure" / "msd" / "msd300_and_slope_summary.csv"


def draw(fig):
    """Draw dual-axis plot of MSD(300s) and late-stage slope alpha vs cargo diameter."""
    ax_msd = fig.add_subplot(111)
    ax_slope = ax_msd.twinx()

    df = pd.read_csv(CSV_PATH).sort_values("diameter_um")
    d_vals = df["diameter_um"].to_numpy(dtype=float)

    msd300_mean = df["msd_300s_mean_um2"].to_numpy(dtype=float)
    if "msd_300s_boot_ci_low_um2" in df.columns:
        msd300_ci_low = df["msd_300s_boot_ci_low_um2"].to_numpy(dtype=float)
        msd300_ci_high = df["msd_300s_boot_ci_high_um2"].to_numpy(dtype=float)
        msd300_yerr = [
            np.clip(msd300_mean - msd300_ci_low, 0, None),
            np.clip(msd300_ci_high - msd300_mean, 0, None),
        ]
    else:
        msd300_yerr = df["msd_300s_sem_um2"].to_numpy(dtype=float)

    alpha_mean = df["alpha_300_1000s_mean"].to_numpy(dtype=float)
    if "alpha_300_1000s_boot_ci_low" in df.columns:
        alpha_ci_low = df["alpha_300_1000s_boot_ci_low"].to_numpy(dtype=float)
        alpha_ci_high = df["alpha_300_1000s_boot_ci_high"].to_numpy(dtype=float)
        alpha_yerr = [
            np.clip(alpha_mean - alpha_ci_low, 0, None),
            np.clip(alpha_ci_high - alpha_mean, 0, None),
        ]
    else:
        alpha_yerr = df["alpha_300_1000s_sem"].to_numpy(dtype=float)
    n_long_vals = df["alpha_300_1000s_n_long_particles"].to_numpy(dtype=int)

    color_msd = "#1f77b4"
    line1 = ax_msd.errorbar(
        d_vals,
        msd300_mean,
        yerr=msd300_yerr,
        fmt="o-",
        color=color_msd,
        ecolor=color_msd,
        capsize=3.0,
        markersize=4.0,
        label=r"$\mathrm{MSD}(\Delta t = 300\,\mathrm{s})$",
        zorder=4,
    )

    color_slope = "#d62728"
    line2 = ax_slope.errorbar(
        d_vals,
        alpha_mean,
        yerr=alpha_yerr,
        fmt="s--",
        color=color_slope,
        ecolor=color_slope,
        capsize=3.0,
        markersize=3.8,
        label=r"Slope $\alpha$ ($\Delta t \in [300, 1000]\,\mathrm{s}$)",
        zorder=5,
    )

    bg_color = "black" if is_dark() else "white"
    for x_d, y_a, n_l in zip(d_vals, alpha_mean, n_long_vals):
        if np.isfinite(y_a):
            ax_slope.annotate(
                f"$N={n_l}$",
                (x_d, y_a),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                color=color_slope,
                bbox=dict(boxstyle="round,pad=0.15", facecolor=bg_color, edgecolor=color_slope, alpha=0.85),
                zorder=6,
            )

    ax_slope.axhline(1.0, color="#999999", linestyle=":", zorder=2)
    ax_slope.axhline(2.0, color="#999999", linestyle=":", zorder=2)

    ax_msd.set_xscale("log")
    ax_msd.set_xlim(0.4, 28.0)
    ax_msd.xaxis.set_major_locator(ticker.FixedLocator([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0]))
    ax_msd.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax_msd.set_xlabel(r"Cargo Diameter $2R_c$ [$\mu\mathrm{m}$]")

    ax_msd.set_yscale("log")
    ax_msd.set_ylim(0.05, 5000.0)
    ax_msd.set_ylabel(r"MSD $\langle \Delta r^2(\Delta t = 300\,\mathrm{s}) \rangle$ [$\mu\mathrm{m}^2$]", color=color_msd)
    ax_msd.tick_params(axis="y", labelcolor=color_msd)

    ax_slope.set_ylim(0.0, 2.3)
    ax_slope.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax_slope.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax_slope.set_ylabel(r"MSD Slope $\alpha$ ($\Delta t \in [300, 1000]\,\mathrm{s}$)", color=color_slope)
    ax_slope.tick_params(axis="y", labelcolor=color_slope)

    ax_msd.grid(True, which="both", linestyle="--", alpha=0.35)

    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax_msd.legend(lines, labels, loc="upper right", framealpha=0.92)
