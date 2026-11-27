from pathlib import Path
import numpy as np
import pandas as pd
from style import METRIC_PALETTE

ROOT = Path(__file__).resolve().parent.parent.parent
CSV_CANDIDATES = [
    ROOT / "figure" / "global_order" / "global_nematic_polar_order_summary.csv",
    ROOT / "figure" / "global_nematic_polar_order_summary.csv",
    ROOT / "global_nematic_polar_order_summary.csv",
    Path("/Volumes/data/Sasaki/MTsingleBeads/figure/global_order/global_nematic_polar_order_summary.csv"),
]


def draw(fig):
    """Draw dual-axis plot of global polar order and nematic order vs cargo diameter."""
    ax_p = fig.add_subplot(111)
    ax_n = ax_p.twinx()

    csv_path = next((p for p in CSV_CANDIDATES if p.exists()), None)
    if csv_path is None:
        ax_p.text(0.5, 0.5, "Missing:\nglobal_nematic_polar_order_summary.csv",
                  ha="center", va="center", transform=ax_p.transAxes)
        ax_p.set_axis_off()
        ax_n.set_axis_off()
        return

    df = pd.read_csv(csv_path).sort_values("diameter_um")
    d_vals = df["diameter_um"].to_numpy(dtype=float)

    p_mean = df["polar_mean"].to_numpy(dtype=float)
    p_sem = df["polar_std"].to_numpy(dtype=float) / np.sqrt(np.maximum(df["n_pol"].to_numpy(dtype=float), 1.0))

    n_mean = df["nematic_mean"].to_numpy(dtype=float)
    n_sem = df["nematic_std"].to_numpy(dtype=float) / np.sqrt(np.maximum(df["n_nem"].to_numpy(dtype=float), 1.0))

    color_p = METRIC_PALETTE[0]
    color_n = METRIC_PALETTE[1]

    exp_path = csv_path.parent / "global_nematic_polar_order_experiments.csv"
    if exp_path.exists():
        df_exp = pd.read_csv(exp_path)
        ax_p.scatter(df_exp["diameter_um"], df_exp["polar_order"], marker="o", s=12, color=color_p, alpha=0.25, edgecolors="none", zorder=2)
        ax_n.scatter(df_exp["diameter_um"], df_exp["nematic_order"], marker="s", s=12, color=color_n, alpha=0.25, edgecolors="none", zorder=2)

    l1 = ax_p.errorbar(d_vals, p_mean, yerr=p_sem, marker="o", linestyle="-", markersize=3.5,
                       color=color_p, ecolor=color_p, capsize=2.2,
                       label=r"Global Polar Order $\Phi_{\mathrm{global}}$", zorder=4)
    l2 = ax_n.errorbar(d_vals, n_mean, yerr=n_sem, marker="s", linestyle="--", markersize=3.5,
                       color=color_n, ecolor=color_n, capsize=2.2,
                       label=r"Global Nematic Order $S_{\mathrm{global}}$", zorder=4)

    ax_p.set_xlabel(r"Cargo Diameter $2R_c$ [$\mu\mathrm{m}$]")
    ax_p.set_ylabel(r"Global Polar Order $\Phi_{\mathrm{global}}$", color=color_p)
    ax_p.tick_params(axis="y", labelcolor=color_p)
    ax_p.set_ylim(0.0, 1.05)

    ax_n.set_ylabel(r"Global Nematic Order $S_{\mathrm{global}}$", color=color_n)
    ax_n.tick_params(axis="y", labelcolor=color_n)
    ax_n.set_ylim(0.0, 1.05)

    #ax_p.grid(True, linestyle="--", alpha=0.35)
    lines = [l1, l2]
    #ax_p.legend(lines, [l.get_label() for l in lines], loc="center right", framealpha=0.88, markerscale=0.85, handlelength=1.4)
