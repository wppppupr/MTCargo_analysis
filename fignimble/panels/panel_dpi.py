from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
CSV_CANDIDATES = [
    ROOT / "figure" / "per_point_v_dot_u" /"data_processing_inequality_summary.csv",
    ROOT / "data_processing_inequality_summary.csv",
    Path("/Volumes/data/Sasaki/MTsingleBeads/figure/per_point_v_dot_u/data_processing_inequality_summary.csv")
]


def draw(fig):
    """Draw Data Processing Inequality hierarchy: D_KL(obs) <= D_KL^conv <= D_MT."""
    ax = fig.add_subplot(111)
    csv_path = next((p for p in CSV_CANDIDATES if p.exists()), None)
    if csv_path is None:
        ax.text(0.5, 0.5, "Missing:\ndata_processing_inequality_summary.csv\nRun analysis/calc_data_processing_inequality.py",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    df_dpi = pd.read_csv(csv_path).sort_values("rc_over_xi")
    ax.fill_between(
        df_dpi["rc_over_xi"],
        df_dpi["D_MT_total"],
        10.0,
        color="#ffcccc",
        alpha=0.25,
        edgecolor="none",
        label=r"Forbidden ($D_{\mathrm{KL}} > D_{\mathrm{MT}}$)",
        zorder=1,
    )
    ax.errorbar(df_dpi["rc_over_xi"], df_dpi["D_MT_total"], yerr=df_dpi["D_MT_total_sem"],
                xerr=df_dpi["rc_over_xi_sem"], fmt="o-", color="#117733", markersize=4.5,
                markeredgecolor="k", capsize=3.0, label=r"$D_{\mathrm{MT}}$ (Total MT KL)", zorder=4)
    ax.plot(df_dpi["rc_over_xi"], df_dpi["D_MT_1_zarr"], "^--", color="#44AA99", markersize=4.0,
            markeredgecolor="k", label=r"$D_{\mathrm{MT}}^{(1)}$ (from $\Phi_{\mathrm{bg}}$)", zorder=3)

    df_conv = df_dpi.iloc[1:]
    ax.plot(df_conv["rc_over_xi"], df_conv["D_KL_conv_bound"], "D-.", color="#555555", markersize=3.8,
            label=r"$D_{\mathrm{KL}}^{\mathrm{conv}}$", zorder=4)
    ax.plot(df_dpi["rc_over_xi"], df_dpi["D_KL_obs"], "s-", color="#882255", markersize=4.5,
            markeredgecolor="k", label=r"$D_{\mathrm{KL}}$ (Observed cargo)", zorder=6)

    for _, r in df_dpi.iterrows():
        ax.annotate(f"{r['diameter_um']:.2g} µm", xy=(r["rc_over_xi"], r["D_KL_obs"]),
                    xytext=(0, -10), textcoords="offset points", ha="center", color="#882255")

    ax.axvline(1.0, color="0.25", linestyle="-.", alpha=0.7, label=r"$x = 1\;(R_c = \xi)$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.04, 3.3)
    ax.set_ylim(1e-3, 6.0)
    ax.set_xlabel(r"Dimensionless Cargo Radius $x = R_c / \xi$")
    ax.set_ylabel("KL Divergence [nats]")
    #ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="lower left", framealpha=0.92)
