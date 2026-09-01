import re
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Try to use project style if available
try:
    plt.style.use('libs/my_style.mplstyle')
    style_colors = ['#332288', '#44AA99', '#CC6677', '#88CCEE', '#332288', '#AA4499']#plt.rcParams['axes.prop_cycle'].by_key()['color']
except Exception:
    style_colors = ['#882255', '#CC6677', '#DDCC77', '#999933', '#117733', '#44AA99', '#88CCEE', '#332288', '#AA4499']

# Condition to diameter (um) mapping
DIAMETER_MAP = {
    'beads06um': 0.6,
    'beads1um': 1.0,
    'beads3um': 3.0,
    'beads5um': 5.0,
    'beads7um': 7.0,
    'beads20um': 20.0,
}

def parse_diameter(cond_str):
    cond = str(cond_str).strip()
    if cond in DIAMETER_MAP:
        return DIAMETER_MAP[cond]
    # Regex fallback (e.g., beads06um -> 0.6, beads1um -> 1.0)
    m = re.search(r'beads(\d+)(?:_?(\d+))?um', cond)
    if m:
        val_str = m.group(1)
        if val_str.startswith('0') and len(val_str) > 1:
            val_str = f"0.{val_str[1:]}"
        return float(val_str)
    return np.nan

def load_xi_csv(csv_path):
    # Skip first line if it contains the header text
    with open(csv_path, 'r') as f:
        first_line = f.readline()
    
    skiprows = 1 if '---' in first_line else 0
    df = pd.read_csv(csv_path, skiprows=skiprows)
    
    # Strip whitespaces from column names and string cells
    df.columns = df.columns.str.strip()
    for col in df.columns:
        if df[col].dtype == object or isinstance(df[col].dtype, pd.StringDtype):
            df[col] = df[col].astype(str).str.strip()
    
    # Map diameter
    df['diameter_um'] = df['condition'].map(parse_diameter)
    return df

def main():
    parser = argparse.ArgumentParser(description="Plot correlation length xi vs bead diameter from xi.csv.")
    parser.add_argument('--root_dir', type = str, default= '/Volumes/data/Sasaki/MTsingleBeads')
    parser.add_argument('--csv', type=str, default='xi.csv', help='Path to xi.csv')
    parser.add_argument('--xscale', type=str, default='log', choices=['linear', 'log'], help='X-axis scale (linear or log)')
    parser.add_argument('--yscale', type=str, default='log', choices=['linear', 'log'], help='Y-axis scale (linear or log)')
    parser.add_argument('--save_fig', type=str, default='figure/xi_diameter_summary.svg', help='Output figure path')
    parser.add_argument('--error_type', type=str, default='sem', choices=['sem', 'std'], help='Error bar type (sem or std)')
    args = parser.parse_args()

    csv_path = Path(args.root_dir) / args.csv
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = load_xi_csv(csv_path)

    # Condition & marker ordering matching angular_correlation.py
    conditions = ['beads06um', 'beads1um', 'beads3um', 'beads5um', 'beads7um', 'beads20um']
    marker_list = ['o']#['^', 'o', 'd', 'p', 'h', 's']
    cond_marker_map = dict(zip(conditions, marker_list))

    # Types to plot (3 subplots)
    types_info = [
        ('flow_particle', r"$\bf{Flow\ around\ Particle}$" + "\n" + r"$\langle \hat{\mathbf{u}}_{\mathrm{flow}}(0) \cdot \hat{\mathbf{u}}_{\mathrm{flow}}(r) \rangle$"),
        ('bead_flow', r"$\bf{Bead\ Velocity\ vs\ Flow}$" + "\n" + r"$\langle \hat{\mathbf{u}}_{\mathrm{bead}} \cdot \hat{\mathbf{u}}_{\mathrm{flow}}(r) \rangle$"),
        ('background', r"$\bf{Background\ Flow}$" + "\n" + r"$\langle \hat{\mathbf{u}}_{\mathrm{bg}}(0) \cdot \hat{\mathbf{u}}_{\mathrm{bg}}(r) \rangle$")
    ]

    # Components
    components = ['total', 'par', 'perp']
    comp_colors = {
        'total': style_colors[0] if len(style_colors) > 0 else '#882255',
        'par': style_colors[1] if len(style_colors) > 1 else '#CC6677',
        'perp': style_colors[2] if len(style_colors) > 2 else '#DDCC77'
    }
    comp_labels = {
        'total': 'Total',
        'par': r'Parallel ($\parallel$)',
        'perp': r'Perpendicular ($\perp$)'
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for ax_idx, (t_key, t_title) in enumerate(types_info):
        ax = axes[ax_idx]
        df_type = df[df['type'] == t_key]

        # Calculate mean and error across components for each bead condition
        grouped = df_type.groupby(['condition', 'diameter_um'])['xi_um']
        summary = grouped.agg(
            mean='mean',
            std='std',
            sem=lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0,
            count='count'
        ).reset_index().sort_values('diameter_um')

        err_col = args.error_type

        # 1. Plot individual components (total, par, perp) with matching markers per condition
        for comp in components:
            df_comp = df_type[df_type['component'] == comp].sort_values('diameter_um')
            if not df_comp.empty:
                # Plot connected line for component
                ax.plot(
                    df_comp['diameter_um'],
                    df_comp['xi_um'],
                    linestyle='-',
                    color=comp_colors[comp],
                    alpha=0.6,
                    linewidth=1.8,
                    label=f"{comp_labels[comp]}"
                )

                # Plot condition markers
                for _, row in df_comp.iterrows():
                    cond = row['condition']
                    marker = cond_marker_map.get(cond, 'o')
                    ax.scatter(
                        row['diameter_um'],
                        row['xi_um'],
                        marker=marker,
                        color=comp_colors[comp],
                        s=100,
                        edgecolor='k',
                        linewidth=0.8,
                        zorder=4
                    )

        # 2. Plot overall mean line with error bars across components
        """ax.errorbar(
            summary['diameter_um'],
            summary['mean'],
            yerr=summary[err_col],
            color='#222222',
            linestyle='--',
            linewidth=2.0,
            capsize=5,
            capthick=1.5,
            label=f"Mean ± {args.error_type.upper()}",
            zorder=5
        )"""

        ax.set_title(t_title, fontsize=16)
        ax.set_xlabel(r'Bead Diameter $d$ [$\mu$m]')
        ax.set_xscale(args.xscale)
        ax.set_yscale(args.yscale)
        ax.set_xticks([0.6, 1, 3, 5, 7, 20])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_ylim(-1, 50)
        ax.minorticks_off()
        ax.grid(True, which='both', linestyle='--', alpha=0.5)

        if ax_idx == 0:
            ax.set_ylabel(r'Correlation Length $\xi$ [$\mu$m]')
        
        ax.legend(loc='lower right' if t_key == 'background' else 'upper left', fontsize=11, frameon=True)

    plt.tight_layout()

    out_path = Path(args.root_dir) / args.save_fig
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    print(f"[SUCCESS] Plot saved to {out_path.resolve()}")

    # Print summary table
    print("\n--- Summary Statistics across components by Diameter ---")
    agg_df = df.groupby(['type', 'condition', 'diameter_um'])['xi_um'].agg(['mean', 'std', 'sem', 'count']).reset_index()
    print(agg_df.to_string(index=False))

if __name__ == '__main__':
    main()
