#!/usr/bin/env python3
"""
plot_edge_slowdown_tracer_vs_cargo.py

実験の貨物と仮想点トレーサー（受動的に移流させ、同じ半径 R_c の仮想マスクを付けたもの）について、
縁での流速低下 |u|(r)/u_inf を同じ図で比べる。

物理的な狙い:
  貨物まわりで流れが遅いのは、(i) 貨物があることで流れが乱される「排除・抗力効果」なのか、
  (ii) 受動的に流される物体なら誰でも遅い領域に集まる「サンプリング効果」なのかを切り分ける。
  仮想トレーサーは流れに影響を与えないため、トレーサーでも同じ低下が出るなら (ii)、
  実験だけで低下が大きいなら (i) が効いていると言える。

入力:
  scratch/cargo_flow_profile_extended_cache.pkl            (calc_and_plot_cargo_flow_profile.py)
  scratch/virtual_tracer_flow_profile_cache_uinf_frame.pkl (scratch/simulate_virtual_tracer_flow_profile.py)
  figure/bg_angular_correlation/bg_angular_correlation_length_per_experiment.csv (動画ごとの xi)
規格化はどちらもフレームごとの u_inf(t)（r >= 20 um の遠方プラトー）。
出力: figure/edge_slowdown_tracer_vs_cargo/
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CUR = Path(__file__).resolve().parent
style_path = CUR / 'libs' / 'my_style.mplstyle'
if style_path.exists():
    plt.style.use(str(style_path))

OUT = CUR / 'figure' / 'edge_slowdown_tracer_vs_cargo'
OUT.mkdir(parents=True, exist_ok=True)

BEADS = ['beads06um', 'beads1um', 'beads3um', 'beads5um', 'beads7um', 'beads20um']
COLORS = ['#882255', '#CC6677', '#DDCC77', '#117733', '#44AA99', '#332288']
NEAR_RXI = 0.5      # 縁近傍平均の範囲 r/xi < 0.5
GRID = np.linspace(0, 3.5, 141)


def load(p):
    with open(p, 'rb') as f:
        return pickle.load(f)


def main():
    exp = load(CUR / 'scratch' / 'cargo_flow_profile_extended_cache.pkl')
    trc = load(CUR / 'scratch' / 'virtual_tracer_flow_profile_cache_uinf_frame.pkl')
    xi_df = pd.read_csv(CUR / 'figure' / 'bg_angular_correlation' / 'bg_angular_correlation_length_per_experiment.csv')
    xi_map = dict(zip(xi_df['exp_dir'].astype(str).str.strip(), xi_df['xi_um']))

    rows, prof = [], {}
    for src, df in [('cargo', exp), ('tracer', trc)]:
        for _, r in df.iterrows():
            name = Path(r['exp_dir']).name
            xi = xi_map.get(name, np.nan)
            ns = np.asarray(r['norm_speed_frame'], float)
            r_um = np.asarray(r['r_edge_um'], float)
            if not np.isfinite(xi) or not np.isfinite(ns[0]):
                continue
            rx = r_um / xi
            near = rx < NEAR_RXI
            rows.append(dict(source=src, bead_name=r['bead_name'], exp_dir=name, xi_um=xi,
                             rc_over_xi=r['radius_um'] / xi,
                             edge_norm=ns[0], edge_slowdown=1 - ns[0],
                             near_norm=np.nanmean(ns[near]), near_slowdown=1 - np.nanmean(ns[near])))
            prof.setdefault((src, r['bead_name']), []).append(np.interp(GRID, rx, ns, right=np.nan))
    per = pd.DataFrame(rows)
    per.to_csv(OUT / 'edge_slowdown_per_video.csv', index=False)

    # サイズごとの平均 ± SEM
    def sem(x):
        return np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
    agg = per.groupby(['source', 'bead_name']).agg(
        n=('edge_norm', 'size'), rc_over_xi=('rc_over_xi', 'mean'),
        edge_slowdown=('edge_slowdown', 'mean'), edge_slowdown_sem=('edge_slowdown', sem),
        near_slowdown=('near_slowdown', 'mean'), near_slowdown_sem=('near_slowdown', sem)).reset_index()
    agg.to_csv(OUT / 'edge_slowdown_per_size.csv', index=False)

    # ---------- 図 ----------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    for b, c in zip(BEADS, COLORS):
        for src, ls in [('cargo', '-'), ('tracer', '--')]:
            m = prof.get((src, b))
            if not m:
                continue
            ax.plot(GRID, np.nanmean(np.vstack(m), axis=0), ls=ls, color=c, lw=1.8,
                    label=b.replace('beads', '').replace('06', '0.6') if src == 'cargo' else None)
    ax.axhline(1, color='gray', lw=0.8)
    ax.plot([], [], 'k-', label='cargo (exp.)')
    ax.plot([], [], 'k--', label='virtual tracer')
    ax.set_xlabel(r'$r/\xi$ (distance from edge)')
    ax.set_ylabel(r'$|\mathbf{u}|(r)\,/\,u_\infty(t)$')
    ax.set_xlim(0, 3.5)
    ax.legend(fontsize=8, ncol=2, frameon=False)
    ax.set_title('(a) radial flow-speed profile')

    ax = axes[1]
    for b, c in zip(BEADS, COLORS):
        for src, mk, fc in [('cargo', 'o', c), ('tracer', 's', 'none')]:
            p = per[(per.source == src) & (per.bead_name == b)]
            ax.scatter(p.rc_over_xi, p.edge_slowdown, marker=mk, s=18, facecolors=fc, edgecolors=c, alpha=0.45, lw=0.8)
            a = agg[(agg.source == src) & (agg.bead_name == b)]
            if len(a):
                ax.errorbar(a.rc_over_xi, a.edge_slowdown, yerr=a.edge_slowdown_sem, fmt=mk, ms=8,
                            mfc=fc, mec=c, ecolor=c, capsize=3, lw=1.2)
    ax.axhline(0, color='gray', lw=0.8)
    ax.scatter([], [], marker='o', c='k', label='cargo (exp.)')
    ax.scatter([], [], marker='s', facecolors='none', edgecolors='k', label='virtual tracer (same $R_c$ mask)')
    ax.set_xscale('log')
    ax.set_xlabel(r'$R_c/\xi$')
    ax.set_ylabel(r'edge slowdown $1-|\mathbf{u}|(r\!=\!0)/u_\infty$')
    ax.legend(fontsize=8, frameon=False)
    ax.set_title('(b) edge slowdown: cargo vs tracer')
    fig.tight_layout()
    for ext in ('png', 'svg'):
        fig.savefig(OUT / f'edge_slowdown_tracer_vs_cargo.{ext}', dpi=200)

    # 数値要約
    w = agg.pivot(index='bead_name', columns='source', values=['edge_slowdown', 'edge_slowdown_sem', 'near_slowdown']).reindex(BEADS)
    with open(OUT / 'edge_slowdown_stats.txt', 'w') as f:
        f.write(w.round(3).to_string() + '\n')
    print(w.round(3).to_string())


if __name__ == '__main__':
    main()
