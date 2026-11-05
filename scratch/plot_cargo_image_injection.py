#!/usr/bin/env python3
"""画像注入テストの結果を、実験の縁の低下・仮想トレーサーと同じ図で比べる。
(a) |u_inj|/|u_orig| vs r/xi（像による見かけの低下）、実験 |u|/u_inf と重ねる
(b) 縁の低下 (r<0.5um) : 実験 / 像だけ / トレーサー をサイズごとに比較
(c) 実験 ÷ 像の比 = 像の影響を割り戻した補正プロファイル（一次近似）
"""
import pickle
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
CUR = Path(__file__).resolve().parents[1]
sp = CUR / 'libs' / 'my_style.mplstyle'
if sp.exists(): plt.style.use(str(sp))
OUT = CUR / 'figure' / 'cargo_image_injection'
BEADS = ['beads06um', 'beads1um', 'beads3um', 'beads5um', 'beads7um', 'beads20um']
COLORS = ['#882255', '#CC6677', '#DDCC77', '#117733', '#44AA99', '#332288']
GRID = np.linspace(0, 3.5, 141)
lab = lambda b: b.replace('beads', '').replace('06', '0.6').replace('um', ' µm')
inj = pickle.load(open(CUR / 'scratch' / 'cargo_image_injection_cache.pkl', 'rb'))
exp = pickle.load(open(CUR / 'scratch' / 'cargo_flow_profile_extended_cache.pkl', 'rb'))
trc = pickle.load(open(CUR / 'scratch' / 'virtual_tracer_flow_profile_cache_uinf_frame.pkl', 'rb'))
xi = pd.read_csv(CUR / 'figure' / 'bg_angular_correlation' / 'bg_angular_correlation_length_per_experiment.csv')
xi = dict(zip(xi.exp_dir.astype(str).str.strip(), xi.xi_um))
r_mid = (np.arange(len(inj[0]['n'])) + 0.5) * 0.5   # um（縁から）
r_lo = np.arange(len(inj[0]['n'])) * 0.5
rows, P = [], {}
for a in inj:
    x = xi.get(a['exp_dir'], np.nan)
    ratio = a['s_inj'] / a['s_orig']
    err = a['s_err'] / a['s_orig']
    rows.append(dict(source='image', bead_name=a['bead_name'], exp_dir=a['exp_dir'], xi_um=x, rc_over_xi=a['radius_um'] / x,
                     edge_slowdown=1 - ratio[0], near_slowdown=1 - np.nanmean(ratio[r_lo / x < 0.5]), edge_rel_err=err[0]))
    P.setdefault(('image', a['bead_name']), []).append(np.interp(GRID, r_lo / x, ratio, right=np.nan))
for src, df in [('cargo', exp), ('tracer', trc)]:
    for _, r in df.iterrows():
        n = Path(r['exp_dir']).name; x = xi.get(n, np.nan)
        ns = np.asarray(r['norm_speed_frame'], float); ru = np.asarray(r['r_edge_um'], float)
        if not np.isfinite(x) or not np.isfinite(ns[0]): continue
        rows.append(dict(source=src, bead_name=r['bead_name'], exp_dir=n, xi_um=x, rc_over_xi=r['radius_um'] / x,
                         edge_slowdown=1 - ns[0], near_slowdown=1 - np.nanmean(ns[ru / x < 0.5])))
        P.setdefault((src, r['bead_name']), []).append(np.interp(GRID, ru / x, ns, right=np.nan))
per = pd.DataFrame(rows); per.to_csv(OUT / 'image_injection_per_video.csv', index=False)
sem = lambda v: np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
agg = per.groupby(['source', 'bead_name']).agg(n=('edge_slowdown', 'size'), rc_over_xi=('rc_over_xi', 'mean'),
      edge=('edge_slowdown', 'mean'), edge_sem=('edge_slowdown', sem), near=('near_slowdown', 'mean'), near_sem=('near_slowdown', sem)).reset_index()
agg.to_csv(OUT / 'image_injection_per_size.csv', index=False)

fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
for b, c in zip(BEADS, COLORS):
    m = lambda s: np.nanmean(np.vstack(P[(s, b)]), 0)
    ax[0].plot(GRID, m('cargo'), '-', c=c, lw=1.8, label=lab(b))
    ax[0].plot(GRID, m('image'), ':', c=c, lw=2.0)
    ax[2].plot(GRID, m('cargo') / m('image'), '-', c=c, lw=1.8, label=lab(b))
ax[0].plot([], [], 'k-', label='cargo (exp.)'); ax[0].plot([], [], 'k:', label='image only (injected)')
for a_ in (ax[0], ax[2]):
    a_.axhline(1, c='gray', lw=.8); a_.set_xlim(0, 3.5); a_.set_xlabel(r'$r/\xi$ (distance from edge)')
ax[0].set_ylabel(r'$|\mathbf{u}|/u_\infty$  or  $|\mathbf{u}_{inj}|/|\mathbf{u}_{orig}|$'); ax[0].legend(fontsize=8, ncol=2, frameon=False)
ax[0].set_title('(a) exp. profile vs image-induced drop')
ax[2].set_ylabel(r'$|\mathbf{u}|_{exp}/u_\infty \;\div\; |\mathbf{u}_{inj}|/|\mathbf{u}_{orig}|$'); ax[2].set_title('(c) exp. corrected for image artifact')
ax[2].legend(fontsize=8, ncol=2, frameon=False)
x = np.arange(len(BEADS)); w = .27
for k, (src, fc, nm) in enumerate([('cargo', 'k', 'cargo (exp.)'), ('image', '#999999', 'image only'), ('tracer', 'white', 'virtual tracer')]):
    a = agg[agg.source == src].set_index('bead_name').reindex(BEADS)
    ax[1].bar(x + (k - 1) * w, a.edge, w, yerr=a.edge_sem, color=fc, edgecolor='k', capsize=2, label=nm)
ax[1].axhline(0, c='gray', lw=.8); ax[1].set_xticks(x, [lab(b) for b in BEADS])
ax[1].set_ylabel(r'edge slowdown $1-$ratio at $r<0.5\,\mu$m'); ax[1].legend(fontsize=8, frameon=False)
ax[1].set_title('(b) edge slowdown: exp. vs image vs tracer')
fig.tight_layout()
for e in ('png', 'svg'): fig.savefig(OUT / f'cargo_image_injection.{e}', dpi=200)
w_ = agg.pivot(index='bead_name', columns='source', values=['edge', 'edge_sem', 'near']).reindex(BEADS).round(3)
open(OUT / 'image_injection_stats.txt', 'w').write(w_.to_string() + '\n'); print(w_.to_string())
im = per[per.source == 'image']; print('image edge rel. vector error |u_inj-u_orig|/|u_orig| (mean):', im.edge_rel_err.mean().round(2))
