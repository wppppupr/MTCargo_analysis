#!/usr/bin/env python3
"""
回転 (転がり) がカーゴ輸送をどの程度下げうるかの見積もりと、実測での検定。

物理的な狙い:
  カーゴ球が底面の MT カーペットに接点 (下端) だけで駆動され、上側は流体に囲まれているとする。
  球は自由に回転できるので、接点速度 u (足印流) のうち一部は「並進 U」ではなく「後ろ向きの回転 aΩ」
  に食われる (転がり)。接点 no-slip: u = U - aΩ。力・トルク釣り合いから
      κ_roll ≡ U/u = (4/3 t_r - g) / (4/3 t_r - g + f_t - g)
  バルク (壁なし):  f_t=1, g=0, t_r=1  ->  κ_roll = 4/7 ≈ 0.57
  壁近傍 (Goldman-Cox-Brenner 1967, 隙間 δ, L=ln(a/δ)):
      f_t = 8/15 L + 0.9588,  t_r = 2/5 L + 0.3817,  g = 2/15 L + 0.2526
      L→∞ で κ_roll → 1/2。 a=0.3–10 µm, δ=20–200 nm で κ_roll ≈ 0.50–0.53 (ほぼサイズに依らない)。
  したがって「回転が主因」なら、(i) 方向が揃ったフレームでも |v|/|<u>| ≲ 0.57、(ii) サイズにほぼ依らない、
  の 2 点が観測されるはず。面内スピン (z 軸回り, Ω_z = ω_z/2) は等方球の並進と結合しないので η を直接下げない。

実測:
  scratch/df_inst_cache.pkl (足印平均流 <u>_Rc とカーゴ速度 v の瞬時レコード, dt = 4 s)
  κ (速さ伝達率) = median(|v|/|<u>|)  ただし cos ψ > COS_TH のフレーム (方向が揃ったときの速さの伝わり方)
  動画ごとに κ, <cos ψ> を求め、log R_c と log(R_c/ξ) のどちらで説明されるかを重回帰で比べる。
出力: figure/rotation_rolling/rotation_rolling_estimate.{png,svg}, *_per_video.csv, *_stats.txt
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

CUR = Path(__file__).parent.resolve()
OUT = CUR / 'figure' / 'rotation_rolling'
style_path = CUR / 'libs' / 'my_style.mplstyle'
if style_path.exists():
    plt.style.use(str(style_path))
COLORS = ['#882255', '#CC6677', '#DDCC77', '#117733', '#44AA99', '#332288']
ORDER = ['beads06um', 'beads1um', 'beads3um', 'beads5um', 'beads7um', 'beads20um']
LABEL = {'beads06um': '0.63 µm', 'beads1um': '1.18 µm', 'beads3um': '3.37 µm',
         'beads5um': '5.00 µm', 'beads7um': '7.24 µm', 'beads20um': '20.0 µm'}
MARKERS = ['^', 'o', 'd', 'p', 'h', 's']
COS_TH = 0.8
N_BOOT = 1000
RNG = np.random.default_rng(0)


def kappa_roll(a_um, delta_um=None):
    """接点駆動・自由回転の球の並進/接点速度比。delta=None はバルク (4/7)。"""
    if delta_um is None:
        return 4.0 / 7.0
    L = np.log(np.asarray(a_um) / delta_um)
    ft, tr, g = 8 / 15 * L + 0.9588, 2 / 5 * L + 0.3817, 2 / 15 * L + 0.2526
    return (4 / 3 * tr - g) / (4 / 3 * tr - g + ft - g)


def load():
    d = pd.read_pickle(CUR / 'scratch' / 'df_inst_cache.pkl')
    d = d[(d['v_flow_norm'] > 0) & (d['v_cargo_norm'] > 0) & d['cos_bulk'].notna() & d['rc_over_xi'].notna()].copy()
    d['ratio'] = d['v_cargo_norm'] / d['v_flow_norm']
    d['bead'] = d['exp_dir'].astype(str) + ':' + d['particle'].astype(str)
    return d


def bead_stats(g):
    a = g[g['cos_bulk'] > COS_TH]
    sb = a.groupby('bead')['ratio'].median()
    out = dict(n_beads=g['bead'].nunique(), n_frames=len(g), n_aligned=len(a),
               kappa=a['ratio'].median() if len(a) else np.nan, cos=g.groupby('bead')['cos_bulk'].mean().mean(),
               kappa_lo=np.nan, kappa_hi=np.nan)
    if len(sb) >= 2:
        idx = RNG.integers(0, len(sb), size=(N_BOOT, len(sb)))
        out['kappa_lo'], out['kappa_hi'] = np.percentile(np.median(sb.to_numpy()[idx], axis=1), [16, 84])
    return out


def ols(y, X):
    X = np.column_stack([np.ones(len(y))] + X)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    s2 = r @ r / (len(y) - X.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    r2 = 1 - r @ r / ((y - y.mean()) @ (y - y.mean()))
    return b, se, r2


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = load()
    vid = pd.DataFrame([dict(bead_name=b, exp_dir=e, Rc=g['radius_um'].iloc[0], rc_over_xi=g['rc_over_xi'].iloc[0], **bead_stats(g))
                        for (b, e), g in d.groupby(['bead_name', 'exp_dir'])])
    vid = vid[vid['n_aligned'] >= 20].reset_index(drop=True)
    sz = pd.DataFrame([dict(bead_name=b, Rc=d.loc[d.bead_name == b, 'radius_um'].iloc[0],
                            rc_over_xi=d.loc[d.bead_name == b, 'rc_over_xi'].median(), **bead_stats(d[d.bead_name == b]))
                       for b in ORDER])
    vid.to_csv(OUT / 'rotation_rolling_per_video.csv', index=False)

    lines = [f'kappa = median |v|/|<u>_Rc| over frames with cos psi > {COS_TH} (dt = 4 s)',
             'Rolling prediction (contact-driven free sphere): bulk 4/7 = 0.571; near-wall GCB:']
    for dl in (0.02, 0.05, 0.2):
        lines.append(f'  delta={dl*1e3:.0f} nm: ' + ', '.join(f'a={a:g}um->{kappa_roll(a, dl):.3f}' for a in (0.315, 1.685, 3.62, 10.0)))
    lines += ['', sz.round(3).to_string(index=False), '', vid.round(3).to_string(index=False), '']
    lR, lX = np.log10(vid['Rc'].to_numpy()), np.log10(vid['rc_over_xi'].to_numpy())
    for y in ('kappa', 'cos'):
        for nm, x in (('R_c', lR), ('R_c/xi', lX)):
            rho, p = spearmanr(x, vid[y])
            lines.append(f'Spearman {y:5s} vs {nm:6s} (per-video n={len(vid)}): rho={rho:+.3f} p={p:.3g}')
        yy = np.log10(vid[y].clip(lower=1e-3).to_numpy()) if y == 'kappa' else vid[y].to_numpy()
        b, se, r2 = ols(yy, [lR, lX])
        lines.append(f'  OLS {"log kappa" if y == "kappa" else "cos"} = b0 + bR log R_c + bX log(R_c/xi): '
                     f'bR={b[1]:+.3f}±{se[1]:.3f}, bX={b[2]:+.3f}±{se[2]:.3f}, R2={r2:.2f}')
    frac = (vid['kappa'] > kappa_roll(vid['Rc'], 0.05)).mean()
    lines.append(f'Fraction of videos with kappa above rolling bound (delta=50nm): {frac:.2f}')
    txt = '\n'.join(lines)
    (OUT / 'rotation_rolling_estimate_stats.txt').write_text(txt + '\n')
    print(txt)

    fig, axs = plt.subplots(1, 3, figsize=(12.0, 3.8))
    a_grid = np.logspace(np.log10(0.25), np.log10(12), 100)
    ax = axs[0]
    ax.fill_between(a_grid, kappa_roll(a_grid, 0.2), kappa_roll(a_grid, 0.02), color='0.75', alpha=0.6, lw=0,
                    label='rolling, near wall\n($\\delta$=20–200 nm)')
    ax.axhline(4 / 7, color='0.4', ls='--', lw=1, label='rolling, bulk (4/7)')
    ax.axhline(1.0, color='0.3', ls=':', lw=0.8)
    for col, (ax, xk, xl, yk, yl) in enumerate(((axs[0], 'Rc', r'$R_c$ [µm]', 'kappa', r"$\kappa=|\mathbf{v}|/|\langle\mathbf{u}\rangle_{R_c}|$ (aligned)"),
                                                 (axs[1], 'rc_over_xi', r'$R_c/\xi$', 'kappa', r'$\kappa$'),
                                                 (axs[2], 'Rc', r'$R_c$ [µm]', 'cos', r'$\langle\cos\psi\rangle$'))):
        for i, b in enumerate(ORDER):
            vv = vid[vid.bead_name == b]
            ax.scatter(vv[xk], vv[yk], marker=MARKERS[i], facecolor='none', edgecolor=COLORS[i], s=30, lw=0.9, zorder=2)
            r = sz[sz.bead_name == b].iloc[0]
            yerr = [[r[yk] - r[f'{yk}_lo']], [r[f'{yk}_hi'] - r[yk]]] if yk == 'kappa' else None
            ax.errorbar(r[xk], r[yk], yerr=yerr, marker=MARKERS[i], color=COLORS[i], mec='k', mew=0.6, ms=8,
                        capsize=2.5, ls='none', zorder=3, label=LABEL[b] if col == 1 else None)
        ax.set_xscale('log')
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(f'({"abc"[col]})', loc='left')
    axs[0].set_ylim(0, 1.25)
    axs[1].set_ylim(0, 1.25)
    axs[1].axhline(4 / 7, color='0.4', ls='--', lw=1)
    axs[2].axhline(0, color='0.3', lw=0.6)
    axs[0].legend(fontsize=7, frameon=False, loc='lower left')
    axs[1].legend(fontsize=7, frameon=False, loc='lower left', ncol=2, title=r'$2R_c$', title_fontsize=7)
    fig.tight_layout()
    for ext in ('png', 'svg'):
        fig.savefig(OUT / f'rotation_rolling_estimate.{ext}', dpi=300, bbox_inches='tight')
    print(f'saved: {OUT}/rotation_rolling_estimate.png')


if __name__ == '__main__':
    main()
