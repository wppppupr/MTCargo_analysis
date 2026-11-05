#!/usr/bin/env python3
"""
結合効率 η と R_c/ξ の関係を関数形で評価する。

入力: figure/eta_heterogeneity/eta_vs_rcxi_by_size.csv  (eta_vs_heterogeneity.py の出力)
  η = Σv_par / Σ|<u>|_Rc (サイズ別), 68% ビーズブートストラップ CI

問い:
  (1) η(x), x = R_c/ξ はどの関数形で記述されるか (クロスオーバー x0 ≈ 1 か?)
  (2) スケーリング変数として R_c/ξ は R_c (ビーズ半径そのもの) より良いか?
候補 (η0, x0, n は自由パラメータ):
  power   : η = η0 x^-α
  exp     : η = η0 exp(-x/x0)
  hill    : η = η0 / (1 + (x/x0)^n)     (x0 で結合→脱結合へのクロスオーバー)
  domain  : η = η0 / sqrt(1 + (x/x0)^2) (足印内の独立ドメイン数 N~1+(x/x0)^2 の 1/√N 平均化)
重み付き最小二乗 (σ = CI 半幅), AICc で比較。20 µm を除いた頑健性も確認。
出力: figure/eta_heterogeneity/eta_rcxi_model_fit.{png,svg,txt}
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

CUR = Path(__file__).parent.resolve()
OUT = CUR / 'figure' / 'eta_heterogeneity'
style_path = CUR / 'libs' / 'my_style.mplstyle'
if style_path.exists():
    plt.style.use(str(style_path))
COLORS = ['#882255', '#CC6677', '#DDCC77', '#117733', '#44AA99', '#332288']
ORDER = ['beads06um', 'beads1um', 'beads3um', 'beads5um', 'beads7um', 'beads20um']
MARKERS = ['^', 'o', 'd', 'p', 'h', 's']
DIAM_UM = {'beads06um': 0.6, 'beads1um': 1.0, 'beads3um': 3.0, 'beads5um': 5.0, 'beads7um': 7.0, 'beads20um': 20.0}

MODELS = {
    'power':  (lambda x, a, al: a * x ** (-al), [0.2, 0.5]),
    'exp':    (lambda x, a, x0: a * np.exp(-x / x0), [0.5, 1.0]),
    'hill':   (lambda x, a, x0, n: a / (1 + (x / x0) ** n), [0.5, 0.5, 1.0]),
    'domain': (lambda x, a, x0: a / np.sqrt(1 + (x / x0) ** 2), [0.5, 0.3]),
}


def fit_all(x, y, s):
    res = {}
    n = len(x)
    for name, (f, p0) in MODELS.items():
        k = len(p0)
        try:
            p, cov = curve_fit(f, x, y, p0=p0, sigma=s, absolute_sigma=True, maxfev=20000,
                               bounds=(0, np.inf))
        except RuntimeError:
            continue
        chi2 = float(np.sum(((y - f(x, *p)) / s) ** 2))
        aicc = chi2 + 2 * k + (2 * k * (k + 1) / (n - k - 1) if n - k - 1 > 0 else np.inf)
        res[name] = dict(p=p, perr=np.sqrt(np.diag(cov)), chi2=chi2, aicc=aicc, k=k)
    return res


def report(tag, res, n):
    lines = [f'[{tag}] n={n}']
    best = min(r['aicc'] for r in res.values())
    for name, r in sorted(res.items(), key=lambda kv: kv[1]['aicc']):
        ps = ', '.join(f'{v:.3g}±{e:.2g}' for v, e in zip(r['p'], r['perr']))
        lines.append(f'  {name:7s} chi2/dof={r["chi2"] / max(n - r["k"], 1):6.2f}  '
                     f'AICc={r["aicc"]:7.2f} (Δ={r["aicc"] - best:5.2f})  params=[{ps}]')
    return lines


def main():
    sz = pd.read_csv(OUT / 'eta_vs_rcxi_by_size.csv')
    sz['sigma'] = (sz['eta_hi'] - sz['eta_lo']) / 2
    sz['Rc_um'] = sz['bead_name'].map(DIAM_UM) / 2
    sz['xi_um'] = sz['Rc_um'] / sz['rc_over_xi']

    lines = ['eta(x) model comparison; weighted LS (sigma = 68% CI half width)',
             sz[['bead_name', 'Rc_um', 'xi_um', 'rc_over_xi', 'eta', 'sigma']].round(3).to_string(index=False), '']
    fits = {}
    for var in ('rc_over_xi', 'Rc_um'):
        for tag, sub in (('all', sz), ('no20um', sz[sz['bead_name'] != 'beads20um'])):
            r = fit_all(sub[var].to_numpy(), sub['eta'].to_numpy(), sub['sigma'].to_numpy())
            fits[(var, tag)] = r
            lines += report(f'x={var}, {tag}', r, len(sub))
    # Spearman 順位相関 (関数形に依らない単調性)
    from scipy.stats import spearmanr
    for var in ('rc_over_xi', 'Rc_um'):
        rho, p = spearmanr(sz[var], sz['eta'])
        lines.append(f'Spearman eta vs {var}: rho={rho:+.3f}, p={p:.3f}')
    txt = '\n'.join(lines)
    (OUT / 'eta_rcxi_model_fit.txt').write_text(txt + '\n')
    print(txt)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, var, xl in ((axs[0], 'rc_over_xi', r'$R_c/\xi$'), (axs[1], 'Rc_um', r'$R_c$ [µm]')):
        for _, r in sz.iterrows():
            i = ORDER.index(r['bead_name'])
            ax.errorbar(r[var], r['eta'], yerr=[[r['eta'] - r['eta_lo']], [r['eta_hi'] - r['eta']]],
                        marker=MARKERS[i], color=COLORS[i], ms=7, capsize=2, ls='none',
                        label=r['bead_name'].replace('beads', ''))
        xx = np.logspace(np.log10(sz[var].min() * 0.7), np.log10(sz[var].max() * 1.4), 200)
        for j, (name, r) in enumerate(sorted(fits[(var, 'all')].items(), key=lambda kv: kv[1]['aicc'])):
            ax.plot(xx, MODELS[name][0](xx, *r['p']), color=f'{0.15 + 0.2 * j:.2f}',
                    ls=['-', '--', ':', '-.'][j], lw=1.3, label=f'{name} (ΔAICc={r["aicc"] - min(v["aicc"] for v in fits[(var, "all")].values()):.1f})')
        ax.set_xscale('log')
        ax.set_xlabel(xl)
        ax.set_ylabel(r'$\eta=\Sigma v_\parallel/\Sigma|\langle u\rangle|$')
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7, frameon=False)
    axs[0].axvline(1, color='0.7', lw=0.8)
    axs[0].set_title(r'(a) $\eta$ vs $R_c/\xi$', loc='left')
    axs[1].set_title(r'(b) $\eta$ vs $R_c$', loc='left')
    fig.tight_layout()
    for ext in ('png', 'svg'):
        fig.savefig(OUT / f'eta_rcxi_model_fit.{ext}', dpi=300)
    print(f'saved: {OUT}/eta_rcxi_model_fit.png')


if __name__ == '__main__':
    main()
