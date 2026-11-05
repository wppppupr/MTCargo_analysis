#!/usr/bin/env python3
"""
結合効率 η と R_c/ξ の関係: 「平均化 (足印内の不均一さ) が η を下げるか」の検証。

入力: scratch/df_inst_cache.pkl (verify_cos_decomposition.py が生成する瞬時レコード)
  - Phi_mag     : 足印 (ビーズ直下窓, ビーズ像はくり抜き済み tophat) 内の流れ方向の局所極性秩序 Φ_t
                  (local_polar_w.zarr 由来, 大きさを使わず方向のみ)
  - v_flow_norm : |<u>|_Rc  (足印平均流速の大きさ)
  - dot_product : v · <u>
定義:
  v_par = v·<u>/|<u>|,  H_t = 1 - Φ_t  (|<u>| を含まない不均一さ指標)
  η_bin = Σ v_par / Σ |<u>|   (比の平均ではなく平均の比)
  誤差: ビーズ (exp_dir, particle) 単位のブートストラップ

追加の分離検定 (サイズ効果 vs 平均化効果):
  v_par = |<u>| (β0 + β_H H + β_R log10(R_c/ξ))   を Σ 重みの最小二乗で当てはめ、
  H を入れたとき β_R が消えるか (=平均化で説明できるか) をビーズブートストラップで評価。

出力: figure/eta_heterogeneity/*.png|svg, figure/eta_heterogeneity/*.csv
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CUR = Path(__file__).parent.resolve()
style_path = CUR / 'libs' / 'my_style.mplstyle'
if style_path.exists():
    plt.style.use(str(style_path))
    COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']
else:
    COLORS = ['#882255', '#CC6677', '#DDCC77', '#117733', '#44AA99', '#332288']

ORDER = ['beads06um', 'beads1um', 'beads3um', 'beads5um', 'beads7um', 'beads20um']
MARKERS = ['^', 'o', 'd', 'p', 'h', 's']
OUT = CUR / 'figure' / 'eta_heterogeneity'
N_BOOT = 500
MIN_FRAMES, MIN_BEADS = 40, 2
RNG = np.random.default_rng(0)


def load() -> pd.DataFrame:
    d = pd.read_pickle(CUR / 'scratch' / 'df_inst_cache.pkl')
    d = d[(d['v_flow_norm'] > 0) & d['Phi_mag'].notna()].copy()
    d['U'] = d['v_flow_norm'].astype(float)
    d['v_par'] = d['dot_product'].astype(float) / d['U']
    d['H'] = 1.0 - d['Phi_mag'].clip(0, 1)
    d['bead'] = d['exp_dir'].astype(str) + ':' + d['particle'].astype(str)
    d['log_rcxi'] = np.log10(d['rc_over_xi'].astype(float))
    return d


def ratio_boot(g: pd.DataFrame):
    """η = Σv_par/ΣU とビーズブートストラップ 68% CI (±1σ 相当)。"""
    s = g.groupby('bead').agg(v=('v_par', 'sum'), u=('U', 'sum'))
    eta = s['v'].sum() / s['u'].sum()
    if len(s) < 2:
        return eta, np.nan, np.nan
    idx = RNG.integers(0, len(s), size=(N_BOOT, len(s)))
    vb, ub = s['v'].to_numpy()[idx].sum(1), s['u'].to_numpy()[idx].sum(1)
    bs = vb / ub
    return eta, *np.percentile(bs, [16, 84])


def binned_eta(d: pd.DataFrame, edges: np.ndarray) -> pd.DataFrame:
    rows = []
    d = d.assign(hbin=pd.cut(d['H'], edges, include_lowest=True, labels=False))
    for (b, k), g in d.groupby(['bead_name', 'hbin'], observed=True):
        if len(g) < MIN_FRAMES or g['bead'].nunique() < MIN_BEADS:
            continue
        eta, lo, hi = ratio_boot(g)
        rows.append(dict(bead_name=b, hbin=int(k), H_mean=g['H'].mean(), n_frames=len(g),
                         n_beads=g['bead'].nunique(), eta=eta, eta_lo=lo, eta_hi=hi,
                         rc_over_xi=g['rc_over_xi'].median()))
    return pd.DataFrame(rows)


def wls_fit(d: pd.DataFrame, cols):
    """v_par = U*(β0 + Σβ_c x_c) の最小二乗 (比の和に相当する重み)。"""
    X = np.column_stack([d['U']] + [d['U'] * d[c] for c in cols])
    beta, *_ = np.linalg.lstsq(X, d['v_par'].to_numpy(), rcond=None)
    return beta


def boot_fit(d: pd.DataFrame, cols):
    beads = d['bead'].unique()
    groups = {b: g for b, g in d.groupby('bead')}
    est = wls_fit(d, cols)
    bs = []
    for _ in range(N_BOOT):
        pick = RNG.choice(beads, size=len(beads), replace=True)
        bs.append(wls_fit(pd.concat([groups[b] for b in pick]), cols))
    bs = np.array(bs)
    return est, np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = load()
    print(f'frames used: {len(d)}, beads: {d["bead"].nunique()}')

    # --- 1. H で条件づけた η (サイズ別, 共通ビン = プール分位点) ---
    edges = np.unique(np.quantile(d['H'], np.linspace(0, 1, 9)))
    edges[0], edges[-1] = 0.0, 1.0
    eb = binned_eta(d, edges)
    # H→0 ベースライン η0: 全サイズをプールした最下位ビン
    g0 = d[d['H'] <= edges[1]]
    eta0, eta0_lo, eta0_hi = ratio_boot(g0)
    eb['eta_norm'] = eb['eta'] / eta0
    eb.to_csv(OUT / 'eta_vs_H_binned.csv', index=False)

    # --- 2. サイズ別: 全体 η, 平均 H, H を揃えた η (共通 H 範囲) ---
    rows = []
    for b in ORDER:
        g = d[d['bead_name'] == b]
        if g.empty:
            continue
        e, lo, hi = ratio_boot(g)
        rows.append(dict(bead_name=b, rc_over_xi=g['rc_over_xi'].median(), n_beads=g['bead'].nunique(),
                         n_frames=len(g), H_mean=g['H'].mean(), eta=e, eta_lo=lo, eta_hi=hi))
    sz = pd.DataFrame(rows)
    # H 分布の差を除いた η: 各サイズを全体の H 分布に再重み付け (ビン毎の η をプール H 分布で平均)
    pool_w = pd.cut(d['H'], edges, include_lowest=True, labels=False).value_counts(normalize=True)
    adj = []
    for b in sz['bead_name']:
        e = eb[eb['bead_name'] == b].set_index('hbin')['eta']
        w = pool_w.reindex(e.index)
        adj.append(float((e * w).sum() / w.sum()) if len(e) >= 3 else np.nan)
    sz['eta_H_standardized'] = adj
    sz.to_csv(OUT / 'eta_vs_rcxi_by_size.csv', index=False)

    # --- 3. 分離検定 ---
    fits = {}
    for name, cols in [('R only', ['log_rcxi']), ('H only', ['H']), ('H + R', ['H', 'log_rcxi'])]:
        est, lo, hi = boot_fit(d, cols)
        fits[name] = (cols, est, lo, hi)
    with open(OUT / 'eta_regression.txt', 'w') as f:
        f.write(f'eta0 (pooled H <= {edges[1]:.3f}) = {eta0:.3f} [{eta0_lo:.3f}, {eta0_hi:.3f}] (68% CI)\n')
        f.write('model: v_par = U*(b0 + sum b_c x_c); 95% bead-bootstrap CI\n')
        for name, (cols, est, lo, hi) in fits.items():
            terms = ', '.join(f'{c}={est[i+1]:+.3f} [{lo[i+1]:+.3f},{hi[i+1]:+.3f}]' for i, c in enumerate(cols))
            f.write(f'{name:7s}: b0={est[0]:+.3f} [{lo[0]:+.3f},{hi[0]:+.3f}]; {terms}\n')
    print((OUT / 'eta_regression.txt').read_text())
    print(sz.to_string(index=False))

    # --- 図 ---
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.0))
    for i, b in enumerate(ORDER):
        s = eb[eb['bead_name'] == b]
        if s.empty:
            continue
        rcxi = s['rc_over_xi'].iloc[0]
        axs[0].errorbar(s['H_mean'], s['eta_norm'],
                        yerr=[(s['eta'] - s['eta_lo']) / eta0, (s['eta_hi'] - s['eta']) / eta0],
                        marker=MARKERS[i], color=COLORS[i % len(COLORS)], ms=6, lw=1.5, capsize=2,
                        label=f'{b.replace("beads", "")} ($R_c/\\xi$={rcxi:.2f})')
    axs[0].axhline(1, color='0.6', lw=1, ls='--')
    axs[0].set_xscale('symlog', linthresh=0.01)
    axs[0].set_xlim(0, 1)
    axs[0].set_xlabel(r'footprint heterogeneity $H_t = 1-\Phi_t$')
    axs[0].set_ylabel(r'$\eta/\eta_0$  ($\eta=\Sigma v_\parallel/\Sigma|\langle u\rangle|$)')
    axs[0].set_title('(a) η conditioned on $H_t$', loc='left')
    axs[0].legend(fontsize=7, frameon=False)

    for i, r in sz.iterrows():
        c = COLORS[ORDER.index(r['bead_name']) % len(COLORS)]
        m = MARKERS[ORDER.index(r['bead_name'])]
        axs[1].errorbar(r['rc_over_xi'], r['eta'] / eta0,
                        yerr=[[(r['eta'] - r['eta_lo']) / eta0], [(r['eta_hi'] - r['eta']) / eta0]],
                        marker=m, color=c, ms=7, capsize=2)
        axs[1].plot(r['rc_over_xi'], r['eta_H_standardized'] / eta0, marker=m, mfc='none', mec=c, ms=9, ls='none')
        axs[2].plot(r['rc_over_xi'], r['H_mean'], marker=m, color=c, ms=7, ls='none')
    axs[1].plot([], [], 'ko', label='raw η')
    axs[1].plot([], [], 'ko', mfc='none', label='H-standardized η')
    axs[1].legend(fontsize=8, frameon=False)
    axs[1].set_xscale('log')
    axs[1].set_xlabel(r'$R_c/\xi$')
    axs[1].set_ylabel(r'$\eta/\eta_0$')
    axs[1].set_title('(b) η vs $R_c/\\xi$', loc='left')
    axs[2].set_xscale('log')
    axs[2].set_xlabel(r'$R_c/\xi$')
    axs[2].set_ylabel(r'$\langle H_t\rangle$')
    axs[2].set_title('(c) footprint heterogeneity', loc='left')
    fig.tight_layout()
    for ext in ('png', 'svg'):
        fig.savefig(OUT / f'eta_vs_heterogeneity.{ext}', dpi=300)
    print(f'saved: {OUT}/eta_vs_heterogeneity.png')


if __name__ == '__main__':
    sys.exit(main())
