#!/usr/bin/env python3
"""
本文主図: 方向追従 cos ψ と結合効率 η を、動画ごとの ξ で無次元化した R_c/ξ に対して描く。

入力: scratch/df_inst_cache.pkl (verify_cos_decomposition.py の瞬時レコード, ビーズ直下 tophat 窓)
  - cos_bulk    : cos ψ = v·<u>_Rc / (|v| |<u>_Rc|)   (足印平均流 <u>_Rc に対する進行方向の角度 ψ)
  - dot_product : v·<u>_Rc,  v_flow_norm : |<u>_Rc|
  - rc_over_xi  : R_c / ξ_video  (ξ は動画ごとの背景配向相関長, bg_angular_correlation_length_per_experiment.csv)
定義:
  cos ψ (群)  = ビーズ平均 <cos ψ>_bead の算術平均 (ビーズを等重み)
  η (群)      = Σ v_par / Σ |<u>|,  v_par = v·<u>/|<u>|   (平均の比)
  誤差        : ビーズ (exp_dir, particle) 単位のブートストラップ 68% CI
小マーカー = 動画 1 本 (x = その動画の R_c/ξ), 大マーカー = サイズ別プール
  (x = フレーム数重みの R_c/ξ 中央値, 横棒 = 動画間の R_c/ξ 範囲)
出力: figure/main_cos_eta_vs_rcxi/main_cos_eta_vs_rcxi.{png,svg}, *_per_video.csv, *_per_size.csv, *_stats.txt
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

CUR = Path(__file__).parent.resolve()
OUT = CUR / 'figure' / 'main_cos_eta_vs_rcxi'
style_path = CUR / 'libs' / 'my_style.mplstyle'
if style_path.exists():
    plt.style.use(str(style_path))
COLORS = ['#882255', '#CC6677', '#DDCC77', '#117733', '#44AA99', '#332288']
ORDER = ['beads06um', 'beads1um', 'beads3um', 'beads5um', 'beads7um', 'beads20um']
LABEL = {'beads06um': '0.6 µm', 'beads1um': '1 µm', 'beads3um': '3 µm',
         'beads5um': '5 µm', 'beads7um': '7 µm', 'beads20um': '20 µm'}
MARKERS = ['^', 'o', 'd', 'p', 'h', 's']
N_BOOT = 1000
RNG = np.random.default_rng(0)


def load() -> pd.DataFrame:
    d = pd.read_pickle(CUR / 'scratch' / 'df_inst_cache.pkl')
    d = d[(d['v_flow_norm'] > 0) & d['cos_bulk'].notna() & d['rc_over_xi'].notna()].copy()
    d['U'] = d['v_flow_norm'].astype(float)
    d['v_par'] = d['dot_product'].astype(float) / d['U']
    d['bead'] = d['exp_dir'].astype(str) + ':' + d['particle'].astype(str)
    return d


def stats(g: pd.DataFrame) -> dict:
    """cos ψ (ビーズ等重み) と η (平均の比) + ビーズブートストラップ 68% CI。"""
    s = g.groupby('bead').agg(c=('cos_bulk', 'mean'), v=('v_par', 'sum'), u=('U', 'sum'))
    c, v, u = s['c'].to_numpy(), s['v'].to_numpy(), s['u'].to_numpy()
    out = dict(n_beads=len(s), n_frames=len(g), cos=c.mean(), eta=v.sum() / u.sum(),
               cos_lo=np.nan, cos_hi=np.nan, eta_lo=np.nan, eta_hi=np.nan)
    if len(s) >= 2:
        idx = RNG.integers(0, len(s), size=(N_BOOT, len(s)))
        out['cos_lo'], out['cos_hi'] = np.percentile(c[idx].mean(1), [16, 84])
        out['eta_lo'], out['eta_hi'] = np.percentile(v[idx].sum(1) / u[idx].sum(1), [16, 84])
    return out


def wmedian(x, w):
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return float(x[o][np.searchsorted(cw, 0.5 * cw[-1])])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = load()

    vid = []
    for (b, e), g in d.groupby(['bead_name', 'exp_dir']):
        vid.append(dict(bead_name=b, exp_dir=e, xi_um=g['xi'].iloc[0], rc_over_xi=g['rc_over_xi'].iloc[0], **stats(g)))
    vid = pd.DataFrame(vid)
    vid['bead_name'] = pd.Categorical(vid['bead_name'], ORDER, ordered=True)
    vid = vid.sort_values(['bead_name', 'rc_over_xi']).reset_index(drop=True)

    sz = []
    for b in ORDER:
        g = d[d['bead_name'] == b]
        if g.empty:
            continue
        vv = vid[vid['bead_name'] == b]
        sz.append(dict(bead_name=b, rc_over_xi=wmedian(g['rc_over_xi'].to_numpy(), np.ones(len(g))),
                       rcxi_min=vv['rc_over_xi'].min(), rcxi_max=vv['rc_over_xi'].max(),
                       n_videos=len(vv), **stats(g)))
    sz = pd.DataFrame(sz)
    vid.to_csv(OUT / 'main_cos_eta_vs_rcxi_per_video.csv', index=False)
    sz.to_csv(OUT / 'main_cos_eta_vs_rcxi_per_size.csv', index=False)

    # 単調性 (関数形に依らない): 動画単位・サイズ単位の Spearman
    lines = ['cos psi & eta vs R_c/xi (per-video xi); bead bootstrap 68% CI',
             sz.round(3).to_string(index=False), '',
             vid[['bead_name', 'exp_dir', 'xi_um', 'rc_over_xi', 'n_beads', 'n_frames', 'cos', 'eta']].round(3).to_string(index=False), '']
    for lvl, df in (('per-video', vid), ('per-size', sz)):
        for y in ('cos', 'eta'):
            rho, p = spearmanr(df['rc_over_xi'], df[y])
            lines.append(f'Spearman {y:3s} vs R_c/xi ({lvl}, n={len(df)}): rho={rho:+.3f}, p={p:.3g}')
    for y in ('cos', 'eta'):
        rho, p = spearmanr(vid.loc[vid['n_beads'] >= 2, 'rc_over_xi'], vid.loc[vid['n_beads'] >= 2, y])
        lines.append(f'Spearman {y:3s} vs R_c/xi (per-video, n_beads>=2, n={int((vid["n_beads"] >= 2).sum())}): rho={rho:+.3f}, p={p:.3g}')
    txt = '\n'.join(lines)
    (OUT / 'main_cos_eta_vs_rcxi_stats.txt').write_text(txt + '\n')
    print(txt)

    # --- 図: 横軸共有 2 段 ---
    fig, axs = plt.subplots(2, 1, figsize=(5.2, 7.0), sharex=True)
    for ax, y, yl in ((axs[0], 'cos', r'$\langle\cos\psi\rangle$'),
                      (axs[1], 'eta', r'$\eta=\Sigma v_\parallel/\Sigma|\langle\mathbf{u}\rangle_{R_c}|$')):
        for i, b in enumerate(ORDER):
            vv = vid[vid['bead_name'] == b]
            ax.scatter(vv['rc_over_xi'], vv[y], s=10 + 6 * np.sqrt(vv['n_beads']), marker=MARKERS[i],
                       facecolor='none', edgecolor=COLORS[i], lw=0.9, alpha=0.8, zorder=2)
            r = sz[sz['bead_name'] == b]
            if r.empty:
                continue
            r = r.iloc[0]
            ax.errorbar(r['rc_over_xi'], r[y],
                        xerr=[[r['rc_over_xi'] - r['rcxi_min']], [r['rcxi_max'] - r['rc_over_xi']]],
                        yerr=[[r[y] - r[f'{y}_lo']], [r[f'{y}_hi'] - r[y]]],
                        marker=MARKERS[i], color=COLORS[i], mec='k', mew=0.6, ms=9, capsize=2.5,
                        elinewidth=1.1, ls='none', zorder=3, label=LABEL[b])
        ax.axvline(1.0, color='0.6', lw=0.8, ls='--', zorder=1)
        ax.set_xscale('log')
        ax.set_ylabel(yl)
        ymin = min(vid[y].min(), sz[f'{y}_lo'].min())
        ax.set_ylim(bottom=min(0.0, ymin - 0.05))
        ax.axhline(0.0, color='0.3', lw=0.6, zorder=1)
    axs[0].legend(fontsize=8, frameon=False, ncol=2, loc='upper right', title=r'$2R_c$', title_fontsize=8)
    axs[1].set_xlabel(r'$R_c/\xi$  ($\xi$: per video)')
    axs[0].set_title('(a)', loc='left')
    axs[1].set_title('(b)', loc='left')
    fig.tight_layout()
    for ext in ('png', 'svg'):
        fig.savefig(OUT / f'main_cos_eta_vs_rcxi.{ext}', dpi=300, bbox_inches='tight')
    print(f'saved: {OUT}/main_cos_eta_vs_rcxi.png')


if __name__ == '__main__':
    main()
