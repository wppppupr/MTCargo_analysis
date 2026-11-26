#!/usr/bin/env python3
"""
κ の選び方の偏り (rotation_rolling_estimate.py の κ = median(|v|/|<u>|) | cos ψ > 0.8 の検定)。

物理的な狙い:
  κ は「方向が揃ったフレーム」だけで速さの比を取っている。カーゴ速度 v にブラウン運動などの
  等方な雑音 (各成分 σ) が乗ると、
    (i) |v| ≥ |v∥| なので雑音は |v| を常に増やす (Rice バイアス, 上向き)、
    (ii) cos ψ > 0.8 の選別は |<u>| が雑音に比べて小さいフレームでは「たまたま u 方向に
         大きく揺れたフレーム」を選ぶ (選択バイアス, 上向き)、
    (iii) |<u>| が小さいフレームは比を発散させる。
  ので、κ は真の伝達率より大きく出る恐れがある。特に揃いの悪い (⟨cos ψ⟩ が小さい) 大きいビーズほど
  効くはず。ここでは
    A(c)  = median(|v|/|u|) | cos ψ > c     (c = -1, 0, 0.5, 0.8, 0.9, 0.95)
    B     = Σ v∥|u| / Σ|u|²  (全フレーム、選別なし; v∥ = |v|cos ψ。等方雑音で不偏)
  を比べ、さらにモンテカルロ (実測の |u| 列 + v = κ_true u + N(0, σ_bead²), σ_bead² = ⟨v⊥²⟩)
  で A(0.8) の κ_true → κ_A 写像を作り、逆算した補正値 κ_corr を求める。
  σ_bead を v⊥ から取るのは「u に乗らない運動すべて」を雑音とみなす上限の見積もり。
  u 自体の測定雑音 (B を下げる errors-in-variables) はモデル化していない。
出力: figure/kappa_selection_bias/kappa_selection_bias.{png,svg}, kappa_selection_bias_{per_size.csv,stats.txt}
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rotation_rolling_estimate import load, kappa_roll, ORDER, LABEL, COLORS, MARKERS, CUR

OUT = CUR / 'figure' / 'kappa_selection_bias'
THS = [-1.0, 0.0, 0.5, 0.8, 0.9, 0.95]
K_GRID = np.linspace(0.0, 2.0, 41)
N_MC, N_BOOT = 20, 500
RNG = np.random.default_rng(1)


def est_A(vpar, vperp, u, c):
    vn = np.hypot(vpar, vperp)
    cos = vpar / np.where(vn > 0, vn, np.nan)
    m = cos > c
    return np.median(vn[m] / u[m]) if m.sum() else np.nan


def est_B(vpar, u):
    return np.sum(vpar * u) / np.sum(u * u)


def mc_curve(u, sig):
    """κ_true ごとに A(0.8) の期待値 (雑音を N_MC 回振る)。"""
    out = []
    for k in K_GRID:
        vals = [est_A(k * u + sig * RNG.standard_normal(len(u)), sig * RNG.standard_normal(len(u)), u, 0.8)
                for _ in range(N_MC)]
        out.append(np.nanmean(vals))
    return np.array(out)


def invert(curve, obs):
    """単調部分で κ_A = obs となる κ_true を線形補間。範囲外は nan。"""
    c = np.maximum.accumulate(curve)
    if not (c[0] <= obs <= c[-1]):
        return np.nan
    return float(np.interp(obs, c, K_GRID))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = load()
    d['vpar'] = d['v_cargo_norm'] * d['cos_bulk']
    d['vperp'] = d['v_cargo_norm'] * np.sqrt(np.clip(1 - d['cos_bulk'] ** 2, 0, None))
    d['sig'] = d.groupby('bead')['vperp'].transform(lambda x: np.sqrt(np.mean(x ** 2)))

    rows, curves = [], {}
    for b in ORDER:
        g = d[d.bead_name == b]
        vpar, vperp, u, sig = (g[k].to_numpy() for k in ('vpar', 'vperp', 'v_flow_norm', 'sig'))
        r = dict(bead_name=b, Rc=g['radius_um'].iloc[0], n_frames=len(g),
                 snr=np.median(u / sig), cos=g['cos_bulk'].mean())
        for c in THS:
            r[f'A{c:+.2f}'] = est_A(vpar, vperp, u, c)
            r[f'frac{c:+.2f}'] = (g['cos_bulk'] > c).mean()
        r['B'] = est_B(vpar, u)
        beads = g['bead'].unique()
        gb = {k: g[g.bead == k] for k in beads}
        boot = []
        for _ in range(N_BOOT):
            s = pd.concat([gb[k] for k in RNG.choice(beads, len(beads))])
            boot.append(est_B(s['vpar'].to_numpy(), s['v_flow_norm'].to_numpy()))
        r['B_lo'], r['B_hi'] = np.percentile(boot, [16, 84])
        cur = mc_curve(u, sig)
        curves[b] = cur
        r['A_at_ktrue0'] = cur[0]          # κ_true = 0 でも出てしまう A(0.8) (純雑音の床)
        r['A_corr'] = invert(cur, r['A+0.80'])
        r['roll_50nm'] = float(kappa_roll(r['Rc'], 0.05))
        rows.append(r)
    sz = pd.DataFrame(rows)
    sz.to_csv(OUT / 'kappa_selection_bias_per_size.csv', index=False)

    cols = ['bead_name', 'Rc', 'snr', 'cos', 'A-1.00', 'A+0.00', 'A+0.80', 'A+0.95', 'A_at_ktrue0', 'A_corr', 'B', 'B_lo', 'B_hi', 'roll_50nm']
    txt = '\n'.join([
        'A(c) = median |v|/|u| over frames with cos psi > c; B = sum(v_par |u|)/sum(|u|^2) (all frames, no selection)',
        'snr = median |u|/sigma, sigma^2 = <v_perp^2> per bead (upper bound of isotropic noise)',
        'A_at_ktrue0 = MC value of A(0.8) when true kappa = 0 (noise floor); A_corr = MC-inverted A(0.8)',
        '', sz[cols].round(3).to_string(index=False), '',
        'fraction of frames passing cos>c:', sz[['bead_name'] + [f'frac{c:+.2f}' for c in THS]].round(3).to_string(index=False)])
    (OUT / 'kappa_selection_bias_stats.txt').write_text(txt + '\n')
    print(txt)

    fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.8))
    ax = axs[0]
    for i, b in enumerate(ORDER):
        r = sz.iloc[i]
        ax.plot(THS, [r[f'A{c:+.2f}'] for c in THS], marker=MARKERS[i], color=COLORS[i], mec='k', mew=0.5, label=LABEL[b])
    ax.axvline(0.8, color='0.5', ls=':', lw=0.8)
    ax.set_xlabel(r'selection threshold $c$ (cos $\psi > c$)')
    ax.set_ylabel(r'$A(c)$ = median $|\mathbf{v}|/|\langle\mathbf{u}\rangle|$')
    ax.set_yscale('log')
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.set_title('(a)', loc='left')

    ax = axs[1]
    for i, b in enumerate(ORDER):
        ax.plot(K_GRID, curves[b], color=COLORS[i], lw=1.2)
        ax.axhline(sz.iloc[i]['A+0.80'], color=COLORS[i], ls=':', lw=0.8)
    ax.plot(K_GRID, K_GRID, 'k--', lw=0.8, label='no bias')
    ax.set_xlabel(r'true $\kappa$ (MC input)')
    ax.set_ylabel(r'MC $\langle A(0.8)\rangle$')
    ax.legend(fontsize=7, frameon=False)
    ax.set_title('(b)  dotted: observed A(0.8)', loc='left', fontsize=9)

    ax = axs[2]
    a_grid = np.logspace(np.log10(0.25), np.log10(12), 100)
    ax.fill_between(a_grid, kappa_roll(a_grid, 0.2), kappa_roll(a_grid, 0.02), color='0.8', lw=0, label='rolling (near wall)')
    ax.axhline(4 / 7, color='0.4', ls='--', lw=0.8)
    ax.axhline(0, color='0.3', lw=0.5)
    for i, b in enumerate(ORDER):
        r = sz.iloc[i]
        x = r['Rc']
        ax.plot(x, r['A+0.80'], marker=MARKERS[i], mfc='none', mec=COLORS[i], ms=8, ls='none')
        ax.plot(x * 1.08, r['A_corr'], marker=MARKERS[i], color=COLORS[i], mec='k', mew=0.5, ms=7, ls='none')
        ax.errorbar(x * 0.92, r['B'], yerr=[[r['B'] - r['B_lo']], [r['B_hi'] - r['B']]], marker='x', color=COLORS[i],
                    ms=7, capsize=2, ls='none')
    for m, lab in (('o', 'A(0.8) (current)'), ('s', 'A(0.8) MC-corrected'), ('x', 'B (projection, no cut)')):
        ax.plot([], [], marker=m, color='0.3', mfc='none' if m == 'o' else '0.3', ls='none', label=lab)
    ax.set_xscale('log')
    ax.set_xlabel(r'$R_c$ [µm]')
    ax.set_ylabel(r'$\kappa$ estimate')
    ax.legend(fontsize=7, frameon=False, loc='upper right')
    ax.set_title('(c)', loc='left')
    fig.tight_layout()
    for ext in ('png', 'svg'):
        fig.savefig(OUT / f'kappa_selection_bias.{ext}', dpi=300, bbox_inches='tight')
    print(f'saved: {OUT}/kappa_selection_bias.png')


if __name__ == '__main__':
    main()
