#!/usr/bin/env python3
"""
phi_ring_model.py
=================

足印（貨物まわりの円環）の極性秩序 Φ_fp を、背景の配向相関長 ξ だけから予測するモデル。

モデル
------
背景の単位流向ベクトルの相関を C(r) = exp(-r/ξ) と仮定する。
半径 R の円環上の 2 点（中心角 θ）の距離は 2R sin(θ/2)。φ = θ/2 として円環上で平均すると

    Φ_fp² = <|<n>_ring|²> = (1/π) ∫_0^π exp(-2 (R/ξ) sin φ) dφ

R には足印の円環（RAD_MAP < r ≤ BEST_MAP [px]）の中央の半径
    R = 0.11 µm/px × (RAD + BEST) / 2
を使う。ξ は動画ごとの背景配向相関長
(figure/bg_angular_correlation/bg_angular_correlation_length_per_experiment.csv の xi_um)。

モデルは <Φ²> を予測するので、実測側は Φ_rms = sqrt(<Φ_t²>) と比べるのが正しい。
図 (vpar_and_phi_vs_rc_over_xi.png) の Φ は <Φ_t> なので、両方を表に出す
(Jensen の不等式で <Φ_t> ≤ Φ_rms)。

出力 (figure/phi_ring_model/)
-----------------------------
- phi_ring_model_per_movie.csv / .md : 動画ごとの実測と予測
- phi_ring_model_per_size.csv        : サイズごとのまとめ (frame 数重み)
- phi_ring_model_parity.png / .svg   : 予測 vs 実測
20 µm は参考扱い (reference=True; 統計値の計算から除外, 図では灰色)。

単体実行:  .pixi/envs/default/bin/python phi_ring_model.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import quad

CURRENT_DIR = Path(__file__).resolve().parent
XI_CSV = CURRENT_DIR / "figure" / "bg_angular_correlation" / "bg_angular_correlation_length_per_experiment.csv"
OUT_DIR = CURRENT_DIR / "figure" / "phi_ring_model"

# scratch/extract_footprint_heterogeneity.py の足印定義と同じ値 (px)
SCALE_UM_PER_PX = 0.11
RAD_MAP = {'beads06um': 3, 'beads1um': 6, 'beads3um': 15, 'beads5um': 25, 'beads7um': 35, 'beads20um': 95}
BEST_MAP = {'beads06um': 5, 'beads1um': 10, 'beads3um': 20, 'beads5um': 30, 'beads7um': 40, 'beads20um': 100}
REFERENCE_BEADS = {'beads20um'}


def ring_mid_radius_um(bead_name: str) -> float:
    """足印の円環の中央の半径 [µm]。"""
    return SCALE_UM_PER_PX * 0.5 * (RAD_MAP[bead_name] + BEST_MAP[bead_name])


def ring_phi2(R_over_xi: float) -> float:
    """Φ_fp² = (1/π) ∫_0^π exp(-2 (R/ξ) sin φ) dφ"""
    return quad(lambda p: np.exp(-2.0 * R_over_xi * np.sin(p)), 0.0, np.pi)[0] / np.pi


def ring_phi(R_over_xi) -> np.ndarray:
    """Φ_fp = sqrt(Φ_fp²)。スカラー/配列どちらも可。"""
    return np.sqrt(np.vectorize(ring_phi2)(np.asarray(R_over_xi, dtype=float)))


def load_xi() -> pd.Series:
    return pd.read_csv(XI_CSV).set_index('exp_dir')['xi_um']


def movie_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    df: plot_vpar_and_phi_vs_scaled_radius.load_data() の出力 (frame 単位, phi_t 列を含む)
    動画ごとに実測 Φ と予測 Φ を並べた表を返す。
    """
    xi = load_xi()
    rows = []
    for (bn, ed), g in df.groupby(['bead_name', 'exp_dir']):
        if ed not in xi.index or bn not in RAD_MAP:
            continue
        x = float(xi.loc[ed])
        R = ring_mid_radius_um(bn)
        pred = float(ring_phi(R / x))
        phi_mean = float(g['phi_t'].mean())
        phi_rms = float(np.sqrt((g['phi_t'] ** 2).mean()))
        rows.append(dict(
            bead_name=bn, diameter_um=float(g['diameter_um'].iloc[0]), exp_dir=ed,
            reference=bn in REFERENCE_BEADS,
            xi_um=x, Rc_um=float(g['radius_um'].iloc[0]), R_ring_um=R,
            Rc_over_xi=float(g['radius_um'].iloc[0]) / x, R_over_xi=R / x,
            phi_mean_obs=phi_mean, phi_rms_obs=phi_rms, phi_pred=pred,
            resid_rms=phi_rms - pred, ratio_rms=phi_rms / pred,
            n_frames_obs=len(g), n_cargo=int(g['unique_pid'].nunique()),
        ))
    return pd.DataFrame(rows).sort_values(['diameter_um', 'exp_dir']).reset_index(drop=True)


def size_table(mt: pd.DataFrame) -> pd.DataFrame:
    """サイズごとのまとめ。観測数 (frame) で重み付けした平均 (図の大マーカーと同じ重み)。"""
    out = []
    for bn, g in mt.groupby('bead_name', sort=False):
        w = g['n_frames_obs'].values
        avg = lambda c: float(np.average(g[c], weights=w))
        out.append(dict(
            bead_name=bn, diameter_um=g['diameter_um'].iloc[0], reference=bool(g['reference'].iloc[0]),
            n_movies=len(g), R_ring_um=g['R_ring_um'].iloc[0],
            Rc_over_xi=avg('Rc_over_xi'), R_over_xi=avg('R_over_xi'),
            phi_mean_obs=avg('phi_mean_obs'), phi_rms_obs=avg('phi_rms_obs'),
            phi_pred=avg('phi_pred'), phi_pred_min=g['phi_pred'].min(), phi_pred_max=g['phi_pred'].max(),
        ))
    st = pd.DataFrame(out).sort_values('diameter_um').reset_index(drop=True)
    st['resid_rms'] = st['phi_rms_obs'] - st['phi_pred']
    return st


def _summary_lines(mt: pd.DataFrame) -> list:
    from scipy.stats import pearsonr, spearmanr
    m = mt[~mt['reference']]
    lines = []
    for col in ['phi_rms_obs', 'phi_mean_obs']:
        r = pearsonr(m['phi_pred'], m[col])[0]
        rho = spearmanr(m['phi_pred'], m[col])[0]
        res = m[col] - m['phi_pred']
        lines.append(f"{col} vs phi_pred (20 µm 除く, n={len(m)}): Pearson r={r:.2f}, Spearman ρ={rho:.2f}, "
                     f"平均残差={res.mean():+.3f}, RMS 残差={np.sqrt((res ** 2).mean()):.3f}")
    return lines


def parity_plot(mt: pd.DataFrame, st: pd.DataFrame, out_dir: Path):
    import matplotlib.pyplot as plt
    style = CURRENT_DIR / 'libs' / 'my_style.mplstyle'
    if style.exists():
        try:
            plt.style.use(str(style))
        except Exception:
            pass
    colors = {'beads06um': '#882255', 'beads1um': '#CC6677', 'beads3um': '#DDCC77',
              'beads5um': '#999933', 'beads7um': '#117733', 'beads20um': '#999999'}
    markers = {'beads06um': '^', 'beads1um': 'o', 'beads3um': 'd', 'beads5um': 'p', 'beads7um': 'h', 'beads20um': 's'}

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    # (a) 予測 vs 実測 (rms)
    ax = axes[0]
    ax.plot([0, 1.05], [0, 1.05], color='#555555', lw=1, ls='--', zorder=1)
    for bn, g in mt.groupby('bead_name', sort=False):
        ref = bn in REFERENCE_BEADS
        lab = f"{g['diameter_um'].iloc[0]:g} µm" + (" (ref.)" if ref else "")
        ax.scatter(g['phi_pred'], g['phi_rms_obs'], marker=markers[bn], s=55,
                   color=colors[bn], alpha=0.5 if ref else 0.9, edgecolors='#222222', linewidths=0.6,
                   label=lab, zorder=3)
    ax.set_xlim(0.0, 1.05); ax.set_ylim(0.0, 1.05); ax.set_aspect('equal')
    ax.set_xlabel(r"Predicted $\Phi_{fp}$ (ring model, $R$ = ring mid-radius)")
    ax.set_ylabel(r"Observed $\sqrt{\langle\Phi_t^2\rangle}$ (per movie)")
    ax.set_title("(a) Per-movie: observed vs predicted")
    ax.legend(fontsize=8.5, loc='upper left')

    # (b) Φ vs R/ξ with model curve
    ax = axes[1]
    xs = np.logspace(-2, 1, 300)
    ax.plot(xs, ring_phi(xs), color='#d62728', lw=2, label=r"model $\Phi_{fp}(R/\xi)$", zorder=2)
    for bn, g in mt.groupby('bead_name', sort=False):
        ref = bn in REFERENCE_BEADS
        ax.scatter(g['R_over_xi'], g['phi_rms_obs'], marker=markers[bn], s=55, color=colors[bn],
                   alpha=0.5 if ref else 0.9, edgecolors='#222222', linewidths=0.6, zorder=3)
        ax.scatter(g['R_over_xi'], g['phi_mean_obs'], marker=markers[bn], s=30, facecolors='none',
                   edgecolors=colors[bn], alpha=0.5 if ref else 0.9, linewidths=0.9, zorder=3)
    ax.scatter([], [], marker='o', s=45, color='#777777', edgecolors='#222222', label=r"obs. $\sqrt{\langle\Phi_t^2\rangle}$")
    ax.scatter([], [], marker='o', s=30, facecolors='none', edgecolors='#777777', label=r"obs. $\langle\Phi_t\rangle$")
    ax.set_xscale('log'); ax.set_xlim(0.05, 5); ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(r"$R/\xi$  ($R$: footprint ring mid-radius)")
    ax.set_ylabel(r"$\Phi_{fp}$")
    ax.set_title(r"(b) $\Phi_{fp}$ vs $R/\xi$ (grey = 20 µm, reference)")
    ax.legend(fontsize=8.5, loc='lower left')
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / 'phi_ring_model_parity.png', dpi=300, bbox_inches='tight')
    fig.savefig(out_dir / 'phi_ring_model_parity.svg', bbox_inches='tight')
    plt.close(fig)


def _to_md(t: pd.DataFrame) -> str:
    """tabulate 不要の簡易 markdown 表。"""
    def fmt(v):
        return f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v)
    head = "| " + " | ".join(t.columns) + " |\n|" + "---|" * len(t.columns) + "\n"
    return head + "\n".join("| " + " | ".join(fmt(v) for v in row) + " |" for row in t.itertuples(index=False))


def write_outputs(mt: pd.DataFrame, st: pd.DataFrame, out_dir: Path = OUT_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    mt.to_csv(out_dir / 'phi_ring_model_per_movie.csv', index=False)
    st.to_csv(out_dir / 'phi_ring_model_per_size.csv', index=False)
    cols = ['diameter_um', 'exp_dir', 'xi_um', 'R_ring_um', 'R_over_xi',
            'phi_mean_obs', 'phi_rms_obs', 'phi_pred', 'resid_rms', 'n_frames_obs']
    with open(out_dir / 'phi_ring_model_per_movie.md', 'w') as f:
        f.write("# 足印円環モデル: 動画ごとの実測 Φ と予測 Φ\n\n")
        f.write("予測: Φ_fp² = (1/π)∫_0^π exp(−2(R/ξ)sinφ)dφ, R = 足印円環の中央半径, "
                "ξ = 背景配向相関長 (動画ごと)。resid_rms = sqrt(<Φ_t²>) − 予測。\n")
        f.write("20 µm は参考扱い (*印, 統計から除外)。\n\n")
        t = mt.copy()
        t['exp_dir'] = [e + (' *' if r else '') for e, r in zip(t['exp_dir'], t['reference'])]
        f.write(_to_md(t[cols]))
        f.write("\n\n## サイズごと (frame 数重み)\n\n")
        f.write(_to_md(st.drop(columns=['bead_name'])))
        f.write("\n\n## 一致の指標\n\n")
        for line in _summary_lines(mt):
            f.write(f"- {line}\n")
    parity_plot(mt, st, out_dir)


def main():
    import plot_vpar_and_phi_vs_scaled_radius as V
    df = V.load_data(u_inf_mode='frame')
    mt = movie_table(df)
    st = size_table(mt)
    write_outputs(mt, st)
    pd.set_option('display.width', 220)
    print(mt.round(3).to_string(index=False))
    print()
    print(st.round(3).to_string(index=False))
    print()
    for line in _summary_lines(mt):
        print(line)
    print(f"Saved to {OUT_DIR}")


if __name__ == '__main__':
    main()
