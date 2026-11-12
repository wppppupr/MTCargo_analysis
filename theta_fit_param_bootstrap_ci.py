#!/usr/bin/env python3
"""
theta_fit_param_bootstrap_ci.py

verify_theta_decomposition.py の θ ポテンシャルフィット
    U_model(θ) = A ln tan(θ/2) + 2 κ₂ cos²θ
のパラメータ A, κ₂, Λ = A/(4κ₂) に、軌道（cargo_id = exp_dir + particle）単位の
クラスタブートストラップで信頼区間を付ける。

- フィットの手順（ビン・フィット範囲・境界 [0,5]・U の基準 P(π/2)）は元スクリプトと同じ。
- Λ はブートストラップの各回で A_b/(4κ₂_b) を計算する（κ₂_b = 0 なら Λ = +∞）。
- κ₂ が境界 0 に張り付く回が多いサイズでは、両側 CI の上限が ∞ になるので、
  Λ の片側 95% 下限（5 パーセンタイル）を「Λ > 下限」として報告する。
- P(Λ_b > Λ_c) も出す（Λ_c = 2/(3√3)）。

出力: figure/theta_fit_ci/
  theta_fit_param_bootstrap_ci.csv / .png / .svg
  theta_fit_param_bootstrap_samples.csv（全ブートストラップ値）
キャッシュ: figure/theta_fit_ci/theta_instant_cache.pkl（--rebuild で再作成）
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

CUR = Path(__file__).resolve().parent
sys.path.insert(0, str(CUR))
import verify_theta_decomposition as vtd  # noqa: E402

OUT = CUR / "figure" / "theta_fit_ci"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "theta_instant_cache.pkl"

LAM_C = 2.0 / (3.0 * np.sqrt(3.0))
N_BINS = 28
BINS = np.linspace(0.0, np.pi, N_BINS + 1)
CENT = 0.5 * (BINS[:-1] + BINS[1:])
I_HALF = int(np.argmin(np.abs(CENT - np.pi / 2)))
K2_ZERO = 1e-3  # これ未満を「境界 κ₂=0 に張り付き」とみなす


def model(th, A, k2):
    t = np.clip(th, 1e-4, np.pi - 1e-4)
    return A * np.log(np.tan(t / 2.0)) + 2.0 * k2 * np.cos(t) ** 2


def fit_hist(pts, p0=(0.35, 0.8)):
    P, _ = np.histogram(pts, bins=BINS, density=True)
    if P[I_HALF] <= 0:
        return np.nan, np.nan
    with np.errstate(divide='ignore', invalid='ignore'):
        U = np.where(P > 0, -np.log(P / P[I_HALF]), np.nan)
    m = (P > 1e-4) & (CENT > 0.15) & (CENT < np.pi - 0.15)
    if m.sum() < 3:
        return np.nan, np.nan
    try:
        popt, _ = curve_fit(model, CENT[m], U[m], p0=p0,
                            bounds=([0.0, 0.0], [5.0, 5.0]), maxfev=5000)
        return float(popt[0]), float(popt[1])
    except Exception:
        return np.nan, np.nan


def lam_of(A, k2):
    A = np.asarray(A, float)
    k2 = np.asarray(k2, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(k2 > K2_ZERO, A / (4.0 * k2), np.inf)


def load_instant(rebuild=False):
    if CACHE.exists() and not rebuild:
        return pd.read_pickle(CACHE)
    active = vtd.load_active_particles(vtd.ROOT_DIR)
    xi = vtd.load_experiment_correlation_lengths(vtd.ROOT_DIR, CUR)
    df_i, df_w = vtd.load_all_data(vtd.ROOT_DIR, active)
    df_i['cargo_id'] = df_i['exp_dir'].astype(str) + '_' + df_i['particle'].astype(str)
    vals = [v for v in xi.values() if np.isfinite(v) and v > 0]
    dxi = float(np.mean(vals)) if vals else 6.0
    df_w['xi_um'] = df_w['exp_dir'].map(xi).fillna(dxi)
    df_w['rc_over_xi'] = df_w['radius_um'] / df_w['xi_um']
    rc = (df_w.drop_duplicates(['bead_name', 'particle'])
          .groupby('bead_name')['rc_over_xi'].mean())
    df_i['rc_over_xi_bead'] = df_i['bead_name'].map(rc)
    df_i = df_i[['bead_name', 'cargo_id', 'theta', 'rc_over_xi_bead']]
    df_i.to_pickle(CACHE)
    return df_i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-boot', type=int, default=2000)
    ap.add_argument('--rebuild', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()

    df = load_instant(a.rebuild)
    rng = np.random.default_rng(a.seed)
    rows, samples = [], []
    for b in vtd.BEADS_INFO:
        sub = df[df['bead_name'] == b['name']].dropna(subset=['theta'])
        if sub.empty:
            continue
        groups = [g['theta'].values for _, g in sub.groupby('cargo_id')]
        n = len(groups)
        A0, k0 = fit_hist(np.concatenate(groups))
        L0 = float(lam_of(A0, k0))
        bA = np.full(a.n_boot, np.nan)
        bk = np.full(a.n_boot, np.nan)
        if n >= 2:
            p0 = (A0 if np.isfinite(A0) else 0.35, max(k0, 0.05) if np.isfinite(k0) else 0.8)
            for i in range(a.n_boot):
                sel = rng.integers(0, n, n)
                bA[i], bk[i] = fit_hist(np.concatenate([groups[j] for j in sel]), p0)
        ok = np.isfinite(bA) & np.isfinite(bk)
        bL = lam_of(bA[ok], bk[ok])
        nb = int(ok.sum())
        frac_k0 = float(np.mean(bk[ok] <= K2_ZERO)) if nb else np.nan
        pct = lambda x, q: float(np.percentile(x, q, method='inverted_cdf')) if len(x) else np.nan
        L_lo95, L_hi95 = pct(bL, 2.5), pct(bL, 97.5)
        L_lb_1s = pct(bL, 5.0)  # 片側 95% 下限
        p_gt = float(np.mean(bL > LAM_C)) if nb else np.nan
        # 判定
        if not nb:
            verdict = 'CI なし'
        elif L_lo95 > LAM_C:
            verdict = 'Λ > Λ_c（95%）'
        elif np.isfinite(L_hi95) and L_hi95 < LAM_C:
            verdict = 'Λ < Λ_c（95%）'
        else:
            verdict = '判定不能（CI が Λ_c を含む）'
        rows.append(dict(
            bead_name=b['name'], diameter_um=b['diameter_um'],
            rc_over_xi=float(sub['rc_over_xi_bead'].iloc[0]),
            n_cargo=n, n_points=len(sub), n_boot_ok=nb,
            A=A0, A_lo=pct(bA[ok], 2.5), A_hi=pct(bA[ok], 97.5),
            kappa2=k0, k2_lo=pct(bk[ok], 2.5), k2_hi=pct(bk[ok], 97.5),
            frac_k2_at_0=frac_k0,
            Lambda=L0, Lam_lo=L_lo95, Lam_hi=L_hi95, Lam_lower_1sided95=L_lb_1s,
            Lam_median=pct(bL, 50), P_Lam_gt_Lc=p_gt,
            Lambda_is_lower_bound=bool(k0 <= K2_ZERO),
            Lam_lower_conservative=(pct(bA[ok], 2.5) / (4 * pct(bk[ok], 97.5))
                                    if nb and pct(bk[ok], 97.5) > 0 else np.nan),
            verdict=verdict))
        samples.append(pd.DataFrame(dict(bead_name=b['name'], A=bA, kappa2=bk)))
        print(f"{b['name']}: n={n} A={A0:.3f}[{rows[-1]['A_lo']:.3f},{rows[-1]['A_hi']:.3f}] "
              f"k2={k0:.3f}[{rows[-1]['k2_lo']:.3f},{rows[-1]['k2_hi']:.3f}] k2=0 frac={frac_k0:.3f} "
              f"Λ={L0:.3f}[{L_lo95:.3f},{L_hi95:.3f}] LB1s={L_lb_1s:.3f} P(Λ>Λc)={p_gt:.3f} {verdict}",
              flush=True)

    res = pd.DataFrame(rows).sort_values('rc_over_xi').reset_index(drop=True)
    res.to_csv(OUT / "theta_fit_param_bootstrap_ci.csv", index=False)
    pd.concat(samples).to_csv(OUT / "theta_fit_param_bootstrap_samples.csv", index=False)
    plot(res)


def plot(res):
    cmap = {b['name']: b for b in vtd.BEADS_INFO}
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8))
    LO, HI = 1e-2, 1e2  # Λ パネルの表示範囲（対数）
    for _, r in res.iterrows():
        b = cmap[r['bead_name']]
        kw = dict(color=b['color'], marker=b['marker'], ms=9, capsize=4, lw=1.6, mec='k', mew=0.6,
                  label=f"{b['diameter_um']:.2f} µm (N={r['n_cargo']})")
        xv = r['rc_over_xi']
        axs[0].errorbar(xv, r['A'], yerr=[[r['A'] - r['A_lo']], [r['A_hi'] - r['A']]], **kw)
        axs[1].errorbar(xv, r['kappa2'], yerr=[[r['kappa2'] - r['k2_lo']], [r['k2_hi'] - r['kappa2']]], **kw)
        if r['Lambda_is_lower_bound']:
            lb = r['Lam_lower_1sided95']
            axs[2].plot(xv, lb, color=b['color'], marker=b['marker'], ms=10, mec=b['color'], mew=2, mfc='none',
                        ls='none', label=kw['label'] + ' lower bound')
            axs[2].annotate('', xy=(xv, HI * 0.9), xytext=(xv, lb),
                            arrowprops=dict(arrowstyle='->', color=b['color'], lw=2))
            axs[2].annotate(f"Λ > {lb:.1f} (1-sided 95%)\nκ₂=0 in {r['frac_k2_at_0']*100:.0f}% of resamples",
                            (xv, lb), xytext=(10, -14), textcoords='offset points', fontsize=8.5, color=b['color'])
        else:
            c = float(np.clip(r['Lambda'], LO, HI))
            lo = float(np.clip(r['Lam_lo'], LO, HI)); hi = float(np.clip(r['Lam_hi'], LO, HI))
            axs[2].errorbar(xv, c, yerr=[[c - lo], [hi - c]], **kw)
        axs[2].annotate(f"P(Λ>Λc)={r['P_Lam_gt_Lc']:.2f}", (xv, LO), xytext=(0, 4),
                        textcoords='offset points', fontsize=8, rotation=90, va='bottom', ha='right',
                        color=b['color'])
    for ax, yl in zip(axs, [r"$A$", r"$\kappa_2$", r"$\Lambda = A/(4\kappa_2)$"]):
        ax.set_xscale('log'); ax.set_xlabel(r"$R_c/\xi$"); ax.set_ylabel(yl); ax.grid(alpha=0.3)
    axs[2].axhline(LAM_C, color='darkred', ls='-.', label=rf"$\Lambda_c={LAM_C:.3f}$")
    axs[2].set_yscale('log'); axs[2].set_ylim(LO, HI)
    axs[2].legend(loc='upper right', fontsize=7.5)
    axs[0].set_ylim(bottom=0); axs[1].set_ylim(bottom=0)
    axs[0].set_title("(A) A  [95% cluster bootstrap CI]")
    axs[1].set_title(r"(B) $\kappa_2$  [95% CI]")
    axs[2].set_title(r"(C) $\Lambda$  [95% CI; open = lower bound]")
    fig.tight_layout()
    for ext in ('png', 'svg'):
        fig.savefig(OUT / f"theta_fit_param_bootstrap_ci.{ext}", dpi=200, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
