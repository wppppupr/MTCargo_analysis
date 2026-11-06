#!/usr/bin/env python3
"""
scratch/plot_footprint_inner_radius_sweep.py

footprint_inner_radius_sweep.py のキャッシュから、足印の内径 R_c+δ の δ 依存を図にする。
δ=0 との「対の」差（同じ貨物・同じフレーム）を階層ブートストラップ（動画→貨物）で出す。
  Δη = η(δ) − η(0),  Δcosψ = cosψ(δ) − cosψ(0),  r_η = η(δ)/η(0),  r_Φ = Φ(δ)/Φ(0)
像の写り込みで η が押し上げられているなら、δ=0→0.5 µm（像の効く縁の帯, cargo_image_injection）で
η が Φ より大きく落ちるはず（r_η < r_Φ が最初の一歩に集中）。距離による向きの相関の減衰だけなら r_η ≈ r_Φ。
出力: figure/footprint_inner_radius_sweep/{inner_radius_sweep.png, paired_delta_table.csv/.txt}
"""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sp = ROOT / "libs" / "my_style.mplstyle"
if sp.exists(): plt.style.use(str(sp))
OUT = ROOT / "figure" / "footprint_inner_radius_sweep"
BEADS = ["beads06um", "beads1um", "beads3um", "beads5um", "beads7um", "beads20um"]
COLORS = ["#882255", "#CC6677", "#DDCC77", "#117733", "#44AA99", "#332288"]
DELTAS = ["0", "0.5", "1", "xi/2"]
NB = 1000
rng = np.random.default_rng(1)
lab = lambda b: b.replace("beads", "").replace("06", "0.6").replace("um", " µm")

d = pd.read_pickle(ROOT / "scratch" / "footprint_inner_radius_sweep_cache.pkl")
d = d[d.groupby(["unique_pid", "frame"]).delta.transform("nunique") == len(DELTAS)].copy()
d["cos_ok"] = d.cos_psi.notna().astype(float)
d["cos_s"] = d.cos_psi.fillna(0.0)
inj = pd.read_csv(ROOT / "figure" / "cargo_image_injection" / "image_injection_per_size.csv")
inj = inj[inj.source == "image"].set_index("bead_name")

# 貨物ごとに δ 別の和: [Σv∥, Σu_fp, Σcos, n_cos, ΣΦ, n] -> shape (cargo, δ, 6)
COLS = ["v_parallel", "u_fp", "cos_s", "cos_ok", "phi_fp", "one"]
d["one"] = 1.0


def cargo_arrays(sub):
    g = sub.groupby(["exp_dir", "unique_pid", "delta"])[COLS].sum()
    out = {}
    for e, ge in g.groupby(level=0):
        a = ge.unstack("delta")  # rows=cargo
        out[e] = np.stack([a.xs(c, axis=1, level=0)[DELTAS].values for c in COLS], -1)  # (cargo, δ, 6)
    return out


def stats(tot):  # tot (δ, 6)
    eta = tot[:, 0] / tot[:, 1]; cos = tot[:, 2] / tot[:, 3]; phi = tot[:, 4] / tot[:, 5]
    return eta, cos, phi


rows, curves = [], {}
for b in BEADS:
    mov = cargo_arrays(d[d.bead_name == b]); keys = list(mov)
    full = sum(m.sum(0) for m in mov.values())
    eta0, cos0, phi0 = stats(full)
    bs = []
    for _ in range(NB):
        tot = np.zeros_like(full)
        for k in rng.choice(keys, len(keys)):
            m = mov[k]; tot += m[rng.integers(0, len(m), len(m))].sum(0)
        e, c, p = stats(tot)
        bs.append(np.concatenate([e - e[0], c - c[0], (e / e[0]), p / p[0], e, c]))
    bs = np.array(bs); lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    n = len(DELTAS)
    dum = d[d.bead_name == b].groupby("delta").delta_um.mean()[DELTAS].values
    curves[b] = dict(dum=dum, eta=eta0, cos=cos0, phi=phi0,
                     eta_ci=(lo[4 * n:5 * n], hi[4 * n:5 * n]), cos_ci=(lo[5 * n:], hi[5 * n:]),
                     reta=eta0 / eta0[0], rphi=phi0 / phi0[0],
                     reta_ci=(lo[2 * n:3 * n], hi[2 * n:3 * n]))
    for i, t in enumerate(DELTAS):
        rows.append(dict(bead_name=b, delta=t, delta_um=dum[i], Phi_fp=phi0[i],
                         u_fp=d[(d.bead_name == b) & (d.delta == t)].u_fp.mean(),
                         eta=eta0[i], eta_lo=lo[4 * n + i], eta_hi=hi[4 * n + i],
                         cos_psi=cos0[i], cos_lo=lo[5 * n + i], cos_hi=hi[5 * n + i],
                         d_eta=eta0[i] - eta0[0], d_eta_lo=lo[i], d_eta_hi=hi[i],
                         d_cos=cos0[i] - cos0[0], d_cos_lo=lo[n + i], d_cos_hi=hi[n + i],
                         r_eta=eta0[i] / eta0[0], r_eta_lo=lo[2 * n + i], r_eta_hi=hi[2 * n + i],
                         r_phi=phi0[i] / phi0[0],
                         image_edge_slowdown=inj.edge.get(b, np.nan)))
tab = pd.DataFrame(rows)
tab.to_csv(OUT / "paired_delta_table.csv", index=False)
with pd.option_context("display.width", 250, "display.max_rows", 100):
    txt = tab.round(3).to_string(index=False)
(OUT / "paired_delta_table.txt").write_text(txt + "\n")
print(txt)

# ---- figure ----
fig, ax = plt.subplots(2, 2, figsize=(9, 7.5))
for b, c in zip(BEADS, COLORS):
    C = curves[b]; ls = "--" if b == "beads20um" else "-"; al = 0.5 if b == "beads20um" else 1
    kw = dict(color=c, ls=ls, alpha=al, marker="o", ms=4)
    ax[0, 0].plot(C["dum"], C["phi"], label=lab(b), **kw)
    ax[0, 1].plot(C["dum"], C["eta"], **kw)
    ax[0, 1].fill_between(C["dum"], *C["eta_ci"], color=c, alpha=0.12 * al, lw=0)
    ax[1, 0].plot(C["dum"], C["cos"], **kw)
    ax[1, 0].fill_between(C["dum"], *C["cos_ci"], color=c, alpha=0.12 * al, lw=0)
    if b != "beads20um":  # η(0)≈0 なので比は不定
        ax[1, 1].errorbar(C["rphi"], C["reta"], yerr=[C["reta"] - C["reta_ci"][0], C["reta_ci"][1] - C["reta"]],
                          color=c, marker="o", ms=4, capsize=2, lw=1)
ax[1, 1].plot([0.5, 1.05], [0.5, 1.05], "k:", lw=1, label=r"η $\propto$ Φ_fp")
ax[0, 0].set(xlabel="δ (µm)", ylabel="Φ_fp", title="(a) footprint polar order")
ax[0, 1].set(xlabel="δ (µm)", ylabel="η = Σv$_\parallel$ / Σ|<u>_fp|", title="(b) transport efficiency")
ax[1, 0].set(xlabel="δ (µm)", ylabel="<cos ψ>", title="(c) alignment")
ax[1, 1].set(xlabel="Φ_fp(δ) / Φ_fp(0)", ylabel="η(δ) / η(0)", title="(d) η drop vs Φ drop (paired)",
             xlim=(0.55, 1.05), ylim=(0.0, 1.3))
ax[0, 0].legend(fontsize=8, title="cargo size"); ax[1, 1].legend(fontsize=8, loc="lower right")
for a in ax.flat: a.axhline(0, color="0.7", lw=0.5)
fig.suptitle("Footprint ring inner radius R_c+δ (δ = 0, 0.5, 1 µm, ξ/2; ring width fixed)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "inner_radius_sweep.png", dpi=200)
print("saved", OUT / "inner_radius_sweep.png")
