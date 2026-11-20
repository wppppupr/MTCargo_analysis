import numpy as np, pandas as pd
from scipy.special import iv
from scipy.optimize import brentq
d = pd.read_pickle("/home/sasaki/MTCargo_analysis/scratch/harmonics_frames.pkl")
order = ["beads06um","beads1um","beads3um","beads5um","beads7um","beads20um"]
N = 4
def vm_coef(a1, n):
    k = brentq(lambda k: iv(1,k)/iv(0,k)-a1, 1e-6, 200) if a1 > 1e-6 else 0
    return iv(n,k)/iv(0,k)
print("pixel-weighted per-point harmonics a_n, identity check, factorization, vM-matched")
for b in order:
    g = d[d.bead==b]; w = g.npts.values
    rcxi = g.rcxi.median()
    line = [f"{b:10s} x={rcxi:.2f}"]
    for n in range(1,N+1):
        a = np.average(g[f"Ac{n}"], weights=w)                      # per-point <cos n th>
        sn = np.average(g[f"As{n}"], weights=w)                     # mirror check
        ident = np.average(g[f"S{n}"]*g[f"cpsi{n}"], weights=w)     # S_n cos n psi_n (exact)
        S = np.average(g[f"S{n}"], weights=w); c = np.average(g[f"cpsi{n}"], weights=w)
        line.append(f"n={n}: a={a:+.3f} sin={sn:+.3f} id={ident:+.3f} S={S:.3f} cos={c:+.3f} S*cos={S*c:+.3f}")
    print("\n   ".join(line))
print("\nvM test: a_n observed vs vM(a1); DeltaD_high ~ sum_{n>=2}(a_n - a_n^vM)^2 + sum b_n^2")
for b in order:
    g = d[d.bead==b]; w = g.npts.values
    a = [np.average(g[f"Ac{n}"], weights=w) for n in range(1,7)]
    bs = [np.average(g[f"As{n}"], weights=w) for n in range(1,7)]
    vm = [vm_coef(max(a[0],0), n) for n in range(1,7)]
    dD1 = sum(vm[n-1]**2 for n in range(1,7))
    dDh = sum((a[n-1]-vm[n-1])**2 for n in range(2,7)) + sum(x**2 for x in bs)
    per = " ".join(f"{a[n-1]-vm[n-1]:+.3f}" for n in range(2,7))
    print(f"{b:10s} a1={a[0]:.3f} D1~{dD1:.4f} dDhigh~{dDh:.4f}  a_n-vM (n=2..6): {per}")
print("\nsubstrate order <S_n> and cargo factor <cos n psi_n> per size (n=1..6)")
for b in order:
    g = d[d.bead==b]; w = g.npts.values
    S = [np.average(g[f"S{n}"], weights=w) for n in range(1,7)]
    C = [np.average(g[f"cpsi{n}"], weights=w) for n in range(1,7)]
    print(f"{b:10s} S: "+" ".join(f"{x:.3f}" for x in S)+" | cos: "+" ".join(f"{x:+.3f}" for x in C))
