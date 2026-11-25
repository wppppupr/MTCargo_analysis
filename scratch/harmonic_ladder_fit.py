# cargo-factor ladder c_n = <cos n psi> (psi rel. footprint polar axis) vs offset-normal g_n(s) averaged over P(s)
import numpy as np, pandas as pd
from scipy.special import ive
from scipy.optimize import least_squares
from scipy.stats import gamma, lognorm
from numpy.polynomial import chebyshev as C
d = pd.read_pickle("/home/sasaki/MTCargo_analysis/scratch/harmonics_frames.pkl")
order = ["beads06um","beads1um","beads3um","beads5um","beads7um","beads20um"]
NM = 6
def gn(n, s):
    z = s*s/4
    return np.sqrt(np.pi)/2*(s/np.sqrt(2))*(ive((n-1)/2, z)+ive((n+1)/2, z))
sg = np.linspace(1e-4, 60, 6000); ds = sg[1]-sg[0]
G = np.array([gn(n, sg) for n in range(1, NM+1)])
def ladder(pdf):
    w = pdf(sg); w = w/(w.sum()*ds)
    return (G*w).sum(1)*ds
models = {
 "single s":  (lambda p: gn(np.arange(1,NM+1), p[0]), [1.0]),
 "gamma":     (lambda p: ladder(lambda s: gamma.pdf(s, p[0], scale=p[1]/p[0])), [1.0, 1.0]),
 "lognormal": (lambda p: ladder(lambda s: lognorm.pdf(s, p[0], scale=p[1])), [1.0, 0.5]),
}
for b in order:
    g = d[d.bead==b]
    c = np.array([np.mean(C.chebval(g.cpsi1.values, [0]*n+[1])) for n in range(1, NM+1)])
    se = np.array([np.std(C.chebval(g.cpsi1.values, [0]*n+[1]))/np.sqrt(g.pid.nunique()*20) for n in range(1,NM+1)])
    print(f"\n{b}: observed c_n = " + " ".join(f"{x:+.3f}" for x in c))
    for name, (f, p0) in models.items():
        r = least_squares(lambda p: (f(np.abs(p))-c)/se, p0, bounds=(1e-3, 50))
        p = np.abs(r.x); fit = f(p)
        extra = ""
        if name == "gamma": extra = f" mean s={p[1]:.2f} CV={1/np.sqrt(p[0]):.2f}"
        if name == "lognormal": extra = f" median s={p[1]:.2f} sigma_ln={p[0]:.2f} mean={p[1]*np.exp(p[0]**2/2):.2f}"
        if name == "single s": extra = f" s={p[0]:.2f}"
        print(f"  {name:9s} chi2={np.sum(r.fun**2):8.1f} fit=" + " ".join(f"{x:+.3f}" for x in fit) + extra)
