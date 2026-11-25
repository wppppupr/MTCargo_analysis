# independent test: angle-ladder fit -> predicted spread of ln|v_c|; compare with measured speeds
import numpy as np, pandas as pd
rng = np.random.default_rng(1)
fits = {  # lognormal (sigma_ln, median) and gamma (CV, mean) from ladder_fit.py
 "beads06um": ((1.89,0.32),(1.45,1.51)), "beads1um": ((1.84,0.26),(1.45,1.26)),
 "beads3um": ((1.00,0.35),(1.08,0.57)), "beads5um": ((1.54,0.09),(1.45,0.42)),
 "beads7um": ((0.95,0.21),(1.10,0.34)), "beads20um": ((2.52,0.02),(2.43,0.40))}
def pred(sv):
    n = rng.normal(size=(len(sv),2)); v = np.c_[sv+n[:,0], n[:,1]]
    return np.std(np.log(np.hypot(v[:,0],v[:,1])))
df = pd.read_pickle("/home/sasaki/MTCargo_analysis/scratch/contrast_inst_cache.pkl")
d = pd.read_pickle("/home/sasaki/MTCargo_analysis/scratch/harmonics_frames.pkl")
keys = set(zip(d.exp, d.frame))
for b,(ln,ga) in fits.items():
    g = df[df.bead_name==b]
    g = g[[k in keys for k in zip(g.exp_dir, g.frame)]]
    sp = np.hypot(g.cargo_dx, g.cargo_dy); sp = sp[sp>0]
    obs = np.std(np.log(sp))
    # within-particle spread (removes cargo-to-cargo differences)
    lw = np.log(np.hypot(g.cargo_dx,g.cargo_dy)).replace(-np.inf,np.nan)
    within = np.sqrt(np.nanmean((lw - lw.groupby([g.exp_dir,g.particle]).transform("mean"))**2))
    s_ln = rng.lognormal(np.log(ln[1]), ln[0], 200000)
    s_ga = rng.gamma(1/ga[0]**2, ga[1]*ga[0]**2, 200000)
    print(f"{b:10s} sd ln|v| observed={obs:.2f} within-cargo={within:.2f} | pred lognormal={pred(s_ln):.2f} gamma={pred(s_ga):.2f} pure-noise(s=0)={pred(np.zeros(200000)):.2f}")
