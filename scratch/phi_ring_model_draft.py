import sys, numpy as np, pandas as pd
sys.path.insert(0,'.')
import plot_polar_order_vs_normalized_velocity as P
from scipy.integrate import quad
df=P.load_data('frame')
xi=pd.read_csv('figure/bg_angular_correlation/bg_angular_correlation_length_per_experiment.csv').set_index('exp_dir')
win=pd.read_csv('figure/xi_time_window/xi_t_windows_w200.csv')
b_mov=win.groupby('exp_dir').u2_local.mean()
RAD={'beads06um':3,'beads1um':6,'beads3um':15,'beads5um':25,'beads7um':35,'beads20um':95}
BEST={'beads06um':5,'beads1um':10,'beads3um':20,'beads5um':30,'beads7um':40,'beads20um':100}
def ringF(R,x):  # <cos> averaged over pairs on ring radius R with C=exp(-r/xi)
    return quad(lambda p: np.exp(-2*R*np.sin(p)/x),0,np.pi)[0]/np.pi
rows=[]
for (bn,ed),g in df.groupby(['bead_name','exp_dir']):
    if ed not in xi.index: continue
    Rm=0.11*(RAD[bn]+BEST[bn])/2; x=xi.loc[ed,'xi_um']; b=b_mov.get(ed,np.nan)
    F=ringF(Rm,x); 
    rows.append(dict(bead=bn,D=g.diameter_um.iloc[0],movie=ed,Rc_xi=0.11*RAD[bn]/x,
        phi_obs=g.phi_t.mean(), phi2_obs=np.sqrt((g.phi_t**2).mean()),
        phi_pred=np.sqrt(F), phi_pred_b=np.sqrt(b+(1-b)*F),
        vfp_norm=(g.u_fp/g.u_inf).mean(),
        vpar_norm=g.v_parallel.sum()/g.u_inf.sum(),
        eta=g.v_parallel.sum()/g.u_fp.sum(), n=len(g)))
r=pd.DataFrame(rows).sort_values(['D','movie'])
pd.set_option('display.width',200); print(r.round(3).to_string(index=False))
from scipy.stats import spearmanr, pearsonr
for c in ['phi_obs','vfp_norm','vpar_norm','eta']:
    print(c,'vs Rc/xi spearman %.2f'%spearmanr(r.Rc_xi,r[c])[0], '| vs Rc spearman %.2f'%spearmanr(r.D,r[c])[0])
print('phi_obs vs phi_pred pearson %.2f, vs pred_b %.2f'%(pearsonr(r.phi_obs,r.phi_pred)[0],pearsonr(r.phi_obs,r.phi_pred_b)[0]))
# vpar_norm vs phi_obs proportionality
for c in ['phi_obs','vfp_norm']:
    k=(r.vpar_norm*r[c]).sum()/(r[c]**2).sum(); res=r.vpar_norm-k*r[c]
    print(f'vpar_norm ~ k*{c}: k={k:.2f}, R2={1-(res**2).sum()/((r.vpar_norm-r.vpar_norm.mean())**2).sum():.2f}')
import os; os.makedirs('figure/phi_ring_model',exist_ok=True)
r.to_csv('figure/phi_ring_model/phi_model_movies.csv',index=False)
