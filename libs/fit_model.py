import numpy as np

# 冪乗関数
def power_law(x, k, a):
    return a * np.abs(x) ** k

def ln_pl(x, k, a):
        return k * x + np.log10(a)

# MSDのフィッティング
def MSD_fit(t, a, C):
    return a * (t)**2 / (t+C)

def MSD_loglog(t, a, C):
    return np.log10(a) + 2 * np.log10(t) - np.log10(t+C)

# 指数関数
def exp_dist(x, lam):
    return 1/lam * np.exp(-x/lam)

def exp_line(x, lam):
    return - x/(lam*np.log(10)) - np.log10(lam)

# stretced exponential
def stretched_exp(x, a, b):
        return np.exp(-np.abs(x/a) ** b)

# gaussian
def gaussian(x, mu, sd):
    return (1/np.sqrt(2*np.pi*sd**2))*np.exp(-((x-mu)**2)/(2*sd**2))

def log_gaussian(x, mu, sd):
    return -((x-mu)**2)/(2*np.log(10)*sd**2)-np.log10(2*np.pi)/2-np.log10(sd)

def multi_gau(x, p1, mu1, sd1, mu2, sd2):
    return (p1/np.sqrt(2*np.pi*sd1**2))*np.exp(-((x-mu1)**2)/(2*sd1**2)) + ((1-p1)/np.sqrt(2*np.pi*sd2**2))*np.exp(-((x-mu2)**2)/(2*sd2**2))

def log_norm(x, mu, sd):
    return (1/(np.sqrt(2*np.pi*sd**2)*x))*np.exp(-((np.log(x/mu))**2)/(2*sd**2))

# Levy-smirnov
def levy_smirnov(t, tau):
    return (np.sqrt(tau)/(2*np.sqrt(np.pi*t**3))) * np.exp(-tau/(4*t))

def log_ls(t, tau):
    return -tau/(4*t*np.log(10)) + (1/2) * (np.log10(tau)-np.log10(2*np.pi)-3*np.log10(t))

def velocity_model(v, mu_stop, sd_stop, mu, sd, A):
    return (A/(np.sqrt(2*np.pi*sd_stop**2)*v))*np.exp(-((np.log(v/mu_stop))**2)/(2*sd_stop**2)) + ((1-A)/np.sqrt(2*np.pi*sd**2))*np.exp(-((v-mu)**2)/(2*sd**2))

# Tempered power law distribution
def tpl(x, alpha, lam, A):
    return A * x ** (-alpha) * np.exp(-lam * x)

def log_tpl(x, alpha, lam, A):
    return np.log10(A) - alpha * x - (lam * x)*np.log10(np.e)

# waiting time distribution
def wtd(x, alpha, tau):
    return alpha/(tau * (1+x/tau)**(alpha+1))

def log_wtd(x, alpha, tau):
    return np.log10(alpha) - np.log10(tau) - (alpha+1) * np.log10(1+x/tau)

# 2-state Run-and-Tumble Particle (RTP) Total MSD model
def rtp_2state_msd(dt, D0, sigma_noise_sq, f_run=0.0, v_R=0.0, tau_eff=np.nan):
    """
    2-state RTP total MSD:
    <dr^2(t)> = 4 * D0 * t + 2 * f_run * v_R^2 * tau_eff * [ t - tau_eff * (1 - exp(-t / tau_eff)) ] + 4 * sigma_noise_sq
    
    tau_eff が NaN または <= 0 の場合は、Run時間分布がない極限として純粋拡散 4 * D0 * t + 4 * sigma_noise_sq を返す。
    """
    if tau_eff is None or not np.isfinite(tau_eff) or tau_eff <= 0:
        return 4.0 * D0 * dt + 4.0 * sigma_noise_sq
    tr = np.maximum(tau_eff, 1e-12)
    active_term = 2.0 * f_run * (v_R ** 2) * tr * (dt - tr * (1.0 - np.exp(-dt / tr)))
    return 4.0 * D0 * dt + active_term + 4.0 * sigma_noise_sq

def log_rtp_2state_msd(dt, D0, sigma_noise_sq, f_run=0.0, v_R=0.0, tau_eff=np.nan):
    """
    Log10 of 2-state RTP total MSD for log-log curve fitting.
    """
    msd_val = rtp_2state_msd(dt, D0, sigma_noise_sq, f_run, v_R, tau_eff)
    return np.log10(np.maximum(msd_val, 1e-12))


# Caged / Hopping diffusion MSD model: <dr^2(t)> = L_cage^2 * (1 - exp(-t / tau_r)) + 4 * D_eff * t
def caged_msd(dt, L_cage_sq, tau_r, D_eff):
    """
    Caged diffusion MSD model:
    <dr^2(t)> = L_cage^2 * (1 - exp(-t / tau_r)) + 4 * D_eff * t
    """
    tr = np.maximum(tau_r, 1e-12)
    return L_cage_sq * (1.0 - np.exp(-dt / tr)) + 4.0 * D_eff * dt


def log_caged_msd(dt, L_cage_sq, tau_r, D_eff):
    """
    Log10 of caged diffusion MSD for log-log curve fitting.
    """
    msd_val = caged_msd(dt, L_cage_sq, tau_r, D_eff)
    return np.log10(np.maximum(msd_val, 1e-12))


# Nematic flow universal master scaling function g(x) = (2 / x^2) * [1 - (1 + x) * exp(-x)]
def master_function(x):
    """
    Theoretical master function:
    g(x) = (2 / x^2) * [1 - (1 + x) * exp(-x)]   (x = R_c / xi)
    For x -> 0, Taylor expansion is used for numerical stability:
    g(x) = 1 - (2/3)x + (1/4)x^2 - (1/15)x^3 + ...
    """
    x_arr = np.asarray(x, dtype=float)
    res = np.zeros_like(x_arr)
    
    small = x_arr < 1e-4
    large = ~small
    
    if np.any(small):
        xs = x_arr[small]
        res[small] = 1.0 - (2.0 / 3.0) * xs + 0.25 * (xs**2) - (xs**3) / 15.0 + (xs**4) / 72.0
    
    if np.any(large):
        xl = x_arr[large]
        res[large] = (2.0 / (xl**2)) * (1.0 - (1.0 + xl) * np.exp(-xl))
        
    return float(res) if np.ndim(x) == 0 else res


# Unified Dynamic Ising Diffusion Model:
# D(x) = 0.5 * v0^2 * [ 1 / (1 + (8 / 3pi)*x + 0.5*x^2) ] * [ (tau0 * tau_xi) / (tau_xi * g(x) + tau0) ]
def unified_diffusion(x, v0=0.207, tau0=14.00, tau_xi=3.00):
    """
    Unified Dynamic Ising Diffusion Theory:
    D(x) = (1/2) * v0^2 * S(x) * tau_corr(x)
    where:
      S(x) = 1 / (1 + (8 / (3 * pi)) * x + 0.5 * x^2)  (Padé geometric shape factor)
      tau_corr(x) = (tau0 * tau_xi) / (tau_xi * g(x) + tau0)  (Parallel relaxation time)
      g(x) = (2 / x^2) * [1 - (1 + x) * exp(-x)]
    """
    x_arr = np.asarray(x, dtype=float)
    gx = master_function(x_arr)
    s_x = 1.0 / (1.0 + (8.0 / (3.0 * np.pi)) * x_arr + 0.5 * (x_arr ** 2))
    tau_corr_x = (tau0 * tau_xi) / (tau_xi * gx + tau0)
    d_x = 0.5 * (v0 ** 2) * s_x * tau_corr_x
    return float(d_x) if np.ndim(x) == 0 else d_x