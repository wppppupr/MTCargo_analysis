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