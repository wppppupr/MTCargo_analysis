import numpy as np
import time
from numba import njit, prange

@njit(parallel=True)
def calc_numba(bx, by, b_dx, b_dy, flow_u, flow_v, radius, particle_radius):
    N = len(bx)
    H, W = flow_u.shape
    
    mean_dot = np.zeros(N)
    mean_cos = np.zeros(N)
    mean_u = np.zeros(N)
    mean_v = np.zeros(N)
    cos_stds = np.zeros(N)
    
    sigma = float(radius)
    sigma2 = 2 * sigma * sigma
    pr2 = particle_radius * particle_radius
    
    for i in prange(N):
        x = bx[i]
        y = by[i]
        dx = b_dx[i]
        dy = b_dy[i]
        
        x_min = max(0, int(np.floor(x - 3*radius)))
        x_max = min(W, int(np.ceil(x + 3*radius)) + 1)
        y_min = max(0, int(np.floor(y - 3*radius)))
        y_max = min(H, int(np.ceil(y + 3*radius)) + 1)
        
        b_norm = np.sqrt(dx*dx + dy*dy)
        
        w_sum = 0.0
        dot_sum = 0.0
        cos_sum = 0.0
        u_sum = 0.0
        v_sum = 0.0
        
        # To compute cos_var later, we need sum of w * cos^2
        cos_sq_sum = 0.0
        
        for yy in range(y_min, y_max):
            for xx in range(x_min, x_max):
                u = flow_u[yy, xx]
                v = flow_v[yy, xx]
                
                if np.isnan(u) or np.isnan(v):
                    continue
                    
                dist_sq = (xx - x)**2 + (yy - y)**2
                
                if particle_radius > 0 and dist_sq <= pr2:
                    continue
                    
                w = np.exp(-dist_sq / sigma2)
                
                w_sum += w
                u_sum += w * u
                v_sum += w * v
                
                dot = dx * u + dy * v
                dot_sum += w * dot
                
                if b_norm > 0:
                    flow_norm = np.sqrt(u*u + v*v)
                    if flow_norm > 0:
                        cos_val = dot / (b_norm * flow_norm)
                        cos_sum += w * cos_val
                        cos_sq_sum += w * cos_val * cos_val
                        
        if w_sum > 0:
            mean_dot[i] = dot_sum / w_sum
            mean_u[i] = u_sum / w_sum
            mean_v[i] = v_sum / w_sum
            m_cos = cos_sum / w_sum
            mean_cos[i] = m_cos
            
            cos_var = (cos_sq_sum / w_sum) - (m_cos * m_cos)
            if cos_var > 0:
                cos_stds[i] = np.sqrt(cos_var)
            else:
                cos_stds[i] = 0.0

    return mean_dot, mean_cos, mean_u, mean_v, cos_stds

def main():
    H, W = 1024, 1024
    N = 1000
    flow_u = np.random.randn(H, W).astype(np.float32)
    flow_v = np.random.randn(H, W).astype(np.float32)
    bx = np.random.uniform(0, W, N)
    by = np.random.uniform(0, H, N)
    b_dx = np.random.randn(N)
    b_dy = np.random.randn(N)
    
    radius = 200
    particle_radius = 5.0
    
    # Compile
    calc_numba(bx[:2], by[:2], b_dx[:2], b_dy[:2], flow_u, flow_v, radius, particle_radius)
    
    t0 = time.time()
    calc_numba(bx, by, b_dx, b_dy, flow_u, flow_v, radius, particle_radius)
    print("Numba Time:", time.time() - t0)
    
if __name__ == '__main__':
    main()
