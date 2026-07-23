import numpy as np
import time
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def calc_numba_swapped(bx, by, b_dx, b_dy, flow_u, flow_v, radius, particle_radius):
    N = len(bx)
    H, W = flow_u.shape
    
    w_sum = np.zeros(N)
    dot_sum = np.zeros(N)
    cos_sum = np.zeros(N)
    u_sum = np.zeros(N)
    v_sum = np.zeros(N)
    cos_sq_sum = np.zeros(N)
    
    sigma = float(radius)
    sigma2 = 2 * sigma * sigma
    pr2 = particle_radius * particle_radius
    
    b_norm = np.sqrt(b_dx*b_dx + b_dy*b_dy)
    
    for yy in prange(H):
        w_sum_local = np.zeros(N)
        dot_sum_local = np.zeros(N)
        cos_sum_local = np.zeros(N)
        u_sum_local = np.zeros(N)
        v_sum_local = np.zeros(N)
        cos_sq_sum_local = np.zeros(N)
        
        for xx in range(W):
            u = flow_u[yy, xx]
            v = flow_v[yy, xx]
            
            if np.isnan(u) or np.isnan(v):
                continue
                
            flow_norm = np.sqrt(u*u + v*v)
            has_flow = flow_norm > 0
            
            for i in range(N):
                x = bx[i]
                y = by[i]
                
                # Bounding box check
                if abs(xx - x) > 3*radius or abs(yy - y) > 3*radius:
                    continue
                    
                dist_sq = (xx - x)*(xx - x) + (yy - y)*(yy - y)
                
                if particle_radius > 0 and dist_sq <= pr2:
                    continue
                    
                w = np.exp(-dist_sq / sigma2)
                
                w_sum_local[i] += w
                u_sum_local[i] += w * u
                v_sum_local[i] += w * v
                
                dx = b_dx[i]
                dy = b_dy[i]
                dot = dx * u + dy * v
                dot_sum_local[i] += w * dot
                
                bn = b_norm[i]
                if bn > 0 and has_flow:
                    cos_val = dot / (bn * flow_norm)
                    cos_sum_local[i] += w * cos_val
                    cos_sq_sum_local[i] += w * cos_val * cos_val
                    
        for i in range(N):
            w_sum[i] += w_sum_local[i]
            dot_sum[i] += dot_sum_local[i]
            cos_sum[i] += cos_sum_local[i]
            u_sum[i] += u_sum_local[i]
            v_sum[i] += v_sum_local[i]
            cos_sq_sum[i] += cos_sq_sum_local[i]

    mean_dot = np.zeros(N)
    mean_cos = np.zeros(N)
    mean_u = np.zeros(N)
    mean_v = np.zeros(N)
    cos_stds = np.zeros(N)
    
    for i in range(N):
        if w_sum[i] > 0:
            mean_dot[i] = dot_sum[i] / w_sum[i]
            mean_u[i] = u_sum[i] / w_sum[i]
            mean_v[i] = v_sum[i] / w_sum[i]
            m_cos = cos_sum[i] / w_sum[i]
            mean_cos[i] = m_cos
            
            cos_var = (cos_sq_sum[i] / w_sum[i]) - (m_cos * m_cos)
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
    
    particle_radius = 5.0
    
    # Compile
    calc_numba_swapped(bx[:2], by[:2], b_dx[:2], b_dy[:2], flow_u, flow_v, 10, particle_radius)
    
    t0 = time.time()
    for radius in range(5, 1000, 50):
        calc_numba_swapped(bx, by, b_dx, b_dy, flow_u, flow_v, radius, particle_radius)
    print("Numba swapped total time:", time.time() - t0)
    
if __name__ == '__main__':
    main()
