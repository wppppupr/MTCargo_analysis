import numpy as np
import time
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def calc_numba_no_lut(bx, by, b_dx, b_dy, flow_u, flow_v, radius, particle_radius):
    N = len(bx)
    H, W = flow_u.shape
    mean_dot = np.zeros(N)
    sigma2 = 2 * radius * radius
    pr2 = particle_radius * particle_radius
    x_mins = np.maximum(0, np.floor(bx - 3*radius)).astype(np.int32)
    x_maxs = np.minimum(W, np.ceil(bx + 3*radius) + 1).astype(np.int32)
    y_mins = np.maximum(0, np.floor(by - 3*radius)).astype(np.int32)
    y_maxs = np.minimum(H, np.ceil(by + 3*radius) + 1).astype(np.int32)
    
    for i in prange(N):
        x, y = bx[i], by[i]
        dx, dy = b_dx[i], b_dy[i]
        x_min, x_max = x_mins[i], x_maxs[i]
        y_min, y_max = y_mins[i], y_maxs[i]
        w_sum = dot_sum = 0.0
        
        for yy in range(y_min, y_max):
            dy_sq = (yy - y) * (yy - y)
            for xx in range(x_min, x_max):
                u, v = flow_u[yy, xx], flow_v[yy, xx]
                if np.isnan(u) or np.isnan(v): continue
                dist_sq = (xx - x)*(xx - x) + dy_sq
                if particle_radius > 0 and dist_sq <= pr2: continue
                
                w = np.exp(-dist_sq / sigma2)
                w_sum += w
                dot_sum += w * (dx * u + dy * v)
        if w_sum > 0: mean_dot[i] = dot_sum / w_sum
    return mean_dot

@njit(parallel=True, fastmath=True)
def calc_numba_lut(bx, by, b_dx, b_dy, flow_u, flow_v, radius, particle_radius):
    N = len(bx)
    H, W = flow_u.shape
    mean_dot = np.zeros(N)
    sigma2 = 2 * radius * radius
    pr2 = particle_radius * particle_radius
    x_mins = np.maximum(0, np.floor(bx - 3*radius)).astype(np.int32)
    x_maxs = np.minimum(W, np.ceil(bx + 3*radius) + 1).astype(np.int32)
    y_mins = np.maximum(0, np.floor(by - 3*radius)).astype(np.int32)
    y_maxs = np.minimum(H, np.ceil(by + 3*radius) + 1).astype(np.int32)
    
    max_dist_sq = int(2.0 * (3.0 * radius + 1.0)**2) + 5
    exp_lut = np.zeros(max_dist_sq, dtype=np.float64)
    for d2 in range(max_dist_sq):
        exp_lut[d2] = np.exp(-d2 / sigma2)
        
    for i in prange(N):
        x, y = bx[i], by[i]
        dx, dy = b_dx[i], b_dy[i]
        x_min, x_max = x_mins[i], x_maxs[i]
        y_min, y_max = y_mins[i], y_maxs[i]
        w_sum = dot_sum = 0.0
        
        for yy in range(y_min, y_max):
            dy_sq = (yy - y) * (yy - y)
            for xx in range(x_min, x_max):
                u, v = flow_u[yy, xx], flow_v[yy, xx]
                if np.isnan(u) or np.isnan(v): continue
                dist_sq = (xx - x)*(xx - x) + dy_sq
                if particle_radius > 0 and dist_sq <= pr2: continue
                
                dist_sq_int = int(dist_sq + 0.5)
                w = exp_lut[dist_sq_int]
                
                w_sum += w
                dot_sum += w * (dx * u + dy * v)
        if w_sum > 0: mean_dot[i] = dot_sum / w_sum
    return mean_dot

def main():
    H, W = 1024, 1024
    N = 1000
    flow_u = np.random.randn(H, W).astype(np.float32)
    flow_v = np.random.randn(H, W).astype(np.float32)
    bx = np.random.uniform(0, W, N)
    by = np.random.uniform(0, H, N)
    b_dx = np.random.randn(N)
    b_dy = np.random.randn(N)
    radius = 100
    particle_radius = 5.0
    
    calc_numba_no_lut(bx[:2], by[:2], b_dx[:2], b_dy[:2], flow_u, flow_v, radius, particle_radius)
    calc_numba_lut(bx[:2], by[:2], b_dx[:2], b_dy[:2], flow_u, flow_v, radius, particle_radius)
    
    t0 = time.time()
    res1 = calc_numba_no_lut(bx, by, b_dx, b_dy, flow_u, flow_v, radius, particle_radius)
    print("No LUT:", time.time() - t0)
    
    t0 = time.time()
    res2 = calc_numba_lut(bx, by, b_dx, b_dy, flow_u, flow_v, radius, particle_radius)
    print("LUT:", time.time() - t0)
    
    print("Diff:", np.abs(res1 - res2).max())

if __name__ == '__main__':
    main()
