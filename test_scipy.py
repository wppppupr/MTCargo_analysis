import numpy as np
import time
from scipy.ndimage import gaussian_filter

def main():
    H, W = 1024, 1024
    img = np.random.randn(H, W).astype(np.float32)
    
    t0 = time.time()
    for radius in range(5, 1000, 50):
        sigma = float(radius)
        blur = gaussian_filter(img, sigma)
    print("Scipy time:", time.time() - t0)

if __name__ == '__main__':
    main()
