import numpy as np
from scipy.ndimage import convolve
import time
import cv2

def create_circular_kernel(size):
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size / 2.0 - 0.5
    y, x = np.ogrid[:size, :size]
    mask = (x - center)**2 + (y - center)**2 <= (size / 2.0)**2
    kernel[mask] = 1.0
    kernel /= kernel.sum()
    return kernel

size = 100
kernel = create_circular_kernel(size)
print(f"Kernel sum: {kernel.sum()}")
print(f"Kernel shape: {kernel.shape}")

m_ux = np.random.rand(512, 512).astype(np.float32)

t0 = time.time()
res1 = convolve(m_ux, kernel, mode='reflect')
t1 = time.time()
print(f"Scipy convolve time: {t1-t0}")

t0 = time.time()
res2 = cv2.filter2D(m_ux, -1, kernel, borderType=cv2.BORDER_REFLECT)
t1 = time.time()
print(f"cv2 filter2D time: {t1-t0}")

print(f"Diff: {np.abs(res1 - res2).max()}")
