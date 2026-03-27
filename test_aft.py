import numpy as np
import libs.AFT_tools_v2 as AFT
import time

# Create dummy image stack (T, Y, X) = (4, 100, 100)
np.random.seed(42)
imstack = np.random.rand(4, 100, 100).astype(np.float32)

t0 = time.time()
x, y, u, v, im_theta, im_ecc = AFT.image_local_order(imstack, window_size=21, overlap=0.5, n_jobs=2)
t1 = time.time()

print("Parallel execution time:", t1 - t0)
print("Shapes:")
print("x:", x.shape)
print("im_theta len:", len(im_theta))
print("Test passed successfully!")
