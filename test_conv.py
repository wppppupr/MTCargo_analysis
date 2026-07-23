import numpy as np
import time
import cv2

def main():
    H, W = 1024, 1024
    img = np.random.randn(H, W).astype(np.float32)
    
    t0 = time.time()
    for radius in range(5, 1000, 50):
        # OpenCV standard dev approximation
        # For large radii, GaussianBlur might be slow or it might use FFT? Let's see
        sigma = float(radius)
        ksize = int(6 * sigma + 1)
        if ksize % 2 == 0:
            ksize += 1
            
        try:
            blur = cv2.GaussianBlur(img, (ksize, ksize), sigma)
        except Exception as e:
            print("Error at", radius, e)
            
    print("OpenCV time:", time.time() - t0)

if __name__ == '__main__':
    main()
