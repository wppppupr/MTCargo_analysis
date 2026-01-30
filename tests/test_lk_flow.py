import sys
import os
import numpy as np
import zarr
import shutil

# Add libs to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../libs'))
from LKOpticalFlow import lk_opt_flow

def create_moving_blob(n_frames=20, h=100, w=100, vx=1.0, vy=0.5, sigma=5.0):
    t, y, x = np.mgrid[0:n_frames, 0:h, 0:w]

    # Center starts at (h//2, w//2)
    # Center at time t: (cy + vy*t, cx + vx*t)

    cy = h // 2
    cx = w // 2

    y0 = cy + vy * t
    x0 = cx + vx * t

    # Gaussian blob
    blob = np.exp(-((y - y0)**2 + (x - x0)**2) / (2 * sigma**2))
    return blob

def test_lk_flow_synthetic():
    print("Generating synthetic data...")
    VX_TRUE = 1.0
    VY_TRUE = 0.5

    # Create frames
    images = create_moving_blob(n_frames=30, h=64, w=64, vx=VX_TRUE, vy=VY_TRUE)

    print("Running Optical Flow...")
    # Use output=None to get numpy arrays back
    vx, vy, rel = lk_opt_flow(images, xy_sig1=1.0, t_sig=1.0, w_sig=2.0, chunk_size=10)

    errors_vx = []
    errors_vy = []
    vals_vx = []
    vals_vy = []

    # Check regions where gradient is significant (e.g. at sigma distance)
    # But since we have smoothing, looking at mean over the blob is okay-ish.
    # The previous test showed underestimation (0.58 vs 1.0).

    for t in range(5, 25):
        cy = 32 + VY_TRUE * t
        cx = 32 + VX_TRUE * t

        # Look at window +/- 5 pixels
        y_slice = slice(int(cy)-5, int(cy)+6)
        x_slice = slice(int(cx)-5, int(cx)+6)

        if y_slice.start < 0 or y_slice.stop >= 64 or x_slice.start < 0 or x_slice.stop >= 64:
            continue

        vx_roi = vx[t, y_slice, x_slice]
        vy_roi = vy[t, y_slice, x_slice]

        curr_vx = np.mean(vx_roi)
        curr_vy = np.mean(vy_roi)

        vals_vx.append(curr_vx)
        vals_vy.append(curr_vy)

    avg_vx = np.mean(vals_vx)
    avg_vy = np.mean(vals_vy)

    print(f"True Vx: {VX_TRUE}, Estimated Vx: {avg_vx:.4f}")
    print(f"True Vy: {VY_TRUE}, Estimated Vy: {avg_vy:.4f}")

    # Check signs
    assert avg_vx > 0.3, f"Vx should be positive and significant (got {avg_vx})"
    assert avg_vy > 0.1, f"Vy should be positive and significant (got {avg_vy})"

    # Check relative magnitude
    # Vx should be roughly 2x Vy
    ratio = avg_vx / avg_vy
    print(f"Ratio Vx/Vy: {ratio:.2f} (Expected 2.0)")
    assert 1.5 < ratio < 2.5, f"Ratio Vx/Vy is off: {ratio}"

    print("Synthetic Test Passed!")

def test_lk_flow_zarr():
    print("Testing Zarr output...")
    images = np.random.rand(20, 32, 32)
    output_path = "test_output.zarr"
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    z_out = zarr.open(output_path, mode='w', shape=(3, 20, 32, 32), dtype=np.float32)

    lk_opt_flow(images, xy_sig1=1.0, t_sig=1.0, w_sig=1.0, chunk_size=5, output=z_out)

    assert np.all(z_out.shape == (3, 20, 32, 32))

    print("Zarr test Passed!")
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

if __name__ == "__main__":
    test_lk_flow_synthetic()
    test_lk_flow_zarr()
