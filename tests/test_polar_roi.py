import numpy as np
import pandas as pd
import zarr
import os
import shutil
import sys

# Add libs to path to import polar_roi
sys.path.append(os.path.join(os.path.dirname(__file__), '../libs'))
try:
    from polar_roi import calculate_polar_roi
except ImportError:
    # If not running from tests dir, try adjusting path
    sys.path.append(os.path.join(os.path.dirname(__file__), 'libs'))
    from polar_roi import calculate_polar_roi

# Mock data generation
def create_mock_data():
    # 10 frames, 100x100 image
    T, H, W = 10, 100, 100

    # Create flow: constant flow (1, 0) everywhere
    flow = np.zeros((T-1, H, W, 2), dtype=np.float32)
    flow[..., 0] = 1.0 # x-velocity = 1

    # Save as zarr
    if os.path.exists('test_flow_roi.zarr'):
        shutil.rmtree('test_flow_roi.zarr')

    # Using zarr.open_array or zarr.open
    flow_array = zarr.open('test_flow_roi.zarr', mode='w', shape=flow.shape, dtype=flow.dtype)
    flow_array[:] = flow

    return flow_array

def test_polar_roi():
    print("Generating mock data...")
    flow_array = create_mock_data()

    # Test ROI
    x_start, y_start, w, h = 10, 10, 20, 20

    print(f"Testing calculate_polar_roi for ROI: ({x_start}, {y_start}, {w}, {h})...")
    Ps = calculate_polar_roi(flow_array, x_start, y_start, w, h)

    # Check results
    # Frame 0 should be NaN (initialized).
    # Frame >= 1 should be 1.0 (uniform flow).

    if np.isnan(Ps[0]):
        print("SUCCESS: Ps[0] is NaN.")
    else:
        print("FAILURE: Ps[0] is not NaN.")

    Ps_valid = Ps[1:]

    if np.allclose(Ps_valid, 1.0):
        print("SUCCESS: ROI polar parameter is 1.0 for uniform flow.")
    else:
        print("FAILURE: ROI polar parameter is not 1.0.")
        print(Ps_valid)

    # Test invalid ROI
    print("Testing invalid ROI...")
    try:
        Ps_invalid = calculate_polar_roi(flow_array, 200, 200, 50, 50)
        # Should return array of NaNs or handle gracefully
        # My implementation returns Ps initialized with NaNs
        if np.all(np.isnan(Ps_invalid)):
             print("SUCCESS: Invalid ROI returned NaNs.")
        else:
             print("FAILURE: Invalid ROI returned non-NaNs.")
    except Exception as e:
        print(f"Exception caught for invalid ROI: {e}")

    # Cleanup
    if os.path.exists('test_flow_roi.zarr'):
        shutil.rmtree('test_flow_roi.zarr')

if __name__ == "__main__":
    test_polar_roi()
