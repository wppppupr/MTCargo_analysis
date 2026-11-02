import numpy as np
import pandas as pd
import zarr
import os
import shutil
import sys

# Add libs to path to import local_polar_range
sys.path.append(os.path.join(os.path.dirname(__file__), '../libs'))
try:
    from local_polar_range import local_polar_range
except ImportError:
    # If not running from tests dir, try adjusting path
    sys.path.append(os.path.join(os.path.dirname(__file__), 'libs'))
    from local_polar_range import local_polar_range

# Mock data generation
def create_mock_data():
    # 10 frames, 100x100 image
    T, H, W = 10, 100, 100

    # Create flow: constant flow (1, 0) everywhere
    flow = np.zeros((T-1, H, W, 2), dtype=np.float32)
    flow[..., 0] = 1.0 # x-velocity = 1

    # Save as zarr
    if os.path.exists('test_flow_range.zarr'):
        shutil.rmtree('test_flow_range.zarr')

    # Using zarr.open_array or zarr.open
    flow_array = zarr.open('test_flow_range.zarr', mode='w', shape=flow.shape, dtype=flow.dtype)
    flow_array[:] = flow

    # Create tracks
    # 5 particles, present in all frames
    tracks_data = []
    for t in range(T):
        for p in range(5):
            tracks_data.append({
                'particle': p,
                'frame': t,
                'x': 50 + p, # Center of image
                'y': 50 + p
            })
    tracks = pd.DataFrame(tracks_data)

    return flow_array, tracks

def test_local_polar_range():
    print("Generating mock data...")
    flow_array, tracks = create_mock_data()

    scale = 1.0
    ranges = [5, 10, 20] # Test multiple ranges

    print(f"Testing local_polar_range with ranges: {ranges}")
    tracks_new = local_polar_range(flow_array, tracks, ranges, scale)

    # Check if new columns exist
    for r in ranges:
        col_name = f'local_P_{r}'
        if col_name not in tracks_new.columns:
            print(f"FAILURE: Column {col_name} missing.")
            return

        Ps = tracks_new[col_name].to_numpy()

        # Check values
        # Frame 0 should be NaN (because flow[t-1] accessed)
        # Frame >= 1 should be 1.0

        # Helper to get mask for frame > 0
        valid_mask = tracks_new['frame'] > 0

        Ps_valid = Ps[valid_mask]

        if np.allclose(Ps_valid, 1.0):
            print(f"SUCCESS: Range {r} - Local polar parameter is 1.0 for uniform flow.")
        else:
            print(f"FAILURE: Range {r} - Local polar parameter is not 1.0.")
            print(Ps_valid)
            return

    print("All tests passed!")

    # Cleanup
    if os.path.exists('test_flow_range.zarr'):
        shutil.rmtree('test_flow_range.zarr')

if __name__ == "__main__":
    test_local_polar_range()
