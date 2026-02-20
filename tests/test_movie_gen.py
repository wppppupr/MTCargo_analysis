import os
import shutil
import tempfile
import zarr
import numpy as np
import pandas as pd
import sys

# Ensure libs can be imported
sys.path.append(os.getcwd())

from libs.movie import create_movie

def test_movie_generation():
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    print(f"Created temp dir: {temp_dir}")

    try:
        # Create dummy zarr files
        # Shape: (Time, Y, X) = (10, 100, 100)
        shape = (10, 100, 100)

        # MTs
        mts = zarr.open(os.path.join(temp_dir, "MTs.zarr"), mode='w', shape=shape, dtype='uint8')
        mts[:] = np.random.randint(0, 255, size=shape)

        # MTs_red
        mts_red = zarr.open(os.path.join(temp_dir, "MTs_red.zarr"), mode='w', shape=shape, dtype='uint8')
        mts_red[:] = np.random.randint(0, 255, size=shape)

        # beads
        beads = zarr.open(os.path.join(temp_dir, "beads.zarr"), mode='w', shape=shape, dtype='uint8')
        beads[:] = np.random.randint(0, 255, size=shape)

        # Create dummy csv
        # columns: particle, frame, x, y
        # Create one particle track
        frames = np.arange(10)
        x = np.linspace(10, 90, 10)
        y = np.linspace(10, 90, 10)
        particle = np.zeros(10, dtype=int)

        df = pd.DataFrame({
            "particle": particle,
            "frame": frames,
            "x": x,
            "y": y
        })
        df.to_csv(os.path.join(temp_dir, "beads_tracks.csv"), index=False)

        # Run create_movie
        output_file = os.path.join(temp_dir, "tracking.mov")
        print(f"Generating movie at {output_file}...")
        create_movie(temp_dir, output_name=output_file)

        # Check if file exists
        if os.path.exists(output_file):
            print(f"SUCCESS: Movie file created at {output_file}")
            # Check size > 0
            if os.path.getsize(output_file) > 0:
                print("SUCCESS: Movie file is not empty.")
            else:
                print("FAILURE: Movie file is empty.")
        else:
            print("FAILURE: Movie file was not created.")

    except Exception as e:
        print(f"FAILURE: An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        print(f"Removed temp dir: {temp_dir}")

if __name__ == "__main__":
    test_movie_generation()
