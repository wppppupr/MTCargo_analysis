import h5py
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
import tempfile
import os

def calc_strain_rate(input_h5, output_h5, max_frames=None):
    input_h5 = Path(input_h5)
    output_h5 = Path(output_h5)
    
    print(f'Calculating extension rate, strain rate and vorticity for: {input_h5}')
    

    # 一意な一時ファイル名を生成するため delete=False を使用
    # 例外時は後段で明示的に削除する

    with h5py.File(input_h5, 'r', rdcc_nbytes=1024**2 * 100) as h5_in:
        if 'flows' not in h5_in:
            print("Error: 'flows' dataset not found in the input file.")
            return
            
        flows = h5_in['flows']
        num_frames, _, h, w = flows.shape

        if flows.ndim != 4:
            raise ValueError(
                f"Expected shape (frames,2,h,w), got {flows.shape}"
            )

        if flows.shape[1] != 2:
            raise ValueError(
                f"Expected second dimension=2, got {flows.shape}"
            )

        if max_frames is not None:
            num_frames = min(num_frames, max_frames)
            
        # 一時ファイルのディレクトリを指定して作成 (delete=False で自動削除を無効化し、後でmoveする)
        with tempfile.NamedTemporaryFile(dir=tempfile.gettempdir(), suffix=f"_{output_h5.name}", delete=False) as tmp_file:
            temp_output = Path(tmp_file.name)
            
        try:
            print(f"Writing temporarily to local disk: {temp_output}")
            with h5py.File(temp_output, 'w', rdcc_nbytes=1024**2 * 100) as h5_out:
                
                dset_ext = h5_out.create_dataset('extension_rate', shape=(num_frames, h, w), dtype=np.float16, chunks=(1, h, w), compression='lzf')
                dset_shear = h5_out.create_dataset('strain_rate', shape=(num_frames, h, w), dtype=np.float16, chunks=(1, h, w), compression='lzf')
                dset_vort = h5_out.create_dataset('vorticity', shape=(num_frames, h, w), dtype=np.float16, chunks=(1, h, w), compression='lzf')
                                                   
                for i in tqdm(range(num_frames), desc="Computing Rates"):
                    flow = flows[i]  # shape: (2, h, w)
                    u = flow[0].astype(np.float32)
                    v = flow[1].astype(np.float32)
                    
                    u_y, u_x = np.gradient(u, edge_order=2)
                    v_y, v_x = np.gradient(v, edge_order=2)

                    
                    e_xx = u_x
                    e_yy = v_y
                    e_xy = 0.5 * (u_y + v_x)
                    
                    center = (e_xx + e_yy) / 2.0
                    
                    tmp = ((e_xx - e_yy) / 2.0)**2 + e_xy**2
                    radius = np.sqrt(tmp)
                    
                    extension_rate = center + radius
                    strain_rate = radius
                    vorticity = v_x - u_y
                    
                    dset_ext[i] = extension_rate.astype(np.float16)
                    dset_shear[i] = strain_rate.astype(np.float16)
                    dset_vort[i] = vorticity.astype(np.float16)
            
            # すべての書き込みとクローズが正常に終了したら、目的地へ移動
            print(f"Moving file to destination: {output_h5}")
            output_h5.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(temp_output), str(output_h5))
            print(f'Done. Saved to {output_h5}')
            
        except Exception as e:
            # エラーが発生した場合は、ローカルの一時ファイルを確実に削除する
            if temp_output.exists():
                os.remove(temp_output)
            print(f"Error occurred during computation: {e}")
            raise e


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate extension rate, strain rate and vorticity from optical flow h5.')
    parser.add_argument('input_file', type=str, help='Path to the input H5 file containing optical flow.')
    parser.add_argument('output_file', type=str, help='Path to the output H5 file.')
    parser.add_argument('--max_frames', type=int, default=None, help='Maximum number of frames to process (for testing).')
    args = parser.parse_args()

    calc_strain_rate(args.input_file, args.output_file, args.max_frames)
