import h5py
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
import tempfile

def calc_strain_rate(input_h5, output_h5, max_frames=None):
    input_h5 = Path(input_h5)
    output_h5 = Path(output_h5)
    
    # ネットワークドライブの書き込み遅延を回避するため、一度ローカルの一時ファイルに保存します
    temp_dir = tempfile.gettempdir()
    temp_output = Path(temp_dir) / f"temp_{output_h5.name}"
    
    print(f'Calculating extension rate, strain rate and vorticity for: {input_h5}')
    
    h5_in = None
    h5_out = None
    try:
        # HDF5のチャンクキャッシュを100MBに増やして読み書きを高速化 (rdcc_nbytes)
        h5_in = h5py.File(input_h5, 'r', rdcc_nbytes=1024**2 * 100)
        if 'flows' not in h5_in:
            print("Error: 'flows' dataset not found in the input file.")
            return
            
        flows = h5_in['flows']
        num_frames, _, h, w = flows.shape
        
        if max_frames is not None:
            num_frames = min(num_frames, max_frames)
            
        print(f"Writing temporarily to local disk: {temp_output}")
        h5_out = h5py.File(temp_output, 'w', rdcc_nbytes=1024**2 * 100)
        
        # 伸長率 (Extension rate) = 第一主ひずみ速度 (Maximum principal strain rate)
        dset_ext = h5_out.create_dataset('extension_rate', 
                                         shape=(num_frames, h, w), 
                                         dtype=np.float16, 
                                         chunks=(1, h, w),
                                         compression='lzf')
        # ひずみ速度 (Strain rate) = 最大せん断ひずみ速度 (Maximum shear strain rate)
        dset_shear = h5_out.create_dataset('strain_rate', 
                                           shape=(num_frames, h, w), 
                                           dtype=np.float16, 
                                           chunks=(1, h, w),
                                           compression='lzf')
        # 渦度 (Vorticity)
        dset_vort = h5_out.create_dataset('vorticity', 
                                          shape=(num_frames, h, w), 
                                          dtype=np.float16, 
                                          chunks=(1, h, w),
                                          compression='lzf')
                                           
        for i in tqdm(range(num_frames), desc="Computing Rates"):
            flow = flows[i] # shape: (2, h, w)
            u = flow[0].astype(np.float32)
            v = flow[1].astype(np.float32)
            
            # numpy.gradient returns (dy, dx) since image shape is (h, w)
            u_y, u_x = np.gradient(u)
            v_y, v_x = np.gradient(v)
            
            # ひずみ速度テンソルの成分
            e_xx = u_x
            e_yy = v_y
            e_xy = 0.5 * (u_y + v_x)
            
            # 主ひずみ速度の計算 (Mohrの応力円に相当する計算)
            center = (e_xx + e_yy) / 2.0
            radius = np.sqrt(((e_xx - e_yy) / 2.0)**2 + e_xy**2)
            
            extension_rate = center + radius  # 第一主ひずみ速度
            strain_rate = radius              # 最大せん断ひずみ速度
            vorticity = v_x - u_y             # 渦度
            
            dset_ext[i] = extension_rate.astype(np.float16)
            dset_shear[i] = strain_rate.astype(np.float16)
            dset_vort[i] = vorticity.astype(np.float16)
            
        # h5pyのオブジェクト参照が残っているとclose時にエラーになることがあるため明示的に削除
        del flows, dset_ext, dset_shear, dset_vort
        
    finally:
        if h5_out is not None:
            try:
                h5_out.close()
            except Exception as e:
                pass
        if h5_in is not None:
            try:
                h5_in.close()
            except Exception as e:
                pass

    # ローカルの一時ファイルをネットワークドライブ等の最終目的地へ移動
    if temp_output.exists():
        print(f"Moving file to destination: {output_h5}")
        # 出力先ディレクトリが存在しない場合は作成
        output_h5.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(temp_output), str(output_h5))
        print(f'Done. Saved to {output_h5}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate extension rate, strain rate and vorticity from optical flow h5.')
    parser.add_argument('input_file', type=str, help='Path to the input H5 file containing optical flow.')
    parser.add_argument('output_file', type=str, help='Path to the output H5 file.')
    parser.add_argument('--max_frames', type=int, default=None, help='Maximum number of frames to process (for testing).')
    args = parser.parse_args()

    calc_strain_rate(args.input_file, args.output_file, args.max_frames)
