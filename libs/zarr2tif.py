import zarr
import tifffile
from pathlib import Path
from tqdm import tqdm

# パス定義 (pathlibを使用することでOS差分を吸収しやすくします)
base_path = Path('/media/sasaki/myssd/Sasaki/MTSingleBeads')
folder = base_path / '20260121/exp_crop1'
zarr_path = folder / 'MTs.zarr'
output_tiff_path = folder / 'MTs.tif'

# Zarrファイルの読み込み
# mode='r' で読み込み専用として開くことで安全性を確保
try:
    data = zarr.open(str(zarr_path), mode='r')
except FileNotFoundError:
    print(f"Error: ファイルが見つかりません: {zarr_path}")
    exit()

# 時間軸（フレーム数）の取得
t = data.shape[0]

print(f"Processing: {zarr_path}")
print(f"Shape: {data.shape}, Dtype: {data.dtype}")

# TiffWriterを使用した書き込み
# metadata引数を追加することで、ImageJなどで開いた際の情報を付与可能です（必要に応じて）
with tifffile.TiffWriter(output_tiff_path, bigtiff=True) as tif:
    for i in tqdm(range(t), desc="Converting to TIFF"):
        # フレームデータを取得
        frame = data[i]
        
        # データの書き込み
        # method: save -> write に変更
        # photometric: 白黒画像であることを明示 ('minisblack')
        tif.write(
            frame, 
            compression='zlib',
            photometric='minisblack' 
        )

print("Conversion Completed.")