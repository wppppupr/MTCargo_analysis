import numpy as np
from skimage.filters import threshold_local
from tqdm import tqdm

def get_fraction(images):
    """
    各フレームのpacking fraction（二値画像の1の割合）を高速に計算
    """
    box = images.shape[1] * images.shape[2]
    # 3次元配列の各フレームごとに合計を計算し、全体で割る
    pf_array = images.reshape(images.shape[0], -1).sum(axis=1) / box
    return pf_array

if __name__ == "__main__":

    scale = 0.11

    path = r'/Volumes/My Passport/Sasaki/MTsingleBeads/20241114/T2_4uM_mc03_beads_MT_smoothed.npy'
    output = path[:-4]
    out_name = f"{output}_pf.txt"

    block_size = 10 / scale

    images = np.load(path)

    images_bin = np.empty_like(images, dtype=bool)
    for i in tqdm(range(images.shape[0])):
        threshold = threshold_local(images[i], block_size=block_size, method='gaussian', param=1)
        images_bin[i] = images[i] > threshold

    fraction = get_fraction(images_bin)

    pf_mean = np.mean(fraction)
    pf_var = np.std(fraction) ** 2

    # pf_meanをtxtに保存
    with open(out_name, "w") as f:
        f.write(f"{pf_mean}, {pf_var}")