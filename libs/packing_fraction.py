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

def threshold_image(image, block_size):
    """
    画像を局所的に二値化する
    """
    threshold = threshold_local(image, block_size=block_size, method='gaussian', param=1)
    binary_image = image > threshold
    return binary_image

def threshold_images(images, block_size):
    """
    3D画像スタックを局所的に二値化する
    """
    binary_images = np.empty_like(images, dtype=bool)
    for i in tqdm(range(images.shape[0])):
        binary_images[i] = threshold_image(images[i], block_size)
    return binary_images


if __name__ == "__main__":

    scale = 0.11

    path = r'/Volumes/data/Sasaki/backup_git/MTCargo_analysis/experiment/20251226/MC00MT001/MTs.npy'
    output = path[:-4]
    #out_name = f"{output}_pf.txt"

    block_size = 10 / scale

    images = np.load(path)

    images_bin = threshold_images(images, block_size)

    fraction = get_fraction(images_bin)

    np.save(f"{output}_pf.npy", fraction)