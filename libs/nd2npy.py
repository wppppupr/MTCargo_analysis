import nd2
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import exposure
import os


def process_nd2_file(file_path: str,
                     diameter: float,
                     scale: float = 0.11,
                     equalize: bool = True,
                     mt_sigma=(0, 1, 1),
                     beads_sigma=(0, 2, 2),
                     save: bool = True,
                     out_dir: str | None = None):
    """
    nd2ファイルを読み込み、ヒストグラム平坦化・平滑化を行い、結果を保存して返します。

    Args:
        file_path: nd2ファイルのパス
        diameter: ビーズの直径（um）
        scale: ピクセルあたりのum（デフォルト: 0.11）
        equalize: 各フレームに対してヒストグラム平坦化を行うか
        mt_sigma: MTチャンネルへのgaussian_filterのsigma
        beads_sigma: ビーズチャンネルへのgaussian_filterのsigma
        save: 結果をnp.saveで保存するか
        out_dir: 保存先ディレクトリ（Noneなら入力ファイルと同じ場所）

    Returns:
        tuple: (MTs_smoothed, beads_smoothed, output_name)
    """

    file = nd2.imread(file_path)
    comp = 255 / 4095
    file_comp = file * comp

    MTs = file_comp[:, 0, :, :]
    beads = file_comp[:, 1, :, :]

    pxdiameter = int(diameter / scale)
    if pxdiameter % 2 == 0:
        pxdiameter += 1

    if equalize:
        MTs_eq_list = [exposure.equalize_hist(mt) for mt in MTs]
        MTs_eq = np.array(MTs_eq_list)
    else:
        MTs_eq = MTs.copy()

    # 背景データのスムージング
    MTs_smoothed = gaussian_filter(MTs_eq, sigma=mt_sigma)
    beads_smoothed = gaussian_filter(beads, sigma=beads_sigma)

    output_name = file_path[:-4]
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.basename(output_name)
        output_name = os.path.join(out_dir, base)

    if save:
        os.makedirs(out_dir, exist_ok=True)
        np.save(f"experiment/{out_dir}/{output_name}_MT_smoothed", MTs_smoothed)
        np.save(f"experiment/{out_dir}/{output_name}_beads_smoothed", beads_smoothed)

    return MTs_smoothed, beads_smoothed, output_name



# 例: 関数を呼び出す
if __name__ == "__main__":
    # デフォルトのファイルパスと直径（必要に応じて変更）
    file_path = r'/Volumes/data/Sasaki/MTsingleBeads/20251210/MC03_4uM.nd2'
    diameter = 1.18

    MTs_smoothed, beads_smoothed, output_name = process_nd2_file(file_path, diameter, equalize=True, save=True, out_dir="20251210")
    print(f"Saved outputs with base: {output_name}")