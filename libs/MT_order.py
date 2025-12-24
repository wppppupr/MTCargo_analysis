import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import AFT_tools as AFT

from skimage.exposure import equalize_hist
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from skimage.filters import threshold_otsu

def crop_around_beads(traj_solo, image, frame, d, scale, cm_image = 'gray', c_beads = (255, 0, 0) ,display = bool):
    """
    traj_solo : single particles tarjectory
    image:array image
    d: crop range(um)

    return cropped image
    """
    pos = np.array([traj_solo[traj_solo['frame']==frame]['x'], traj_solo[traj_solo['frame']==frame]['y']])
    pos_int = np.floor(pos).astype(int)

    distance = np.floor(d/scale).astype(int)
    crop = np.array([pos_int - distance, pos_int+distance]).reshape(2,2)
    
    croppedimage=image[frame][crop[0][0]:crop[1][0], crop[0][1]:crop[1][1]]
    cropped_flatten = equalize_hist(croppedimage)

    if display == True:
        markersize = 1.18/scale
        s = markersize ** 2
        vmin1, vmax1 = image.min(), image.max() * 0.2
        fig, ax = plt.subplots()

        ax.imshow(croppedimage, cmap=cm_image, interpolation='none', aspect='auto', vmin=vmin1, vmax=vmax1)
        ax.scatter(distance, distance, s=s, color=c_beads)

        size_bar = AnchoredSizeBar(ax.transData,
                               size=1/scale,  # スケールバーの長さ（データ座標）50um
                               label='',  # スケールバーのラベル
                               loc=4,  # 右下に配置
                               pad=0.5,
                               color='white',  # スケールバーの色
                               frameon=False,  # フレームなし
                               size_vertical=5,  # スケールバーの太さ
                               fontproperties=fm.FontProperties(size=12))  # フォントサイズ
        ax.add_artist(size_bar)
        ax.axis('off')

    return cropped_flatten

def crop_around_trajectory(traj_solo, image, d, scale):

    cropped_ims = []

    for frame in traj_solo['frame']:
        croppedim = crop_around_beads(traj_solo, image, frame, d, scale)
        cropped_ims.append(croppedim)

    shapes = [im.shape for im in cropped_ims]
    most_common_shape = max(set(shapes), key=shapes.count)

    # 形状が一致するものだけを保持
    filtered_cropped_ims = [im for im in cropped_ims if im.shape == most_common_shape]

    # 配列化
    cropped_ims_array = np.array(filtered_cropped_ims)
    mask = cropped_ims_array.mean()

    return cropped_ims_array, mask

def get_order(tracks, particle, image_stacks, d, scale,  window_size, overlap, neighborhood_radius, eccentricity_thresh=0.2, cm_image='gray', c_beads=(255, 0, 0)):
    traj_solo = tracks[tracks['particle']==particle]
    croppedims_array, mask = crop_around_trajectory(traj_solo, image_stacks, d = d, scale=scale)
    x, y, u, v, im_theta, im_eccentricity = AFT.image_local_order(croppedims_array, window_size, overlap, intensity_thresh=mask, eccentricity_thresh=eccentricity_thresh)
    im_order_parameter = AFT.calculate_order_parameter(im_theta, neighborhood_radius)
    im_order_parameter_array=np.array(im_order_parameter)

    return im_order_parameter_array