import numpy as np
import zarr
import AFT_tools as AFT

path="/Volumes/data/Sasaki/MTsingleBeads/20260121/beads_trans_crop_crop/MTs.zarr"

print("read MTs")

za = zarr.open_array(path, read_only=True)

# AFT parameters
window_size_um = 10 # MTs length = 10um
frame = 100

scale = 0.11  # um/pixel

#### required parameters ####
window_size = int(window_size_um/scale)
overlap = 0.8
neighborhood_radius = 1

d = 30

print('calc order')

x, y, u, v, im_theta, im_eccentricity = AFT.image_local_order(za[:,:,:], window_size, overlap, save_path='', eccentricity_thresh = 0.2,
                                                             plot_overlay=False, plot_angles=False, plot_eccentricity=False,
                                                             save_figures=False)

interval = 4

im_order_parameter = AFT.calculate_order_parameter(im_theta, neighborhood_radius)

order_parameter = np.array([np.arange(0, interval*len(im_order_parameter), interval) ,im_order_parameter])

zarr.save(f"{path[:-9]}/order_parameter.zarr", order_parameter)

print('done')