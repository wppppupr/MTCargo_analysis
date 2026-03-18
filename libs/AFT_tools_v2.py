import numpy as np
import pyfftw
from pyfftw.interfaces.scipy_fft import fft2, fftshift, ifft2, ifftshift, fft, ifft, fftfreq
pyfftw.interfaces.cache.enable()
pyfftw.interfaces.cache.set_keepalive_time(10.0)
import os
pyfftw.config.NUM_THREADS = int(os.environ.get('OMP_NUM_THREADS', os.cpu_count() or 1))

import skimage.io as io
import cv2                                                     # for filtering vector fields
from skimage.morphology import disk        # morphology operations
import numpy.matlib as matlib
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import mannwhitneyu
import numba as nb


@nb.njit(cache=True)
def image_norm_njit(im):
    N_rows, N_cols = im.shape
    im_norm = np.empty((N_rows, N_cols), dtype=np.float64)
    for i in range(N_rows):
        for j in range(N_cols):
            r = im[i, j].real
            i_val = im[i, j].imag
            im_norm[i, j] = np.sqrt(r*r + i_val*i_val)
    return im_norm

def image_norm(im):
    return image_norm_njit(im)

@nb.njit(cache=True)
def _compute_v(im):
    N_rows, N_cols = im.shape
    v = np.zeros((N_rows,N_cols), dtype=np.float32)
    for j in range(N_cols):
        v[0, j] = im[0, j] - im[-1, j]
        v[-1, j] = -v[0, j]
    for i in range(N_rows):
        v[i, 0] = v[i, 0] + im[i, 0] - im[i, -1]
        v[i, -1] = v[i, -1] - im[i, 0] + im[i, -1]
    return v

def periodic_decomposition(im, precomputed_filter=None):
    if im.dtype != np.float32:
        im = im.astype('float32')
    v = _compute_v(im)
    
    if precomputed_filter is None:
        N_rows, N_cols = im.shape
        fx = matlib.repmat(np.cos(2 * np.pi * np.arange(0,N_cols) / N_cols),N_rows,1)
        fy = matlib.repmat(np.cos(2 * np.pi * np.arange(0,N_rows) / N_rows),N_cols,1).T
        fx[0,0] = 0
        precomputed_filter = 0.5 / (2 - fx - fy)

    v_fft = fft2(v)
    s = np.real(ifft2(v_fft * precomputed_filter))
    p = im - s

    return p, s

@nb.njit(cache=True)
def least_moment_njit(image, xcoords, ycoords):
    N_rows, N_cols = image.shape
    M00 = 0.0
    M10 = 0.0
    M01 = 0.0
    M11 = 0.0
    M20 = 0.0
    M02 = 0.0

    for i in range(N_rows):
        for j in range(N_cols):
            val = image[i, j]
            x = xcoords[i, j]
            y = ycoords[i, j]
            M00 += val
            M10 += val * x
            M01 += val * y
            M11 += val * x * y
            M20 += val * x * x
            M02 += val * y * y

    if M00 == 0:
        return np.nan, np.nan

    xave = M10 / M00
    yave = M01 / M00

    mu20 = M20/M00 - xave**2
    mu02 = M02/M00 - yave**2
    mu11 = M11/M00 - xave*yave

    theta = 0.5 * np.arctan2((2 * mu11), (mu20 - mu02))
    theta = -1.0 * theta

    diff = mu20 - mu02
    sqrt_term = np.sqrt(4.0 * mu11**2 + diff**2)
    sum_term = 0.5 * (mu20 + mu02)

    lambda1 = sum_term + 0.5 * sqrt_term
    lambda2 = sum_term - 0.5 * sqrt_term

    if lambda1 == 0:
        eccentricity = 0.0
    else:
        var = 1.0 - lambda2/lambda1
        if var < 0:
            var = 0.0
        eccentricity = np.sqrt(var)

    return theta, eccentricity

def least_moment(image, xcoords=None, ycoords=None):
    if xcoords is None or ycoords is None:
        N_rows, N_cols = image.shape
        xcoords, ycoords = np.meshgrid(np.arange(0,N_cols) , np.arange(0,N_rows))

    return least_moment_njit(image, xcoords, ycoords)

def image_local_order(imstack, window_size = 33, overlap = 0.5, im_mask = None, intensity_thresh = 0, eccentricity_thresh = 0, 
                        plot_overlay=False, plot_angles=False, plot_eccentricity=False, save_figures=False, save_path = ''):
    
    # check if an output directory is given
    if len(save_path) > 0:
        # if directory doesn't exist, make it
        if os.path.isdir(save_path) == False:
            os.mkdir(save_path)
            
    # check to see if it's a stack of images or a single image
    if len(imstack.shape) == 2:
        imstack = np.expand_dims(imstack, axis=0)
    if im_mask is not None:
        if imstack.shape[0] == 1:
            im_mask = np.expand_dims(im_mask, axis=0)
            
    if imstack.dtype != np.float32:
        imstack = imstack.astype(np.float32)

    # get the image shape
    N_images, N_rows, N_cols = imstack.shape

    # make window size off if it isn't already
    if window_size % 2 == 0:
        window_size += 1
    
    # define the radius of the window
    radius = int(np.floor((window_size) / 2))
    
    # make a list of the r,c positions for the windows
    rpos = np.arange(radius,N_rows-radius,int(window_size * overlap))
    cpos = np.arange(radius,N_cols-radius,int(window_size * overlap))

    # make a structuring element to filter the mask
    bpass_filter = disk(radius * .5)

    # make window mask
    window_mask = np.zeros((window_size, window_size))
    window_mask[int(np.floor(window_size/2)), int(np.floor(window_size/2))] = 1

    # filter the mask with the structuring element to define the ROI
    window_mask = cv2.filter2D(window_mask, -1, bpass_filter)
    window_mask = np.rint(window_mask) == 1

    # check if there is an image mask
    if im_mask is None:
        im_mask = np.ones_like(imstack).astype('bool')
    else:
        # make sure the input mask is a boolean
        im_mask = im_mask.astype('bool')

    # make x and y coordinate matrices
    xcoords, ycoords = np.meshgrid(np.arange(0,window_size) , np.arange(0,window_size))

    # length of orientation vector
    arrow_length = radius / 2

    # make lists to hold for multiple frames
    theta_stack, ecc_stack, u_stack, v_stack = [], [], [], []

    # precomputed filter for decomposition
    fx = matlib.repmat(np.cos(2 * np.pi * np.arange(0,window_size) / window_size),window_size,1)
    fy = matlib.repmat(np.cos(2 * np.pi * np.arange(0,window_size) / window_size),window_size,1).T
    fx[0,0] = 0
    precomputed_filter = 0.5 / (2 - fx - fy)

    N_pts = len(rpos) * len(cpos)

    for frame,im in enumerate(imstack):

        x = np.empty(N_pts, dtype=np.int32)
        y = np.empty(N_pts, dtype=np.int32)
        u = np.empty(N_pts, dtype=np.float64)
        v = np.empty(N_pts, dtype=np.float64)
        im_theta = np.empty((len(rpos), len(cpos)), dtype=np.float64)
        im_ecc = np.empty((len(rpos), len(cpos)), dtype=np.float64)
        
        idx = 0
        for r_idx, r in enumerate(rpos):
            for c_idx, c in enumerate(cpos):
                x[idx] = c
                y[idx] = r

                # check to see if point is within image mask
                if im_mask[frame,r,c] == True:
                    # define the window to analyze
                    im_window = im[r-radius:r+radius+1,c-radius:c+radius+1]

                    # check that it's above the intensity threshold
                    if np.mean(im_window) > intensity_thresh:
                        # separate out the periodic and smooth components
                        im_window_periodic, im_window_smooth = periodic_decomposition(im_window, precomputed_filter)
                        # take the FFT of the periodic component
                        im_window_fft = fftshift(fft2(im_window_periodic))
                        # find the image norm and mulitply by the mask
                        im_window_fft_norm = image_norm_njit(im_window_fft) * window_mask
                        # calculate the angle and eccentricity of orientation based on the FFT moments
                        theta, eccentricity = least_moment_njit(im_window_fft_norm, xcoords, ycoords)

                        # correct for real space
                        theta = theta + np.pi/2

                        # map everything back to between -pi/2 and pi/2
                        if theta > np.pi/2:
                            theta -= np.pi

                        # filter based on eccentricity
                        if eccentricity < eccentricity_thresh:
                            eccentricity = np.nan
                            theta = np.nan

                        # add the values to array
                        im_theta[r_idx, c_idx] = theta
                        im_ecc[r_idx, c_idx] = eccentricity
                        u[idx] = np.cos(theta) * arrow_length if not np.isnan(theta) else np.nan
                        v[idx] = np.sin(theta) * arrow_length if not np.isnan(theta) else np.nan
                    else:
                        im_theta[r_idx, c_idx] = np.nan
                        im_ecc[r_idx, c_idx] = np.nan
                        u[idx] = np.nan
                        v[idx] = np.nan
                else:
                    im_theta[r_idx, c_idx] = np.nan
                    im_ecc[r_idx, c_idx] = np.nan
                    u[idx] = np.nan
                    v[idx] = np.nan

                idx += 1

        if plot_angles:
            plt.figure()
            plt.imshow(im_theta * 180 / np.pi, vmin=-90, vmax=90, cmap='hsv')
            plt.colorbar()
            plt.title('Orientation')
            plt.show()
            if save_figures:
                plt.savefig(save_path + 'angle_map_frame_%03d.tif' % (frame), format='png', dpi=300)

        if plot_eccentricity:
            plt.figure()
            plt.imshow(im_ecc, vmin=0, vmax=1)
            plt.colorbar()
            plt.title('Eccentricity')
            plt.show()
            if save_figures:
                plt.savefig(save_path + 'eccentrcitiy_map_frame_%03d.tif' % (frame), format='png', dpi=300)

        if plot_overlay:
            plt.figure()
            plt.imshow(im, cmap='Greys_r')
            plt.quiver(x,y,u,v, color='yellow', pivot='mid', scale_units='xy', scale=overlap/2, headaxislength=0, headlength=0, width=0.005)
            plt.title('Overlay')
            plt.show()
            if save_figures:
                plt.savefig(save_path + 'overlay_frame_%03d.tif' % (frame), format='png', dpi=300)

        theta_stack.append(im_theta)
        ecc_stack.append(im_ecc)
        u_stack.append(u)
        v_stack.append(v)

    # reduce dimensions if only one frame
    if N_images == 1:
        u_stack = u_stack[0]
        v_stack = v_stack[0]
        theta_stack = theta_stack[0]
        ecc_stack = ecc_stack[0]

    return x, y, u_stack, v_stack, theta_stack, ecc_stack


@nb.njit(cache=True)
def _calculate_order_parameter_loop(im_theta, neighborhood_radius):
    N_rows, N_cols = im_theta.shape
    r_start = neighborhood_radius
    r_end = N_rows - neighborhood_radius
    c_start = neighborhood_radius
    c_end = N_cols - neighborhood_radius
    
    if r_end <= r_start or c_end <= c_start:
        return np.array([np.nan])
        
    out_size = (r_end - r_start) * (c_end - c_start)
    res = np.empty(out_size, dtype=np.float64)
    res_idx = 0
    
    center_idx = (2*neighborhood_radius + 1)**2 // 2
    
    for r in range(r_start, r_end):
        for c in range(c_start, c_end):
            center_val = im_theta[r, c]
            sum_order = 0.0
            count = 0
            
            idx = 0
            for i in range(r - neighborhood_radius, r + neighborhood_radius + 1):
                for j in range(c - neighborhood_radius, c + neighborhood_radius + 1):
                    if idx != center_idx:
                        val = im_theta[i, j]
                        if not np.isnan(val):
                            sum_order += np.cos(val - center_val) ** 2 - 0.5
                            count += 1
                    idx += 1
            
            if count > 0:
                res[res_idx] = 2.0 * sum_order / count
                res_idx += 1
                
    if res_idx == 0:
        return np.array([np.nan])
    return res[:res_idx]

def calculate_order_parameter(im_theta_stack, neighborhood_radius):
    if type(im_theta_stack) == np.ndarray:
        if len(im_theta_stack.shape) == 2:
            im_theta_stack = [im_theta_stack]

    N_images = len(im_theta_stack)
    im_orderparameter_stack = []

    for im_theta in im_theta_stack:
        order_list = _calculate_order_parameter_loop(im_theta, neighborhood_radius)
        if len(order_list) > 0 and not np.isnan(order_list).all():
            im_orderparameter_stack.append(np.nanmedian(order_list))
        else:
            im_orderparameter_stack.append(np.nan)

    if N_images == 1:
        im_orderparameter_stack = im_orderparameter_stack[0]

    return im_orderparameter_stack


@nb.njit(cache=True)
def _calculate_order_parameter_heatmap_loop(im_theta, neighborhood_radius):
    N_rows, N_cols = im_theta.shape
    r_start = neighborhood_radius
    r_end = N_rows - neighborhood_radius
    c_start = neighborhood_radius
    c_end = N_cols - neighborhood_radius
    
    rpos_len = max(0, r_end - r_start)
    cpos_len = max(0, c_end - c_start)
    res = np.full((rpos_len, cpos_len), np.nan, dtype=np.float64)
    
    center_idx = (2*neighborhood_radius + 1)**2 // 2
    
    for r_idx in range(rpos_len):
        r = r_start + r_idx
        for c_idx in range(cpos_len):
            c = c_start + c_idx
            
            center_val = im_theta[r, c]
            sum_order = 0.0
            count = 0
            
            idx = 0
            for i in range(r - neighborhood_radius, r + neighborhood_radius + 1):
                for j in range(c - neighborhood_radius, c + neighborhood_radius + 1):
                    if idx != center_idx:
                        val = im_theta[i, j]
                        if not np.isnan(val):
                            sum_order += np.cos(val - center_val) ** 2 - 0.5
                            count += 1
                    idx += 1
            
            if count > 0:
                res[r_idx, c_idx] = 2.0 * sum_order / count
                
    return res

def calculate_order_parameter_heatmap(im_theta_stack, neighborhood_radius):
    if type(im_theta_stack) == np.ndarray:
        if len(im_theta_stack.shape) == 2:
            im_theta_stack = [im_theta_stack]

    N_images = len(im_theta_stack)
    im_orderparameter_stack = []

    for im_theta in im_theta_stack:
        order_list_reshape = _calculate_order_parameter_heatmap_loop(im_theta, neighborhood_radius)
        im_orderparameter_stack.append(order_list_reshape)

    if N_images == 1:
        im_orderparameter_stack = im_orderparameter_stack[0]

    return im_orderparameter_stack


def parameter_search(image_list, min_win_size, win_size_interval, overlap, plot_figure=True):
    np.seterr(divide='ignore', invalid='ignore')
    
    im = io.imread(image_list[0])
    
    max_win_size = (np.max(im.shape) -1 ) // 3
    win_size_list = np.arange(min_win_size, max_win_size, win_size_interval)
    win_size_list[win_size_list % 2 == 0] += 1

    win_size_result, image_result, order_parameter_result, neighborhood_result = [], [], [], []
    
    for image in image_list:
        im = io.imread(image).astype('float32')
        for win_size in win_size_list:
            _,_,_,_,im_theta,_ = image_local_order(im, window_size = win_size, overlap = overlap, plot_overlay = False, plot_angles=False, plot_eccentricity=False)
            
            n_windows = np.max(im_theta.shape)
            neighborhood_list = np.arange(1, ((n_windows - 1) // 2) + 1)
            
            for neighborhood in neighborhood_list:
                im_orderparameter = calculate_order_parameter(im_theta, neighborhood_radius=neighborhood)
                
                win_size_result.append(win_size)
                image_result.append(image)
                order_parameter_result.append(im_orderparameter)
                neighborhood_result.append(neighborhood)

    data_dict = {
        'window_size' : win_size_result,
        'neighborhood_radius' : neighborhood_result,
        'order_parameter' : order_parameter_result,
        'image' : image_result
    }
    
    Order_dataframe = pd.DataFrame(data_dict)  
    
    N_windows = len(win_size_list)
    neighborhood_list = np.arange(1,np.max(Order_dataframe['neighborhood_radius'])+1)
    N_neighborhood = len(neighborhood_list)

    window_neighborhood = np.empty((N_windows, N_neighborhood))
    window_neighborhood[:] = np.nan

    for i,win_size in enumerate(win_size_list):
        for j, neighborhood in enumerate(neighborhood_list):
            temp = Order_dataframe[(Order_dataframe.window_size==win_size) & (Order_dataframe.neighborhood_radius==neighborhood)]
            if len(temp) > 0:
                window_neighborhood[i,j] = np.median(temp.order_parameter)
    
    if plot_figure:
        win_size_labels = np.unique(Order_dataframe['window_size'])
        neighborhood_labels = np.unique(Order_dataframe['neighborhood_radius'])

        plt.figure()
        plt.imshow(window_neighborhood, vmin=0, vmax=1)
        plt.colorbar(orientation='horizontal')
        plt.yticks(np.arange(0,len(win_size_labels)),labels=win_size_labels)
        plt.ylabel('Window Size (px)')
        plt.xticks(np.arange(0,len(neighborhood_labels),2),labels=neighborhood_labels[::2], rotation='vertical')
        plt.xlabel('Neighbourhood Radius (px)')
        plt.title('Median order parameter')
        plt.tight_layout()
        plt.show()

    return Order_dataframe, window_neighborhood

def parameter_search_np(image_list, min_win_size, win_size_interval, overlap, plot_figure=True):
    np.seterr(divide='ignore', invalid='ignore')
    
    im = image_list[0]
    
    max_win_size = (np.max(im.shape) -1 ) // 3
    win_size_list = np.arange(min_win_size, max_win_size, win_size_interval)
    win_size_list[win_size_list % 2 == 0] += 1

    win_size_result, image_result, order_parameter_result, neighborhood_result = [], [], [], []
    
    for image in image_list:
        im = image
        for win_size in win_size_list:
            _,_,_,_,im_theta,_ = image_local_order(im, window_size = win_size, overlap = overlap, plot_overlay = False, plot_angles=False, plot_eccentricity=False)
            
            n_windows = np.max(im_theta.shape)
            neighborhood_list = np.arange(1, ((n_windows - 1) // 2) + 1)
            
            for neighborhood in neighborhood_list:
                im_orderparameter = calculate_order_parameter(im_theta, neighborhood_radius=neighborhood)
                
                win_size_result.append(win_size)
                image_result.append(image)
                order_parameter_result.append(im_orderparameter)
                neighborhood_result.append(neighborhood)

    data_dict = {
        'window_size' : win_size_result,
        'neighborhood_radius' : neighborhood_result,
        'order_parameter' : order_parameter_result,
        'image' : image_result
    }
    
    Order_dataframe = pd.DataFrame(data_dict)  
    
    N_windows = len(win_size_list)
    neighborhood_list = np.arange(1,np.max(Order_dataframe['neighborhood_radius'])+1)
    N_neighborhood = len(neighborhood_list)

    window_neighborhood = np.empty((N_windows, N_neighborhood))
    window_neighborhood[:] = np.nan

    for i,win_size in enumerate(win_size_list):
        for j, neighborhood in enumerate(neighborhood_list):
            temp = Order_dataframe[(Order_dataframe.window_size==win_size) & (Order_dataframe.neighborhood_radius==neighborhood)]
            if len(temp) > 0:
                window_neighborhood[i,j] = np.median(temp.order_parameter)
    
    if plot_figure:
        win_size_labels = np.unique(Order_dataframe['window_size'])
        neighborhood_labels = np.unique(Order_dataframe['neighborhood_radius'])

        plt.figure()
        plt.imshow(window_neighborhood, vmin=0, vmax=1)
        plt.colorbar(orientation='horizontal')
        plt.yticks(np.arange(0,len(win_size_labels)),labels=win_size_labels)
        plt.ylabel('Window Size (px)')
        plt.xticks(np.arange(0,len(neighborhood_labels),2),labels=neighborhood_labels[::2], rotation='vertical')
        plt.xlabel('Neighbourhood Radius (px)')
        plt.title('Median order parameter')
        plt.tight_layout()
        plt.show()

    return Order_dataframe, window_neighborhood

def parameter_comparison(Order_dataframe1, window_neighborhood1, Order_dataframe2, window_neighborhood2, save_figures=False, save_path = ''):
    win_size_list = sorted(np.unique(Order_dataframe1['window_size']))
    N_windows = len(win_size_list)
    neighborhood_list = sorted(np.unique(Order_dataframe1['neighborhood_radius']))
    N_neighborhood = len(neighborhood_list)

    p_median = np.empty((N_windows, N_neighborhood))
    p_median[:] = np.nan

    for i,win_size in enumerate(win_size_list):
        for j, neighborhood in enumerate(neighborhood_list):
            temp1 = Order_dataframe1[(Order_dataframe1.window_size==win_size) & (Order_dataframe1.neighborhood_radius==neighborhood)]
            temp2 = Order_dataframe2[(Order_dataframe2.window_size==win_size) & (Order_dataframe2.neighborhood_radius==neighborhood)]
            if (len(temp1) > 0) and (len(temp2) > 0):
                _, p_median[i,j] = mannwhitneyu(temp1.order_parameter, temp2.order_parameter)


    order_diff = window_neighborhood1-window_neighborhood2
    plt.figure()
    plt.imshow(order_diff, cmap='jet')
    plt.xlabel('Neighborhood (Vectors)')
    plt.xticks(np.arange(0,len(neighborhood_list),2),labels=neighborhood_list[::2], rotation='vertical')
    plt.ylabel('Window Size (px)')
    plt.yticks(np.arange(0,len(win_size_list)),labels=win_size_list)
    plt.colorbar(orientation='horizontal')
    plt.title('Order Parameter Difference (1st sample - 2nd sample)')
    plt.show()
    if save_figures:
        plt.savefig(save_path + 'parameter_search_difference.png', format='png', dpi=300)

    plt.figure()
    plt.imshow(p_median, cmap='spring_r')
    plt.xlabel('Neighborhood (Vectors)')
    plt.xticks(np.arange(0,len(neighborhood_list),2),labels=neighborhood_list[::2], rotation='vertical')
    plt.ylabel('Window Size (px)')
    plt.yticks(np.arange(0,len(win_size_list)),labels=win_size_list)
    plt.title('P-value comparison')
    plt.colorbar(orientation='horizontal')
    plt.show()
    if save_figures:
        plt.savefig(save_path + 'parameter_search_p_value.png', format='png', dpi=300)
        
    return order_diff, p_median, win_size_list, neighborhood_list
