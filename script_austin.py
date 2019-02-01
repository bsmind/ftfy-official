import warnings
import os
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import rescale
from skimage.feature import corner_harris, corner_peaks, corner_subpix
from scipy.interpolate import interp2d

from utils.iou_sampler import IoUSampler
from tqdm import tqdm

def get_filenames(path):
    fnames = []
    for f in os.listdir(path):
        if f.endswith('jpg'):
            fnames.append(os.path.join(path, f))
    return fnames

def get_multiscale(im, down_factors:list, debug=False):
    ms = []
    for factor in down_factors:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ms.append(rescale(im, 1/factor))
    if debug:
        for i in range(len(down_factors)):
            print('down: {:d}, shape: {}'.format(
                down_factors[i], ms[i].shape
            ))
    return ms

def get_corners(im, ox, oy, min_distance):
    mask = np.zeros_like(im)
    mask[oy:-oy, ox:-ox] = 1
    corners = corner_harris(im) * mask
    corners = corner_peaks(corners, min_distance=min_distance, threshold_rel=1e-6)
    corners = corner_subpix(im, corners, window_size=3, alpha=0.9)
    corners = corners[~np.isnan(corners).any(axis=1)]
    return corners

def aligned_resize(im, out_shape):
    #print('aligned_resize:', im.shape)
    src_h, src_w = im.shape
    dst_h, dst_w = out_shape

    x = np.arange(0, src_w, 1)
    y = np.arange(0, src_h, 1)
    f = interp2d(x, y, im)

    x = np.arange(0, src_w, src_w/dst_w)
    y = np.arange(0, src_h, src_h/dst_h)
    return f(x, y)

def get_ms_patches(im_ms, x, y, down_factors, psz_low, psz_final):
    assert len(im_ms) == len(down_factors), 'Unmatched number of multi-scale images.'
    patch_ms = []
    last_down = down_factors[-1]
    for im, factor in zip(im_ms, down_factors):
        psz = int(psz_low * (last_down / factor))
        cx = int(x // factor)
        cy = int(y // factor)
        x0 = cx - psz//2
        y0 = cy - psz//2
        patch = im[y0:y0+psz, x0:x0+psz]
        patch = aligned_resize(patch, (psz_final, psz_final))
        patch = (patch - patch.min()) / patch.ptp() # normalize
        patch_ms.append(patch)
    return patch_ms

def test_corners(corners, down_factors, psz_low, width, height):
    fcorners = []
    last_down = down_factors[-1]
    for y, x in corners:
        result = True
        for factor in down_factors:
            psz = int(psz_low * (last_down / factor))
            if x-psz < 0 or y-psz < 0 or x+psz >= width or y+psz >= height:
                result = False
                break
        if result:
            fcorners.append((y, x))
    fcorners = np.array(fcorners)
    return fcorners

def generate_info(fname, patch_id, group_id, iou_id, cx, cy, down_factor):
    return "{:s} {:d} {:d} {:d} {:.1f} {:.1f} {:d}\n".format(
        fname, patch_id, group_id, iou_id, cx, cy, down_factor
    )

def merge_patches(patches, is_horizontal=True):
    if is_horizontal:
        return np.hstack(patches)
    return np.vstack(patches)


if __name__ == '__main__':
    base_dir = '/home/sungsooha/Desktop/Data/ftfy/austin'
    data_dir = 'scene'#''campus'
    out_dir = 'scene_patch' #'campus_patch'

    debug = True
    psz_final = 64 # final patch size
    psz_low = 20 # patch size at the lowest resolution (scale)
    down_factors = [1, 2, 4, 6, 8, 10, 12, 14]
    psz_hi = psz_low * down_factors[-1]

    iou_sampler = IoUSampler((psz_hi, psz_hi))
    iou_range = [(0.7, 1.0), (0.5, 0.7), (0.3, 0.5)]

    n_max_corners = 50

    fnames = get_filenames(os.path.join(base_dir, data_dir))
    n_fnames = len(fnames)
    assert n_fnames > 0, 'No files to process in the directory: {}'.format(
        os.path.join(base_dir, data_dir)
    )

    patch_id = 0
    group_id = 0
    info_file = open(os.path.join(base_dir, out_dir, 'info.txt'), mode='w')
    def update_info(fname, gid, iou_id, x, y):
        global patch_id
        for factor in down_factors:
            info_str = generate_info(fname, patch_id, gid, iou_id, x, y, factor)
            info_file.writelines(info_str)
            patch_id += 1

    for fidx, fname in enumerate(fnames):
        print("[{:d}/{:d}] {} ...".format(fidx, n_fnames, fname))
        # read the image
        im = imread(fname, as_gray=True)
        imh, imw = im.shape
        # prepare multiscale image set
        im_ms = [im]+ get_multiscale(im, down_factors[1:])
        # detect key points
        corners = get_corners(im_ms[-1], psz_low, psz_low, 4)
        corners = corners * down_factors[-1] # map to original scale (approximate)
        corners = test_corners(corners, down_factors, psz_low, imw, imh)
        n_corners = len(corners)
        if n_corners == 0: continue

        if n_corners > n_max_corners:
            ind = np.arange(n_corners)
            np.random.shuffle(ind)
            corners = corners[ind[:n_max_corners]]

        for (y, x) in tqdm(corners, desc='Extract patch'):
            patch_ms = []
            patches = get_ms_patches(im_ms, x, y, down_factors, psz_low, psz_final)
            patch_ms.append(merge_patches(patches, True))
            update_info(fname, group_id, 0, x, y)
            for iou_id, (lo, hi) in enumerate(iou_range):
                for ox, oy in zip(*iou_sampler(lo, hi, n=4)):
                    patches = get_ms_patches(im_ms, x+ox, y+oy, down_factors, psz_low, psz_final)
                    patch_ms.append(merge_patches(patches, True))
                    update_info(fname, group_id, iou_id+1, x+ox, y+oy)
            patch_ms = merge_patches(patch_ms, False)
            patch_ms = patch_ms * 255
            patch_ms = patch_ms.astype(np.uint8)

            patch_fname = 'patchset_{:07d}.bmp'.format(patch_id)
            imsave(os.path.join(base_dir, out_dir, patch_fname), patch_ms)
            group_id += 1



    # import matplotlib.pyplot as plt
    # from matplotlib.patches import Circle

    # # visualization
    # sample_idx = 100
    # last_down = down_factors[-1]
    #
    # sample_pos = corners[sample_idx]
    # patch_ms = get_ms_patches(im_ms, sample_pos, down_factors, psz_low, psz_final)
    # fig, ax = plt.subplots(2, len(down_factors)//2)
    # ax = ax.ravel()
    # for patch, _ax in zip(patch_ms, ax):
    #     _ax.imshow(patch, cmap='gray')
    #
    # for (lo, hi) in iou_range:
    #     for ox, oy in zip(*iou_sampler(lo, hi)):
    #         x = sample_pos[1] + ox
    #         y = sample_pos[0] + oy
    #         patch_ms = get_ms_patches(im_ms, np.array([y, x]), down_factors, psz_low, psz_final)
    #         fig, ax = plt.subplots(2, len(down_factors)//2)
    #         ax = ax.ravel()
    #         for patch, _ax in zip(patch_ms, ax):
    #             _ax.imshow(patch, cmap='gray')
    #
    #
    #
    #
    # fig, ax = plt.subplots(2, len(down_factors)//2)
    # ax = ax.ravel()
    # for idx in range(len(down_factors)):
    #     ax[idx].imshow(im_ms[idx], cmap='gray')
    #     for ii, (y, x) in enumerate(corners):
    #         edgecolor = 'r' if ii == sample_idx else 'y'
    #         x = x//down_factors[idx]
    #         y = y//down_factors[idx]
    #         circ = Circle((x, y), last_down/down_factors[idx],
    #                       fill=False, edgecolor=edgecolor)
    #         ax[idx].add_patch(circ)
    #
    # plt.show()



