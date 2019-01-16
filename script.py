import numpy as np
import network.dataset.sem_dataset as sem
from utils.utils import get_fullpath, load_image

from skimage.transform import resize
from scipy.signal import convolve2d
import matplotlib.pyplot as plt


def compute_local_std_map(img, block_size):
    '''
    Compute standard devition of local region
    Args:
        img: (h x w) ndarray
        block_size: tuple, (height x width) of a local block, should be odd number

    Returns:
        standard deviation map
    '''
    # even block size will be treated as odd block size
    hblkh = block_size[0] // 2
    hblkw = block_size[1] // 2

    height, width = img.shape

    stdmap = np.zeros(img.shape, dtype=np.float32)
    for r in range(hblkh, height - hblkh, 1):
        for c in range(hblkw, width - hblkw, 1):
            patch = img[
                r - hblkh: r + hblkh,
                c - hblkw: c + hblkw
            ]
            stdmap[r,c] = np.std(patch)

    return stdmap

def compute_local_var_map(img, block_size):
    img2 = img**2
    ones = np.ones(img.shape)

    kernel = np.ones(block_size)
    s = convolve2d(img, kernel, mode="same")
    s2 = convolve2d(img2, kernel, mode="same")
    ns = convolve2d(ones, kernel, mode="same")

    return (s2 - s**2 / ns) / ns





data_dir = '/home/sungsooha/Desktop/Data/ftfy/data_hxnsem_selected'
patch_size = (208, 208)

f_by_zoom = sem.filelist_by_zoomfactor(data_dir)
for zoom, f_list in f_by_zoom.items():
    fullpath = get_fullpath(data_dir, f_list[0])
    img = load_image(fullpath)

    rimg = resize(img, (512, 512))
    print(zoom)
    print(img.shape)
    #stdmap = compute_local_std_map(img, patch_size)
    stdmap = compute_local_var_map(rimg, (21, 21))
    stdmap[stdmap < 0] = 0
    stdmap = np.sqrt(stdmap)
    stdmap = resize(stdmap, img.shape)
    stdmap = (stdmap - stdmap.min()) / stdmap.ptp()
    stdmap[:patch_size[0], :] = 0
    stdmap[-patch_size[0]:, :] = 0
    stdmap[:, :patch_size[1]] = 0
    stdmap[:, -patch_size[1]:] = 0
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(img, vmin=0, vmax=1)
    ax[1].imshow(stdmap)
    ax[2].imshow(stdmap > 0.1)
    #ax[2].hist(stdmap.flatten(), bins=100)
    #ax[1, 1].imshow(stdmap > minstd*1.5)
    plt.show()
    #break