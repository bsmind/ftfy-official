import os
import numpy as np

from skimage.io import imread

import matplotlib.pyplot as plt

base_dir = '/home/sungsooha/Desktop/Data/ftfy/data_hxnsem_selected'
data_dir = 'AgVOPO_20170601/3'

# ----------------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------------
fnames = []
for f in os.listdir(os.path.join(base_dir, data_dir)):
    if f.endswith('tif'):
        fnames.append(f)
        #fnames.append(os.path.join(base_dir, data_dir, f))
fnames = sorted(fnames)

images = []
for f in fnames:
    im = imread(os.path.join(base_dir, data_dir, f), as_gray=True)
    im = im * 255.
    im = im.astype(np.uint8)
    images.append(im)

# ----------------------------------------------------------------------------
# Load resolution information
# ----------------------------------------------------------------------------
resol_path = os.path.join(base_dir, data_dir, 'resol.txt')
if not os.path.exists(resol_path):
    file = open(resol_path, 'w')
    fig, ax = plt.subplots()
    h = ax.imshow(images[0], cmap='gray')
    plt.ion()
    plt.show()
    for fname, im in zip(fnames, images):
        h.set_data(im)
        ax.set_title(fname)
        plt.pause(0.001)

        scale_bar = im == 255
        _, cols = np.where(scale_bar)
        bar_width = cols.max() - cols.min() + 1
        bar_size = int(input('[{:s}] bar size: '.format(fname)))

        print('File: ', fname)
        print('Bar width  (px): ', bar_width)
        print('Bar size   (um): ', bar_size)
        print('Pixel size (um): ', bar_size / bar_width)
        file.write("{:s} {:d} {:d}\n".format(fname, bar_width, bar_size))
    file.close()
    plt.ioff()
    plt.close(fig)

# todo: load



