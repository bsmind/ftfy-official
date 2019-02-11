import warnings
import os
import numpy as np
from skimage.transform import downscale_local_mean
from skimage.transform import resize as sk_resize
from skimage.feature import blob_dog
from skimage.filters import gaussian as gaussian_filter
from skimage.io import imread
from tqdm import tqdm
from scipy.signal import convolve2d

def resize(img, output_shape, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = sk_resize(img, output_shape, **kwargs)
    return img

def downsample(image, out_shape, factors):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        image = downscale_local_mean(image, factors)
        image = resize(image, out_shape, order=1, mode='reflect', preserve_range=True)
    return image

def random_downsample(image, factors, p=0.7):
    blk = np.random.choice(factors)
    if blk == 1:
        return image

    if len(image.shape) == 3:
        blk = (blk, blk, 1)
        shape = image.shape[:3]
    else:
        blk = (blk, blk)
        shape = image.shape

    image = downsample(image, shape, blk)
    return image

def normalize(img):
    """Normalize an image"""
    img = (img - np.mean(img)) / np.std(img)
    return (img - np.min(img)) / np.ptp(img)

def load_image(fullpath, crop=True):
    """Load an image"""
    #todo: crop the image, if and only if it is necessary.
    #img = imread(fullpath, as_gray=True)
    img = imread(fullpath, as_gray=True)
    if crop:
        return normalize(img[:-110, :])
    return normalize(img)

def get_fullpath(datadir, filename):
    if filename.startswith(os.sep):
        filename = filename[len(os.sep):]
    return os.path.join(datadir, filename)

def get_image_list(data_dir, file_list):
    images = []
    for f in tqdm(file_list, desc='Load Image'):
        img = load_image(get_fullpath(data_dir, f))
        img = np.expand_dims(img, axis=-1)

        if img.shape[0] != 850 or img.shape[1] != 1280:
            #print(f, img.shape)
            continue

        images.append(img)
    return images

def compute_local_var_map(im, block_size):
    im2 = im**2
    ones = np.ones(im.shape)
    kernel = np.ones(block_size)
    s = convolve2d(im, kernel, mode='same')
    s2 = convolve2d(im2, kernel, mode='same')
    ns = convolve2d(ones, kernel, mode='same')
    return (s2 - s**2 / ns) / ns

def compute_local_std_map(im):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rim = resize(im, (512, 512))

    stdmap = compute_local_var_map(rim, (13, 13))
    stdmap[stdmap < 0] = 0
    stdmap = np.sqrt(stdmap)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stdmap = resize(stdmap, im.shape)

    stdmap = (stdmap - stdmap.min()) / stdmap.ptp()
    return stdmap



