import os
import numpy as np
from skimage.io import imsave
from utils.data import get_multiscale, get_interpolator, get_filenames
from utils.iou_sampler import IoUSampler

from tqdm import tqdm

def get_ms_patches(im_f, down_factors, orig_cx, orig_cy, orig_side, patch_size):
    patches = []
    for factor, f in zip(down_factors, im_f):
        cx = orig_cx / factor
        cy = orig_cy / factor
        side = orig_side / factor

        step = side / patch_size
        xx = np.arange(cx - side / 2, cx + side / 2 + step, step)
        yy = np.arange(cy - side / 2, cy + side / 2 + step, step)

        patch = f(xx, yy)
        patch = patch[:patch_size, :patch_size]
        patch = (patch - patch.min()) / patch.ptp()
        patches.append(patch)
    return patches

class PatchExtractor(object):
    '''Extract patches at a given region of interests at multi scales'''
    def __init__(self, down_factors:list, iou_ranges:list=None, patch_size=128, min_patch_size=13):
        self.patch_size = patch_size
        self.min_patch_size = min_patch_size
        self.down_factors = down_factors
        self.iou_ranges = iou_ranges
        #self.iou_sampler = IoUSampler(patch_size)

        self.box = None        # int, (cx, cy, side)
        self.patches = []      # np.ndarray
        self.actual_down_factors = []
        self.patches_iou = []
        self.boxes_iou = []

    def clear(self):
        self.box = None        # int, (cx, cy, side)
        self.patches = []      # np.ndarray
        self.actual_down_factors = []
        self.patches_iou = []
        self.boxes_iou = []

    def get_patches(self):
        return self.patches

    def get_patches_iou(self):
        return self.patches_iou

    def get_patches_with_info(self):
        return self.patches, self.actual_down_factors, self.box

    def get_patches_iou_with_info(self):
        return self.patches_iou, self.actual_down_factors, self.boxes_iou

    def extract_ms_patches(self, im, box, with_iou=0):
        """
        Extract multi-scaled patches
        Args:
            im: image at original scale
            box: float, (x1, y1, x2, y2)
            with_iou: extract patches in different iou ranges if with_iou > 0
        Returns:
            list of multi-scaled patches
        """
        x1, y1, x2, y2 = box
        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1
        orig_cx = int(np.round((x1 + x2)/2))
        orig_cy = int(np.round((y1 + y2)/2))
        orig_side = max(int(np.round(x2 - x1)), int(np.round(y2 - y1)))

        fit_down_factor = 1.
        if orig_side > self.patch_size:
            fit_down_factor *= orig_side/self.patch_size

        down_factors = [int(np.round(fit_down_factor*factor)) for factor in self.down_factors]
        im_f  = [get_interpolator(_im) for _im in get_multiscale(im, down_factors)]
        patches = get_ms_patches(im_f, down_factors,
                                 orig_cx, orig_cy, orig_side, self.patch_size)

        # todo: how to use single iou_sampler to support various sizes
        iou_sampler = IoUSampler((orig_side, orig_side))

        patches_iou = []
        boxes_iou = []
        if with_iou > 0 and self.iou_ranges is not None:
            for (lo, hi) in self.iou_ranges:
                for ox, oy in zip(*iou_sampler(lo, hi, n=with_iou)):
                    patches_iou += get_ms_patches(im_f, down_factors,
                                        orig_cx+ox, orig_cy+oy, orig_side, self.patch_size)
                    boxes_iou.append((orig_cx+ox, orig_cy+oy, orig_side))

        self.patches_iou = patches_iou
        self.boxes_iou = boxes_iou

        self.box = (orig_cx, orig_cy, orig_side)
        self.patches = patches
        self.actual_down_factors = down_factors
        return patches

class PatchDataManager(object):
    def __init__(self, output_dir,
                 patches_per_col=10, patches_per_row=10,
                 info_fname='info.txt'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.output_dir = output_dir
        self.patch_size = output_dir
        self.patches_per_col = patches_per_col
        self.patches_per_row = patches_per_row

        self.patches_row = []
        self.patches_col = []

        fnames = get_filenames(output_dir, 'bmp')
        self.counter = len(fnames)

        self.file = open(os.path.join(self.output_dir, info_fname), 'w')

    def _save_image(self, im, prefix='patchset'):
        im *= 255
        im = im.astype(np.uint8)
        fname = '{:s}_{:06d}.bmp'.format(prefix, self.counter)
        imsave(os.path.join(self.output_dir, fname), im)
        self.counter += 1

    def add(self, patches:list, info:list):
        assert len(patches) == len(info), 'Not matched numbers in patch list and info list.'
        for patch, _info in zip(patches, info):
            self.patches_col.append(patch)
            self.file.write(_info)

            if len(self.patches_col) == self.patches_per_col:
                self.patches_row.append(np.hstack(self.patches_col))
                self.patches_col = []

            if len(self.patches_row) == self.patches_per_row:
                im = np.vstack(self.patches_row)
                self._save_image(im)
                self.patches_row = []
        self.file.flush()

    def add_patches(self, patches:list):
        for patch in patches:
            self.patches_col.append(patch)

            if len(self.patches_col) == self.patches_per_col:
                self.patches_row.append(np.hstack(self.patches_col))
                self.patches_col = []

            if len(self.patches_row) == self.patches_per_row:
                im = np.vstack(self.patches_row)
                self._save_image(im)
                self.patches_row = []

    def add_info(self, info:list):
        for _info in info:
            self.file.write(_info)
        self.file.flush()

    def dump(self):
        if len(self.patches_col) > 0:
            while len(self.patches_col) < self.patches_per_col:
                self.patches_col.append(
                    np.zeros_like(self.patches_col[0])
                )
            self.patches_row.append(np.hstack(self.patches_col))
            self.patches_col = []

        if len(self.patches_row) > 0:
            while len(self.patches_row) > self.patches_per_row:
                im = np.vstack(self.patches_row[:self.patches_per_row])
                self._save_image(im)
                self.patches_row = self.patches_row[self.patches_per_row:]

        if len(self.patches_row) > 0:
            while len(self.patches_row) < self.patches_per_row:
                self.patches_row.append(
                    np.zeros_like(self.patches_row[0])
                )
            im = np.vstack(self.patches_row)
            self._save_image(im)
            self.patches_row = []







