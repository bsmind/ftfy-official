import warnings
import os
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import rescale
from skimage.feature import corner_harris, corner_peaks, corner_subpix
from scipy.interpolate import interp2d

from utils.iou_sampler import IoUSampler
from tqdm import tqdm



def get_filenames(path, ext='jpg'):
    fnames = []
    for f in os.listdir(path):
        if f.endswith(ext):
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
            print('down: {}, shape: {}'.format(
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

def get_interpolator(im, fill_value=0):
    """values outside of input image domain will be extrapolated if fill_value is None"""
    h, w = im.shape
    x = np.arange(0, w, 1)
    y = np.arange(0, h, 1)
    return interp2d(x, y, im, fill_value=fill_value)

def get_ms_patches(im_f, x, y, down_factors, psz_low, psz_final):
    assert len(im_f) == len(down_factors), 'Unmatched number of multiscales.'
    patch_ms = []
    last_down = down_factors[-1]
    for f, factor in zip(im_f, down_factors):
        psz = psz_low * (last_down / factor)
        cx = x / factor
        cy = y / factor

        x0 = cx - psz/2
        x1 = cx + psz/2
        y0 = cy - psz/2
        y1 = cy + psz/2
        step = psz / psz_final

        xx = np.arange(x0, x1, step)
        yy = np.arange(y0, y1, step)
        patch = f(xx, yy)
        patch = patch[:psz_final, :psz_final]
        assert patch.shape == (psz_final, psz_final), 'Wrong patch dimension as %s' % patch.shape
        patch = (patch - patch.min()) / patch.ptp()
        patch_ms.append(patch)

    return patch_ms

def test_corners(corners, down_factors, psz_low, width, height):
    fcorners = []
    last_down = down_factors[-1]
    for y, x in corners:
        result = True
        for factor in down_factors:
            psz = int(psz_low * (last_down / factor))
            if x-psz-1 < 0 or y-psz-1 < 0 or x+psz+1 >= width or y+psz+1 >= height:
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

def create_ms_patchset(
        base_dir, data_dir,
        psz_low, down_factors, iou_range, out_dir=None, n_max_corners=50, psz_final=None
):
    '''
    Create multi-scaled image patch dataset

    Args:
        base_dir: absolute path to base directory
        data_dir: name of image data directory in the base_dir
        psz_low: patch size at the lowest resolution
        down_factors: list of down-scale factors
        iou_range: list of tuple (low, high), IoU ranges
        out_dir: name of patchset directory, if None, `{data_dir}_patch`
        n_max_corners: maximum nember of corners to be considered
                        if the number of detected corners is less than n_max_corners, use them all.
    '''
    if psz_final is None:
        psz_final = psz_low * down_factors[-1] # final patch size

    iou_sampler = IoUSampler((psz_final, psz_final))

    fnames = get_filenames(os.path.join(base_dir, data_dir))
    n_fnames = len(fnames)
    assert n_fnames > 0, 'No files to process in the directory: {}'.format(
        os.path.join(base_dir, data_dir)
    )

    if out_dir is None:
        out_dir = data_dir + '_patch'
    if not os.path.exists(os.path.join(base_dir, out_dir)):
        os.makedirs(os.path.join(base_dir, out_dir))

    image_counter = 0
    patch_id = 0
    group_id = 0

    info_file = None
    info_file = open(os.path.join(base_dir, out_dir, 'info.txt'), mode='w')
    def update_info(fname, pid, gid, iou_id, x, y):
        for factor in down_factors:
            info_str = generate_info(fname, pid, gid, iou_id, x, y, factor)
            info_file.writelines(info_str)
            pid += 1
        return pid

    for fidx, fname in enumerate(fnames):
        # read the image
        im = imread(fname, as_gray=True)
        imh, imw = im.shape
        # prepare multiscale image set
        im_ms = [im]+ get_multiscale(im, down_factors[1:])
        # prepare 2d interpolator
        im_f = [get_interpolator(_im) for _im in im_ms]
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

        pbar = tqdm(corners)
        for (y, x) in pbar:
            pbar.set_description('[{:d}/{:d}] Extract patch'.format(fidx+1, n_fnames))
            patch_ms = []
            # extract multi-scaled patches at the corner
            patches = get_ms_patches(im_f, x, y, down_factors, psz_low, psz_final)
            patch_ms.append(merge_patches(patches, True))
            patch_id = update_info(fname, patch_id, group_id, 0, x, y)
            # extract patches at randomly selected corners based on IoU range
            for iou_id, (lo, hi) in enumerate(iou_range):
                for ox, oy in zip(*iou_sampler(lo, hi, n=4)):
                    patches = get_ms_patches(im_f, x+ox, y+oy, down_factors, psz_low, psz_final)
                    patch_ms.append(merge_patches(patches, True))
                    patch_id = update_info(fname, patch_id, group_id, iou_id+1, x+ox, y+oy)

            # create a group of patches
            patch_ms = merge_patches(patch_ms, False)
            patch_ms = patch_ms * 255
            patch_ms = patch_ms.astype(np.uint8)
            # make output for a group of patches
            patch_fname = 'patchset_{:07d}.bmp'.format(patch_id)
            imsave(os.path.join(base_dir, out_dir, patch_fname), patch_ms)
            group_id += 1
        image_counter += 1

    print('Image dataset     : %s' % data_dir)
    print('# images          : %s' % n_fnames)
    print('# processed images: %s' % image_counter)
    print('# patches         : %s' % patch_id)
    print('# groups          : %s' % group_id)
    print('patch size        : %s' % psz_final)

def load_info_for_triplet(base_dir, data_dir, fname='info.txt'):
    info_fname = os.path.join(base_dir, data_dir, fname)
    assert os.path.isfile(info_fname), 'Cannot find `info.txt`.'

    gIDs = list()    # list of group IDs
    datamap = dict() # {gid}_{iou_id} --> [pid]

    with open(info_fname, 'r') as f:
        for line in f:
            tokens = line.split()
            fname, pid, gid, iouid, cx, cy, factor = tokens

            pID = int(pid)
            gID = int(gid)
            iouID = int(iouid)

            if gID not in gIDs:
                gIDs.append(gID)

            key = '{:d}_{:d}'.format(gID, iouID)
            if datamap.get(key, None) is None:
                datamap[key] = [pID]
            else:
                datamap[key].append(pID)

    return gIDs, datamap

def generate_triplet_samples(
        output_dir,
        data_info,
        n_samples,
        gid_include,
        fname_prefix='triplet'
):
    gIDs, datamap, pid_to_fid = data_info

    # IoU set
    # set A(0): extracted from key points
    # set B(1): 0.7 <= IoU < 1.0 w.r.t A
    # set C(2): 0.5 <= IoU < 0.7 w.r.t A
    # set D(3): 0.3 <= IoU < 0.5 w.r.t A

    fname = os.path.join(output_dir, '{:s}.txt'.format(fname_prefix))
    file = open(fname, 'w')

    def update_file(idx_a, idx_p, idx_n):
        file.writelines('{:d} {:d} {:d}\n'.format(idx_a, idx_p, idx_n))

    def get_key(gid, iou_id):
        return '{:d}_{:d}'.format(gid, iou_id)

    m1, m2 = 0, 0

    for _ in tqdm(range(n_samples), desc='Generating triplet samples'):
        is_valid = False
        while not is_valid:
            gid = np.random.choice(gIDs)
            gid_n = gid
            idx_a = np.random.choice(datamap[get_key(gid, 0)])
            idx_p = idx_a
            idx_n = idx_a

            method_id = -1

            # method I: within a patch group
            # - anchor /in {A}
            # - positive /in {A, B, C} except the anchor
            # - negative /in {B, C, D} except the set where the positive is originated
            if np.random.random() < 0.5:
                used_patches = [idx_a]
                p_iou = 0
                idx_p = idx_a
                while idx_p in used_patches:
                    p_iou = np.random.choice([0, 1, 2])
                    idx_p = np.random.choice(datamap[get_key(gid, p_iou)])

                used_patches.append(idx_p)
                idx_n = idx_p
                while idx_n in used_patches:
                    n_iou = np.random.choice(np.arange(p_iou+1, 4, 1, dtype=np.int32))
                    idx_n = np.random.choice(datamap[get_key(gid_n, n_iou)])

                method_id = 1
                m1 += 1

            # method II: over patch groups
            # - anchor /in {A}
            # - positive /in {A, B} except the anchor
            # - negative /in {A', B'} of other key points
            else:
                idx_p = idx_a
                while idx_p == idx_a:
                    p_iou = np.random.choice([0, 1])
                    idx_p = np.random.choice(datamap[get_key(gid, p_iou)])

                while gid_n == gid:
                    gid_n = np.random.choice(gIDs)
                n_iou = np.random.choice([0, 1])
                idx_n = np.random.choice(datamap[get_key(gid_n, n_iou)])

                method_id = 2
                m2 += 1

            if gid in gid_include and gid_n in gid_include:
                update_file(idx_a, idx_p, idx_n)
                is_valid = True
            else:
                if method_id == 1: m1 -= 1
                if method_id == 2: m2 -= 1

    print("Method 1: %s", m1)
    print("Method 2: %s", m2)

    file.close()

def generate_matched_pairs(
        output_dir,
        data_info,
        n_samples,
        gid_include,
        fname_prefix='matched'
):
    gIDs, datamap, pid_to_fid = data_info

    # IoU set
    # set A(0): extracted from key points
    # set B(1): 0.7 <= IoU < 1.0 w.r.t A
    # set C(2): 0.5 <= IoU < 0.7 w.r.t A
    # set D(3): 0.3 <= IoU < 0.5 w.r.t A

    fname = os.path.join(output_dir, '{:s}.txt'.format(fname_prefix))
    file = open(fname, 'w')

    def update_file(idx_1, idx_2, is_matched):
        file.writelines('{:d} {:d} {:d}\n'.format(idx_1, idx_2, is_matched))

    def get_key(gid, iou_id):
        return '{:d}_{:d}'.format(gid, iou_id)

    for _ in tqdm(range(n_samples), desc='Generating matched pairs'):
        # matched if two patches from either A or B in the same group
        gid = np.random.choice(gIDs)
        while gid not in gid_include:
            gid = np.random.choice(gIDs)
        idx_1 = np.random.choice(datamap[get_key(gid, 0)])
        idx_2 = idx_1
        while idx_1 == idx_2:
            iou_id = np.random.choice([0, 1])
            idx_2 = np.random.choice(datamap[get_key(gid, iou_id)])
        update_file(idx_1, idx_2, 1)

        # unmatched if two patches from different groups
        gid_1, gid_2 = np.random.choice(gIDs, 2, replace=False)
        while gid_1 not in gid_include or gid_2 not in gid_include or gid_1 == gid_2:
            gid_1, gid_2 = np.random.choice(gIDs, 2, replace=False)

        iou_1 = np.random.choice([0, 1])
        iou_2 = np.random.choice([0, 1])
        idx_1 = np.random.choice(datamap[get_key(gid_1, iou_1)])
        idx_2 = np.random.choice(datamap[get_key(gid_2, iou_2)])
        update_file(idx_1, idx_2, 0)

    file.close()

def generate_image_retrieval_samples(
        output_dir,
        data_info,
        gid_include,
        n_query_per_group=1,
        fname_prefix='retrieval'
):
    gIDs, datamap, pid_to_fid = data_info

    # IoU set
    # set A(0): extracted from key points
    # set B(1): 0.7 <= IoU < 1.0 w.r.t A
    # set C(2): 0.5 <= IoU < 0.7 w.r.t A
    # set D(3): 0.3 <= IoU < 0.5 w.r.t A

    fname = os.path.join(output_dir, '{:s}.txt'.format(fname_prefix))
    file = open(fname, 'w')

    def update_file(idx, label, is_query):
        file.writelines('{:d} {:s} {:d}\n'.format(idx, label, is_query))

    def get_key(gid, iou_id):
        return '{:d}_{:d}'.format(gid, iou_id)

    for gid in tqdm(gIDs, desc='Generating retrieval test set'):
        if gid not in gid_include:
            continue

        used_pid = []
        label_q = '{:d}'.format(gid)
        for _ in range(n_query_per_group):
            # randomly choose a patch for query
            idx_q = np.random.choice(datamap[get_key(gid, 0)])
            while idx_q in used_pid:
                idx_q = np.random.choice(datamap[get_key(gid, 0)])
            used_pid.append(idx_q)
            update_file(idx_q, label_q, 1)

            # randomly choose a patch expected to be retrieved
            idx_r = idx_q
            while idx_r in used_pid:
                iou = np.random.choice([0, 1])
                idx_r = np.random.choice(datamap[get_key(gid, iou)])
            used_pid.append(idx_r)
            update_file(idx_r, label_q, 0)

            # randomly choose a patch for dummy
            idx_d = idx_r
            iou = 2
            while idx_d in used_pid:
                iou = np.random.choice([2, 3])
                idx_d = np.random.choice(datamap[get_key(gid, iou)])
            used_pid.append(idx_d)
            label_d = '{:d}_{:d}_dummy'.format(gid, iou)
            update_file(idx_d, label_d, 0)

    file.close()
