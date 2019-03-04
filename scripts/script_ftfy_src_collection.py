import os
import numpy as np

from skimage.io import imread
from utils.data import get_interpolator, get_multiscale
from utils.sem_data import PatchDataManager
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PATCHES_PER_ROW = 10
PATCHES_PER_COL = 10

def load_sem_info(base_dir, data_dir, fname='info.txt'):
    info_path = os.path.join(base_dir, data_dir, 'patches', fname)
    assert os.path.exists(info_path), 'Cannot find a file: %s' % info_path

    fname_to_gid = dict()
    gid_info = dict()
    pid_info = dict()

    with open(info_path, 'r') as f:
        for line in f:
            tokens = line.split()

            img_fname = tokens[0]
            pid = int(tokens[1])
            gid = int(tokens[2])
            iouid = int(tokens[3])
            cx = int(tokens[4])
            cy = int(tokens[5])
            side = int(tokens[6])
            factor = int(tokens[7])

            x0 = cx - side/2
            x1 = cx + side/2
            y0 = cy - side/2
            y1 = cy + side/2

            if fname_to_gid.get(img_fname, None) is None:
                fname_to_gid[img_fname] = []
            if gid not in fname_to_gid[img_fname]:
                fname_to_gid[img_fname].append(gid)

            pid_info[pid] = (cx, cy, side, factor, iouid)

            if gid_info.get(gid, None) is None:
                gid_info[gid] = dict(
                    pids=[],
                    factors=[],
                    x0=np.inf,
                    x1=-np.inf,
                    y0=np.inf,
                    y1=-np.inf
                )
            gid_info[gid]['pids'].append(pid)

            if factor not in gid_info[gid]['factors']:
                gid_info[gid]['factors'].append(factor)

            gid_info[gid]['x0'] = min(x0, gid_info[gid].get('x0'))
            gid_info[gid]['x1'] = max(x1, gid_info[gid].get('x1'))
            gid_info[gid]['y0'] = min(y0, gid_info[gid].get('y0'))
            gid_info[gid]['y1'] = max(y1, gid_info[gid].get('y1'))

    return fname_to_gid, gid_info, pid_info

def load_info(base_dir, data_dir, fname='info.txt', base_dir_file=None):
    info_path = os.path.join(base_dir, data_dir, fname)
    assert os.path.exists(info_path), 'Cannot find a file: %s' % info_path

    fname_to_gid = dict()
    gid_info = dict()
    pid_info = dict()

    with open(info_path, 'r') as f:
        for line in f:
            tokens = line.split()

            img_fname = tokens[0]
            pid = int(tokens[1])
            gid = int(tokens[2])
            iouid = int(tokens[3])
            cx = float(tokens[4])
            cy = float(tokens[5])
            factor = int(tokens[6])
            side = 130#13 * (10 / factor)
            #psz = psz_low * (last_down / factor)

            x0 = cx - side/2
            x1 = cx + side/2
            y0 = cy - side/2
            y1 = cy + side/2

            if base_dir_file is not None:
                img_fname = os.path.join(base_dir_file, os.path.basename(img_fname))

            if fname_to_gid.get(img_fname, None) is None:
                fname_to_gid[img_fname] = []
            if gid not in fname_to_gid[img_fname]:
                fname_to_gid[img_fname].append(gid)

            pid_info[pid] = (cx, cy, side, factor, iouid)

            if gid_info.get(gid, None) is None:
                gid_info[gid] = dict(
                    pids=[],
                    factors=[],
                    x0=np.inf,
                    x1=-np.inf,
                    y0=np.inf,
                    y1=-np.inf
                )
            gid_info[gid]['pids'].append(pid)

            if factor not in gid_info[gid]['factors']:
                gid_info[gid]['factors'].append(factor)

            gid_info[gid]['x0'] = min(x0, gid_info[gid].get('x0'))
            gid_info[gid]['x1'] = max(x1, gid_info[gid].get('x1'))
            gid_info[gid]['y0'] = min(y0, gid_info[gid].get('y0'))
            gid_info[gid]['y1'] = max(y1, gid_info[gid].get('y1'))

    return fname_to_gid, gid_info, pid_info

# for debugging
def load_image_fnames(base_dir, dir_name, ext='bmp'):
    files = []
    # find those files with the specified extension
    dataset_dir = os.path.join(base_dir, dir_name)
    for f in os.listdir(dataset_dir):
        if f.endswith(ext):
            files.append(os.path.join(dataset_dir, f))
    return sorted(files)

# for debugging
def load_patches(dir_name, fnames, patch_size, n_channels, n_patches=None):
    if n_patches is None:
        patches_all = []
        n_patches = np.inf
    else:
        patches_all = np.zeros((n_patches, *patch_size, n_channels), dtype=np.float32)
    #print(self.PATCHES_PER_COL, self.PATCHES_PER_ROW)
    counter = 0
    done = False
    for f in tqdm(fnames, desc='Loading dataset %s' % dir_name):
        if done: break
        assert os.path.isfile(f), 'Not a file: %s' % f
        # todo: what if the maximum value is not 255?
        im = imread(f) / 255.
        patches_row = np.split(im, PATCHES_PER_ROW, axis=0)
        for row in patches_row:
            if done: break
            patches = np.split(row, PATCHES_PER_COL, axis=1)
            for patch in patches:
                if done: break
                patch_tensor = patch.reshape(*patch_size, n_channels)
                if isinstance(patches_all, list):
                    patches_all.append(patch_tensor)
                else:
                    patches_all[counter] = patch_tensor
                counter += 1
                if counter >= n_patches:
                    done = True
    if isinstance(patches_all, list):
        patches_all = np.asarray(patches_all)
    return patches_all

def generate_source_images(
        fname_to_gid, gid_info, pid_info,
        n_blocks=3, make_square=True, src_size=256,
        debug=False, output_dir=None, is_sem=True, do_src_shift=True,
        fixed_factors=None
):
    data_manager = None
    if not debug and output_dir is not None:
        data_manager = PatchDataManager(output_dir)


    src_counter = 0
    for fname, gIDs in tqdm(fname_to_gid.items(), desc='Generate source collection'):
        im = imread(fname, as_gray=True)
        im /= 255.
        # for SEM image, need to crop bottom rows to exclude information strings...
        if is_sem:
            im = im[:-110,:]
        im_h, im_w = im.shape

        if fixed_factors is not None:
            ms_im = get_multiscale(im, fixed_factors, debug=debug)
            ms_im_f = [get_interpolator(_im) for _im in ms_im]

        # plt.imshow(im, cmap='gray')
        # plt.show()

        for gid in gIDs:
            pIDs = gid_info[gid].get('pids')
            factors = gid_info[gid].get('factors')
            factors = sorted(factors)
            x0, y0 = gid_info[gid].get('x0'), gid_info[gid].get('y0')
            x1, y1 = gid_info[gid].get('x1'), gid_info[gid].get('y1')
            blk_width, blk_height = x1 - x0 + 1, y1 - y0 + 1
            if make_square:
                blk_side = max(blk_width, blk_height)
                blk_width = blk_side
                blk_height = blk_side

            # create multi-scaled images
            if fixed_factors is None:
                ms_im = get_multiscale(im, factors, debug=debug)
                ms_im_f = [get_interpolator(_im) for _im in ms_im]
                # ms_im_shape = [_im.shape for _im in ms_im]

            # determine source image size and upper-left corner position
            src_w = int(n_blocks * blk_width)
            src_h = int(n_blocks * blk_height)

            min_src_x0 = x0 - (n_blocks - 1) * blk_width
            min_src_y0 = y0 - (n_blocks - 1) * blk_height
            src_x0 = np.round(np.random.uniform(min_src_x0, x0))
            src_y0 = np.round(np.random.uniform(min_src_y0, y0))

            # shift source box to be within the image
            if do_src_shift:
                src_x0 = max(0, src_x0)
                src_y0 = max(0, src_y0)
                src_x1 = min(src_x0 + src_w, im_w-1)
                src_y1 = min(src_y0 + src_h, im_h-1)
                src_w = src_x1 - src_x0 + 1
                src_h = src_y1 - src_y0 + 1
            # print(im_h, im_w)
            # print(src_x0, src_y0, src_x1, src_y1, src_w, src_h)

            # crop multi-scaled source images
            ms_src = []
            ms_src_scale = []
            for idx, _factor in enumerate(factors):
                _x0 = src_x0 / _factor
                _y0 = src_y0 / _factor
                _w  = np.round(src_w / _factor)
                _h  = np.round(src_h / _factor)
                _step_x = _w / src_size
                _step_y = _h / src_size
                _src = ms_im_f[idx](
                    np.arange(_x0, _x0 + _w, _step_x),
                    np.arange(_y0, _y0 + _h, _step_y)
                )

                _src = _src[:src_size, :src_size]
                _src = (_src - _src.min()) / _src.ptp()
                #print(_src.shape)
                ms_src.append(_src)
                ms_src_scale.append((_step_x, _step_y))

            info_src_id = '{:d} {:d}'.format(src_counter, src_counter + len(ms_src))
            info = []
            for pid in pIDs:
                cx, cy, side, factor, iouid = pid_info[pid]

                # NOTE: in fact, all bboxes have same coordinates because the multi-scaled
                # source images are resized (or sampled) to have the same dimension.
                bboxes = []
                for _factor, (_scale_x, _scale_y) in zip(factors, ms_src_scale):
                    tar_x0 = ((cx - side/2) - src_x0) / _factor / _scale_x
                    tar_y0 = ((cy - side/2) - src_y0) / _factor / _scale_y
                    tar_w  = side / _factor / _scale_x
                    tar_h  = side / _factor / _scale_y
                    bboxes.append((tar_x0, tar_y0, tar_w, tar_h))
                    #print(pid, _factor, bboxes[-1])

                bbox = bboxes[-1]
                info.append('{:d} {:s} {:.3f} {:.3f} {:.3f} {:.3f}\n'.format(
                    pid, info_src_id, bbox[0], bbox[1], bbox[2], bbox[3]
                ))

                # visualization for debugging (src + bboxes)
                if debug:
                    fig, ax = plt.subplots(3, len(bboxes))
                    for _ax, _src, _bbox in zip(ax[0], ms_src, bboxes):
                        _ax.imshow(_src, cmap='gray')
                        _ax.add_patch(Rectangle(
                            (_bbox[0], _bbox[1]), _bbox[2], _bbox[3],
                            linewidth=1, edgecolor='r', facecolor='none'
                        ))

                    for _ax, _src, _bbox in zip(ax[1], ms_src, bboxes):
                        _x0 = int(np.round(_bbox[0]))
                        _y0 = int(np.round(_bbox[1]))
                        _w  = int(np.round(_bbox[2]))
                        _h  = int(np.round(_bbox[3]))
                        _x1 = _x0 + _w
                        _y1 = _y0 + _h
                        _src_h, _src_w = _src.shape

                        if 0 <= _x0 < _src_w and 0 <= _y0 < _src_h and \
                            0 <= _x1 < _src_w and 0 <= _y1 < _src_h:
                            _tar = _src[_y0:_y1, _x0:_x1]
                        else:
                            _tar = np.zeros((_h, _w), dtype=np.float32)
                            _xx0, _yy0 = max(_x0, 0), max(_y0, 0)
                            _xx1, _yy1 = min(_x1, _src_w), min(_y1, _src_h)
                            _ox = _xx0 - _x0
                            _oy = _yy0 - _y0
                            _ow = _xx1 - _xx0
                            _oh = _yy1 - _yy0
                            _tar[_oy:_oy + _oh, _ox:_ox + _ow] = _src[_yy0:_yy1, _xx0:_xx1]

                        _ax.imshow(_tar, cmap='gray')

                    for _ax in ax[2]:
                        _ax.imshow(np.squeeze(patches[pid]), cmap='gray')

                    plt.show()

            if data_manager is not None:
                data_manager.add_patches(ms_src)
                data_manager.add_info(info)
            src_counter += len(ms_src)

    if data_manager is not None:
        data_manager.dump()
    print('# source images: %s' % src_counter)

if __name__ == '__main__':
    np.random.seed(2019)

    #base_dir = '/home/sungsooha/Desktop/Data/ftfy/sem/train'
    base_dir = '/home/sungsooha/Desktop/Data/ftfy/austin'
    base_dir_file = os.path.join(base_dir, 'campus')
    data_dir = 'campus_patch'

    n_blocks = 3 # n x n blocks
    src_size = 256
    make_square = False
    is_sem = False
    do_src_shift = False
    fixed_factors = None

    is_v100_path = True

    debug = True
    output_dir = os.path.join(base_dir, data_dir, 'sources')

    if not is_sem:
        PATCHES_PER_COL = 6
        PATCHES_PER_ROW = 13
        fixed_factors = [1, 2, 4, 6, 8, 10]


    # ------------------------------------------------------------------------
    # Load info
    # ------------------------------------------------------------------------
    if is_sem:
        fname_to_gid, gid_info, pid_info = load_sem_info(base_dir, data_dir)
    else:
        fname_to_gid, gid_info, pid_info = load_info(base_dir, data_dir, base_dir_file=base_dir_file)

    if debug:
        dir_name = os.path.join(data_dir, 'patches') if is_sem else data_dir
        n_patches = len(list(pid_info.keys()))
        patch_fnames = load_image_fnames(base_dir, dir_name)
        patches = load_patches(data_dir, patch_fnames, (128, 128), 1, n_patches)

    print('Data dir : %s' % data_dir)
    print('# images : %s' % len(list(fname_to_gid.keys())))
    print('# groups : %s' % len(list(gid_info.keys())))
    print('# patches: %s' % len(list(pid_info.keys())))
    #print('patches  : {}'.format(patches.shape))



    # ------------------------------------------------------------------------
    # Generate source collection
    # ------------------------------------------------------------------------
    # generate_source_images(
    #     fname_to_gid=fname_to_gid,
    #     gid_info=gid_info,
    #     pid_info=pid_info,
    #     n_blocks=n_blocks,
    #     make_square=make_square,
    #     src_size=src_size,
    #     debug=debug,
    #     output_dir=output_dir,
    #     is_sem=is_sem,
    #     do_src_shift=do_src_shift,
    #     fixed_factors=fixed_factors
    # )

