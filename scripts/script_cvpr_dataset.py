"""
Generate training examples

Authors:
    Sungsoo Ha (sungsooha@bnl.gov)

Last Modified:
    March 12, 2019
"""
import os
import numpy as np

from tqdm import tqdm
from skimage.io import imread

# for triplet
from scripts.script_sem_generate_samples import load_info
from scripts.script_sem_generate_samples import generate_triplet_samples
from scripts.script_sem_generate_samples import generate_matched_pairs
from scripts.script_sem_generate_samples import generate_image_retrieval_samples

# for ftfy (source images)
from scripts.script_ftfy_src_collection import load_sem_info
from utils.sem_data import PatchDataManager, PatchExtractor
from utils.data import get_multiscale, get_interpolator

# modified only for sem images from script_ftfy_src_collection.py
def generate_source_images(
        output_dir,
        fname_to_gid, gid_info, pid_info,
        train_gids, test_gids, gid_offset,
        n_blocks=3, src_size=256, is_sem=True
):
    data_manager = PatchDataManager(output_dir, ['train_info.txt', 'test_info.txt'])

    src_counter = 0
    for fname, gIDs in tqdm(fname_to_gid.items(), desc='Generate source collection'):
        im = imread(fname, as_gray=True)
        #im /= 255.
        # for SEM image, need to crop bottom rows to exclude information strings...
        if is_sem: im = im[:-110,:]
        im_h, im_w = im.shape

        for gid in gIDs:
            pIDs = gid_info[gid].get('pids')
            factors = gid_info[gid].get('factors')
            factors = sorted(factors)
            x0, y0 = gid_info[gid].get('x0'), gid_info[gid].get('y0')
            x1, y1 = gid_info[gid].get('x1'), gid_info[gid].get('y1')
            blk_width, blk_height = x1 - x0 + 1, y1 - y0 + 1

            # determine source image size and upper-left corner position
            src_w = int(n_blocks * blk_width)
            src_h = int(n_blocks * blk_height)

            min_src_x0 = x0 - (n_blocks - 1) * blk_width
            min_src_y0 = y0 - (n_blocks - 1) * blk_height
            src_x0 = np.round(np.random.uniform(min_src_x0, x0))
            src_y0 = np.round(np.random.uniform(min_src_y0, y0))

            # shift source box to be within the image
            src_x0 = max(0, src_x0)
            src_y0 = max(0, src_y0)
            src_x1 = min(src_x0 + src_w, im_w-1)
            src_y1 = min(src_y0 + src_h, im_h-1)
            src_w = src_x1 - src_x0 + 1
            src_h = src_y1 - src_y0 + 1

            # multi-scaled source images
            ms_im_f = [
                get_interpolator(_im)
                for _im in get_multiscale(im, factors, debug=False)
            ]
            ms_src = []
            ms_src_scale = []
            for idx, _factor in enumerate(factors):
                _x0 = src_x0 / _factor
                _y0 = src_y0 / _factor
                _w  = np.round(src_w / _factor)
                _h  = np.round(src_h / _factor)
                _step_x = (_w - 1) / src_size
                _step_y = (_h - 1) / src_size
                _src = ms_im_f[idx](
                    np.arange(_x0, _x0 + _w, _step_x),
                    np.arange(_y0, _y0 + _h, _step_y)
                )
                _src = _src[:src_size, :src_size]
                _src = (_src - _src.min()) / _src.ptp()
                ms_src.append(_src)
                ms_src_scale.append((_factor * _step_x, _factor * _step_y))

            info_src = '{:d} {:d} {:.3f} {:.3f} {:.3f} {:.3f}'.format(
                src_counter, src_counter + len(ms_src), src_x0, src_y0, src_w, src_h)
            src_counter += len(ms_src)
            info = []
            for pid in pIDs:
                # With the same (cx, cy, side), it will have the same bbox coordinate regardless
                # of down factor or iou id of a patch. But, for simplicity, it repeated over
                # all patches.
                cx, cy, side, _, _ = pid_info[pid]

                # Note that with small down scale factor, the coordinates over the different
                # scales are almost identical. However, as it goes larger, the diffences get
                # larger. So, it is better to keep all coordinates.
                info_bboxes = ''
                for _scale_x, _scale_y in ms_src_scale:
                    tar_x0 = ((cx - side/2) - src_x0) / _scale_x
                    tar_y0 = ((cy - side/2) - src_y0) / _scale_y
                    tar_w  = side / _scale_x
                    tar_h  = side / _scale_y
                    tar_x1 = tar_x0 + tar_w
                    tar_y1 = tar_y0 + tar_h

                    # the coordinate can be out of the source image region.
                    # in this case, it will have either negative x0 or y0, or
                    # x1 or y1 that exceeds the width or the height of the source image.
                    # In this case, we want the network to predict the overlapped region.
                    tar_x0 = max(tar_x0, 0.)
                    tar_y0 = max(tar_y0, 0.)
                    tar_x1 = min(tar_x1, src_size-1)
                    tar_y1 = min(tar_y1, src_size-1)
                    tar_w  = tar_x1 - tar_x0
                    tar_h  = tar_y1 - tar_y0

                    bbox_str = ' {:.3f} {:.3f} {:.3f} {:.3f}'.format(
                        tar_x0, tar_y0, tar_w, tar_h
                    )
                    info_bboxes += bbox_str
                    #bboxes.append((tar_x0, tar_y0, tar_w, tar_h))

                info.append('{:d} {:s}{:s}\n'.format(pid, info_src, info_bboxes))

            data_manager.add_patches(ms_src)
            gid += gid_offset
            if gid in train_gids:
                data_manager.add_info(info, 0)
            elif gid in test_gids:
                data_manager.add_info(info, 1)
            else:
                raise ValueError("Possibly wrong `gid offset` value: %s" % gid_offset)

    data_manager.dump()
    print('[{:s}]: # source images: {:d}'.format(output_dir, src_counter))

def generate_for_ftfy(base_dir, train_ratio=.7):

    # ------------------------------------------------------------------------
    # fetch available data directories
    # ------------------------------------------------------------------------
    data_dirs = []
    for f in os.listdir(base_dir):
        if os.path.isdir(os.path.join(base_dir,f)):
            data_dirs.append(f)
    data_dirs = sorted(data_dirs)
    assert len(data_dirs) > 0, 'No available data directories.'

    # data split should be done with triplet data generation
    train_groups = np.load(os.path.join(base_dir, 'train_groups.npy'))
    test_groups = np.load(os.path.join(base_dir, 'test_groups.npy'))

    # ------------------------------------------------------------------------
    # Load information
    # ------------------------------------------------------------------------
    data_info = dict()
    group_offsets = dict()
    n_acc_groups = 0
    n_acc_images = 0
    n_acc_patches = 0
    for data_dir in data_dirs:
        data_info[data_dir], n_images, n_groups, n_patches = load_sem_info(base_dir, data_dir)
        group_offsets[data_dir] = n_acc_groups
        n_acc_groups += n_groups
        n_acc_images += n_images
        n_acc_patches += n_patches

    print("# SEM image sets    : %s" % len(data_dirs))
    print("# SEM images        : %s" % n_acc_images)
    print("# SEM patch groups  : %s" % n_acc_groups)
    print("# SEM patches       : %s" % n_acc_patches)
    print("# Train patch groups: %s" % len(train_groups))
    print("# Test patch groups : %s" % len(test_groups))

    # ------------------------------------------------------------------------
    # Process over each data_dir
    # ------------------------------------------------------------------------
    for data_dir in data_dirs:
        print("Process on: %s" % data_dir)
        fname_to_gid, gid_info, pid_info = data_info[data_dir]
        g_offset = group_offsets[data_dir]
        generate_source_images(
            output_dir=os.path.join(base_dir, data_dir, 'sources'),
            fname_to_gid=fname_to_gid,
            gid_info=gid_info,
            pid_info=pid_info,
            train_gids=train_groups, test_gids=test_groups, gid_offset=g_offset,
            n_blocks=3, src_size=256, is_sem=True
        )

def generate_for_triplet(base_dir, train_ratio=.7,
                         do_triplet=False, do_matched=False, do_retrieval=True):
    n_triplet_samples = 1000000
    n_matched_pairs   =   50000
    n_query_per_group = 1

    # ------------------------------------------------------------------------
    # fetch available data directories
    # ------------------------------------------------------------------------
    data_dirs = []
    for f in os.listdir(base_dir):
        if os.path.isdir(os.path.join(base_dir,f)):
            data_dirs.append(f)
    data_dirs = sorted(data_dirs)
    assert len(data_dirs) > 0, 'No available data directories.'

    # ------------------------------------------------------------------------
    # Load information
    # ------------------------------------------------------------------------
    data_info = dict()
    patch_offsets = dict()
    group_offsets = dict()
    n_acc_patches = 0
    n_acc_groups = 0
    n_acc_images = 0
    for data_dir in data_dirs:
        data_info[data_dir], n_patches, n_groups, n_images = load_info(base_dir, data_dir)

        patch_offsets[data_dir] = n_acc_patches
        group_offsets[data_dir] = n_groups

        n_acc_groups += n_groups
        n_acc_patches += n_patches
        n_acc_images += n_images

    # split train and test dataset using patch groups
    try:
        train_groups = np.load(os.path.join(base_dir, 'train_groups.npy'))
        test_groups = np.load(os.path.join(base_dir, 'test_groups.npy'))
    except FileNotFoundError:
        groups_ind = np.arange(n_acc_groups)
        np.random.shuffle(groups_ind)
        n_train = int(n_acc_groups * train_ratio)
        train_groups = groups_ind[:n_train]
        test_groups = groups_ind[n_train:]

        np.save(os.path.join(base_dir, 'train_groups.npy'), train_groups)
        np.save(os.path.join(base_dir, 'test_groups.npy'), test_groups)

    print("# SEM image sets    : %s" % len(data_dirs))
    print("# SEM images        : %s" % n_acc_images)
    print("# SEM patch groups  : %s" % n_acc_groups)
    print("# SEM patches       : %s" % n_acc_patches)
    print("# Train patch groups: %s" % len(train_groups))
    print("# Test patch groups : %s" % len(test_groups))

    if do_triplet:
        generate_triplet_samples(
            output_dir=base_dir,
            data_info=data_info,
            n_samples=n_triplet_samples,
            pid_offsets=patch_offsets,
            gid_offsets=group_offsets,
            gid_include=train_groups,
            fname_prefix='train_triplet'
        )

    if do_matched:
        for gids, fname in zip([train_groups, test_groups], ['train_matched', 'test_matched']):
            generate_matched_pairs(
                output_dir=base_dir,
                data_info=data_info,
                n_samples=n_matched_pairs,
                pid_offsets=patch_offsets,
                gid_offsets=group_offsets,
                gid_include=gids,
                fname_prefix=fname
            )

    if do_retrieval:
        for gids, fname in zip([train_groups, test_groups], ['train_retrieval', 'test_retrieval']):
            generate_image_retrieval_samples(
                output_dir=base_dir,
                data_info=data_info,
                pid_offsets=patch_offsets,
                gid_offsets=group_offsets,
                gid_include=gids,
                n_query_per_group=n_query_per_group,
                fname_prefix=fname
            )


if __name__ == '__main__':
    np.random.seed(2019)
    do_triplet = False
    do_matched = False
    do_retrieval = False

    base_dir = '/home/sungsooha/Desktop/Data/ftfy/sem/train'

    generate_for_triplet(base_dir, train_ratio=.7,
                         do_triplet=do_triplet,
                         do_matched=do_matched,
                         do_retrieval=do_retrieval)

    # generate_for_ftfy(base_dir, train_ratio=.7)
