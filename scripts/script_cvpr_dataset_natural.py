"""
Generate training examples for natural image dataset
(Only slightly different from script_cvpr_dataset.py)

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
from utils.data import generate_triplet_samples
from utils.data import generate_matched_pairs
from utils.data import generate_image_retrieval_samples

# for ftfy
from scripts.script_ftfy_src_collection import load_info as ftfy_load_info
from scripts.script_cvpr_dataset import generate_source_images

def get_key(gid, iouid):
    return '{:d}_{:d}'.format(gid, iouid)

def load_info(base_dir, data_dir, fname='info.txt'):
    info_fname = os.path.join(base_dir, data_dir, fname)
    assert os.path.isfile(info_fname), 'Cannot find `info.txt`.'

    gIDs = list()    # list of group IDs
    datamap = dict() # {gid}_{iou_id} --> [pid]
    pid_to_fid = dict()
    n_patches = 0
    uImages = []

    with open(info_fname, 'r') as f:
        for line in f:
            fname, pid, gid, iouid, cx, cy, factor = line.split()

            pID = int(pid)
            gID = int(gid)
            iouID = int(iouid)

            if fname not in uImages:
                uImages.append(fname)

            if gID not in gIDs:
                gIDs.append(gID)

            key = get_key(gID, iouID)
            if datamap.get(key, None) is None:
                datamap[key] = [pID]
            else:
                datamap[key].append(pID)

            pid_to_fid[pID] = fname
            n_patches += 1

    return (gIDs, datamap, pid_to_fid), n_patches, len(gIDs), len(uImages)


def generate_for_ftfy(base_dir, data_dir):
    # data split should be done with triplet data generation
    train_groups = np.load(os.path.join(base_dir, 'train_groups_' + data_dir + '.npy'))
    test_groups = np.load(os.path.join(base_dir, 'test_groups_' + data_dir + '.npy'))

    # ------------------------------------------------------------------------
    # Load information
    # ------------------------------------------------------------------------
    data_info, n_images, n_groups, n_patches = ftfy_load_info(
        base_dir, data_dir, base_dir_file=os.path.join(base_dir, data_dir.split('_')[0]))
    print("image set            : %s" % data_dir)
    print("# images             : %s" % n_images)
    print("# patch groups       : %s" % n_groups)
    print("# patches            : %s" % n_patches)
    print("# train patch groups : %s" % len(train_groups))
    print("# test patch groups  : %s" % len(test_groups))

    # ------------------------------------------------------------------------
    # Process over each data_dir
    # ------------------------------------------------------------------------
    fname_to_gid, gid_info, pid_info = data_info
    generate_source_images(
        output_dir=os.path.join(base_dir, data_dir.split('_')[0] + '_sources'),
        fname_to_gid=fname_to_gid,
        gid_info=gid_info,
        pid_info=pid_info,
        train_gids=train_groups, test_gids=test_groups, gid_offset=0,
        n_blocks=3, src_size=256, is_sem=False
    )

def generate_for_triplet(base_dir, data_dir, train_ratio=.7,
                         do_triplet=True, do_matched=True, do_retrieval=True):
    n_triplet_samples = 1000000
    n_matched_pairs   =   50000
    n_query_per_group = 2

    # ------------------------------------------------------------------------
    # Load information
    # ------------------------------------------------------------------------
    data_info, n_patches, n_groups, n_images = load_info(base_dir, data_dir)

    # split train and test dataset using patch groups
    train_fn = 'train_groups_' + data_dir + '.npy'
    test_fn = 'test_groups_' + data_dir + '.npy'
    try:
        train_groups = np.load(os.path.join(base_dir, train_fn))
        test_groups = np.load(os.path.join(base_dir, test_fn))
    except FileNotFoundError:
        groups_ind = np.arange(n_groups)
        np.random.shuffle(groups_ind)
        n_train = int(n_groups * train_ratio)
        train_groups = groups_ind[:n_train]
        test_groups = groups_ind[n_train:]

        np.save(os.path.join(base_dir, train_fn), train_groups)
        np.save(os.path.join(base_dir, test_fn), test_groups)

    print("image set            : %s" % data_dir)
    print("# images             : %s" % n_images)
    print("# patch groups       : %s" % n_groups)
    print("# patches            : %s" % n_patches)
    print("# train patch groups : %s" % len(train_groups))
    print("# test patch groups  : %s" % len(test_groups))

    if do_triplet:
        generate_triplet_samples(
            output_dir=base_dir,
            data_info=data_info,
            n_samples=n_triplet_samples,
            gid_include=train_groups,
            fname_prefix='triplet_' + data_dir
        )

    if do_matched:
        fnames = ['train_matched_' + data_dir, 'test_matched_' + data_dir]
        for gids, fname in zip([train_groups, test_groups], fnames):
            generate_matched_pairs(
                output_dir=base_dir,
                data_info=data_info,
                n_samples=n_matched_pairs,
                gid_include=gids,
                fname_prefix=fname
            )

    if do_retrieval:
        fnames = ['train_retrieval_' + data_dir, 'test_retrieval_' + data_dir]
        for gids, fname in zip([train_groups, test_groups], fnames):
            generate_image_retrieval_samples(
                output_dir=base_dir,
                data_info=data_info,
                gid_include=gids,
                n_query_per_group=n_query_per_group,
                fname_prefix=fname
            )

if __name__ == '__main__':
    np.random.seed(2019)
    do_triplet = True
    do_matched = True
    do_retrieval = True

    base_dir = '/home/sungsooha/Desktop/Data/ftfy/austin'
    data_dir = 'human_patch'

    generate_for_triplet(
        base_dir=base_dir,
        data_dir=data_dir,
        train_ratio=0.7,
        do_triplet=do_triplet,
        do_matched=do_matched
    )

    generate_for_ftfy(base_dir, data_dir)