import os
import numpy as np
from scripts.script_sem_generate_samples import load_info
from scripts.script_sem_generate_samples import generate_triplet_samples
from scripts.script_sem_generate_samples import generate_matched_pairs
from scripts.script_sem_generate_samples import generate_image_retrieval_samples

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

    base_dir = '/home/sungsooha/Desktop/Data/ftfy/sem/train'
    do_triplet = False
    do_matched = False
    do_retrieval = True

    generate_for_triplet(base_dir, train_ratio=.7,
                         do_triplet=do_triplet,
                         do_matched=do_matched,
                         do_retrieval=do_retrieval)


