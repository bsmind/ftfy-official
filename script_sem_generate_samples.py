import os
import numpy as np
from tqdm import tqdm

def get_key(gid, iouid):
    return '{:d}_{:d}'.format(gid, iouid)

def load_info(base_dir, data_dir, patch_dir='patches', fname='info.txt'):
    info_fname = os.path.join(base_dir, data_dir, patch_dir, fname)
    assert os.path.exists(info_fname), 'There is no file: %s' % info_fname

    gIDs = list()    # list of group IDs
    datamap = dict() # {gid}_{iou_id} --> [pid]

    with open(info_fname, 'r') as file:
        for line in file:
            img_fname, pid, gid, iouid, cx, cy, side, down_factor = line.split()

            pID = int(pid)
            gID = int(gid)
            iouID = int(iouid)

            if gID not in gIDs:
                gIDs.append(gID)

            key = get_key(gID, iouID)
            if datamap.get(key, None) is None:
                datamap[key] = [pID]
            else:
                datamap[key].append(pID)

    return gIDs, datamap

def generate_triplet_samples(output_dir, data_info:dict, n_samples):
    data_info_key = list(data_info.keys())

    fname = os.path.join(output_dir, 'triplet_{:d}.txt'.format(n_samples))
    file = open(fname, 'w')

    def update_file(key, idx_a, idx_p, key_n, idx_n):
        file.writelines('{:s} {:d} {:d} {:s} {:d}\n'.format(
            key, idx_a, idx_p, key_n, idx_n
        ))
        file.flush()

    for _ in tqdm(range(n_samples), desc='Generating triplet samples'):
        key = np.random.choice(data_info_key)
        n_key = key
        gIDs, datamap = data_info[key]

        gid = np.random.choice(gIDs)
        idx_a = np.random.choice(datamap[get_key(gid, 0)])

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
                idx_n = np.random.choice(datamap[get_key(gid, n_iou)])

        # method II: over patch set
        # - anchor /in {A}
        # - positive /in {A, B} except the anchor
        # - negative /in {A', B'} of other key points
        else:
            idx_p = idx_a
            while idx_p == idx_a:
                p_iou = np.random.choice([0, 1])
                idx_p = np.random.choice(datamap[get_key(gid, p_iou)])

            while n_key == key:
                n_key = np.random.choice(data_info_key)
            n_gIDs, n_datamap = data_info[n_key]
            n_iou = np.random.choice([0, 1])
            n_gid = np.random.choice(n_gIDs)
            idx_n = np.random.choice(n_datamap[get_key(n_gid, n_iou)])

        update_file(key, idx_a, idx_p, n_key, idx_n)

    file.close()

def generate_matched_pairs(output_dir, data_info:dict, n_samples):
    data_info_key = list(data_info.keys())

    fname = os.path.join(output_dir, 'matched_{:d}.txt'.format(n_samples))
    file = open(fname, 'w')

    def update_file(key_1, idx_1, key_2, idx_2, is_match):
        file.writelines('{:s} {:d} {:s} {:d} {:d}\n'.format(
            key_1, idx_1, key_2, idx_2, is_match
        ))

    for _ in tqdm(range(n_samples), desc='Generating matched pairs'):
        key = np.random.choice(data_info_key)
        gIDs, datamap = data_info[key]

        # matched if two patches from either A or B in the same group
        gid = np.random.choice(gIDs)
        idx_1 = np.random.choice(datamap[get_key(gid, 0)])
        idx_2 = idx_1
        while idx_1 == idx_2:
            iou_id = np.random.choice([0, 1])
            idx_2 = np.random.choice(datamap[get_key(gid, iou_id)])
        update_file(key, idx_1, key, idx_2, 1)

        # unmatched if two patches from different groups (set)
        key_1, key_2 = np.random.choice(data_info_key, 2, replace=False)
        gIDs_1, datamap_1 = data_info[key_1]
        gIDs_2, datamap_2 = data_info[key_2]
        gid_1 = np.random.choice(gIDs_1)
        gid_2 = np.random.choice(gIDs_2)
        iou_1 = np.random.choice([0, 1])
        iou_2 = np.random.choice([0, 1])
        idx_1 = np.random.choice(datamap_1[get_key(gid_1, iou_1)])
        idx_2 = np.random.choice(datamap_2[get_key(gid_2, iou_2)])
        update_file(key_1, idx_1, key_2, idx_2, 0)
    file.close()

def generate_image_retrieval_samples(output_dir, data_info:dict, n_query_per_group=1):
    #data_info_key = list(data_info.keys())

    fname = os.path.join(output_dir, 'retrieval.txt')
    file = open(fname, 'w')

    def update_file(key, idx, label, is_query):
        file.writelines('{:s} {:d} {:s} {:d}\n'.format(key, idx, label, is_query))

    for key, (gIDs, datamap) in data_info.items():
        for gid in tqdm(gIDs, desc='Generating retrieval test set: %s' % key):
            used_pid = []
            label_q = '{:s}_{:d}'.format(key, gid)
            for _ in range(n_query_per_group):
                # randomly choose a patch for query
                idx_q = np.random.choice(datamap[get_key(gid, 0)])
                while idx_q in used_pid:
                    idx_q = np.random.choice(datamap[get_key(gid, 0)])
                used_pid.append(idx_q)
                update_file(key, idx_q, label_q, 1)

                # randomly choose a patch expected to be retrieved
                idx_r = idx_q
                while idx_r in used_pid:
                    iou = np.random.choice([0, 1])
                    idx_r = np.random.choice(datamap[get_key(gid, iou)])
                used_pid.append(idx_r)
                update_file(key, idx_r, label_q, 0)

                # randomly choose a patch for dummy
                idx_d = idx_r
                iou = 2
                while idx_d in used_pid:
                    iou = np.random.choice([2, 3])
                    idx_d = np.random.choice(datamap[get_key(gid, iou)])
                used_pid.append(idx_d)
                label_d = '{:s}_{:d}_{:d}_dummy'.format(key, gid, iou)
                update_file(key, idx_d, label_d, 0)

if __name__ == '__main__':
    base_dir = '/home/sungsooha/Desktop/Data/ftfy/sem/train'
    n_triplet_samples = 5000000
    n_matched_pairs   =   50000
    n_query_per_group = 1 # must 1

    do_triplet = True
    do_matched = True
    do_retrieval = True

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
    for data_dir in data_dirs:
        data_info[data_dir] = load_info(base_dir, data_dir)

    # ------------------------------------------------------------------------
    # Generate triplet samples
    # ------------------------------------------------------------------------
    if do_triplet:
        generate_triplet_samples(
            output_dir=base_dir,
            data_info=data_info,
            n_samples=n_triplet_samples
        )

    # ------------------------------------------------------------------------
    # Generate matched samples
    # ------------------------------------------------------------------------
    if do_matched:
        generate_matched_pairs(
            output_dir=base_dir,
            data_info=data_info,
            n_samples=n_matched_pairs
        )

    # ------------------------------------------------------------------------
    # Generate retrieval test set
    # ------------------------------------------------------------------------
    if do_retrieval:
        generate_image_retrieval_samples(
            output_dir=base_dir,
            data_info=data_info,
            n_query_per_group=n_query_per_group
        )