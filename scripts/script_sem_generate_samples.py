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
    pid_to_fid = dict() # {pid} --> {fname}
    n_patches = 0
    uImages = []
    with open(info_fname, 'r') as file:
        for line in file:
            img_fname, pid, gid, iouid, cx, cy, side, down_factor = line.split()

            pID = int(pid)
            gID = int(gid)
            iouID = int(iouid)

            if img_fname not in uImages:
                uImages.append(img_fname)

            if gID not in gIDs:
                gIDs.append(gID)

            key = get_key(gID, iouID)
            if datamap.get(key, None) is None:
                datamap[key] = [pID]
            else:
                datamap[key].append(pID)

            pid_to_fid[pID] = img_fname
            n_patches += 1

    return (gIDs, datamap, pid_to_fid), n_patches, len(gIDs), len(uImages)

def generate_triplet_samples(
        output_dir,
        data_info:dict,
        n_samples,
        pid_offsets,
        gid_offsets,
        gid_include,
        fname_prefix='triplet'
):
    data_info_key = list(data_info.keys())

    fname = os.path.join(output_dir, '{:s}.txt'.format(fname_prefix))
    file = open(fname, 'w')

    def update_file(idx_a, idx_p, idx_n):
        file.writelines('{:d} {:d} {:d}\n'.format(idx_a, idx_p, idx_n))
        file.flush()

    m1, m2, m3 = 0, 0, 0

    for _ in tqdm(range(n_samples), desc='Generating triplet samples'):
        is_valid = False
        while not is_valid:
            key = np.random.choice(data_info_key)
            n_key = key
            gIDs, datamap, pid_to_fid = data_info[key]

            gid = np.random.choice(gIDs)
            n_gid = gid
            p_gid_offset = gid_offsets[key]
            n_gid_offset = p_gid_offset

            idx_a = np.random.choice(datamap[get_key(gid, 0)])
            offset_a = pid_offsets[key]
            offset_n = offset_a

            method_id = -1

            # method I: within a patch group
            # - anchor /in {A}
            # - positive /in {A, B, C} except the anchor
            # - negative /in {B, C, D} except the set where the positive is originated
            prob = np.random.random()
            if prob < 0.33:
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
                    idx_n = np.random.choice(datamap[get_key(n_gid, n_iou)])

                m1+=1
                method_id = 1

            # method II: over patch groups in a set but in a same file
            # - anchor /in {A}
            # - positive /in {A, B} except the anchor
            # - negative /in {A', B'} of other key points
            elif prob < 0.66:
                used_patches = [idx_a]
                idx_p = idx_a
                while idx_p in used_patches:
                    p_iou = np.random.choice([0, 1])
                    idx_p = np.random.choice(datamap[get_key(gid, p_iou)])

                used_patches.append(idx_p)
                # anchor and positive are from the same file as they are belonging to the same group
                fid = pid_to_fid[idx_a]
                idx_n = idx_p
                fid_n = fid + '_xxx' # to make it different
                while idx_n in used_patches or fid_n != fid:
                    n_iou = np.random.choice([0, 1])
                    n_gid = np.random.choice(gIDs)
                    idx_n = np.random.choice(datamap[get_key(n_gid, n_iou)])
                    fid_n = pid_to_fid[idx_n]

                m2+=1
                method_id = 2
                #print(fid, fid_n)

            # method III: over patch set
            # - anchor /in {A}
            # - positive /in {A, B} except the anchor
            # - negative /in {A', B'} of other key points
            # todo: it supposed to sample the negative example from other key points from different
            # SEM images, but it didn't. But, it seems to work well...
            else:
                idx_p = idx_a
                while idx_p == idx_a:
                    p_iou = np.random.choice([0, 1])
                    idx_p = np.random.choice(datamap[get_key(gid, p_iou)])

                while n_key == key:
                    n_key = np.random.choice(data_info_key)
                n_gIDs, n_datamap, _ = data_info[n_key]
                n_gid_offset = gid_offsets[n_key]
                n_iou = np.random.choice([0, 1])
                n_gid = np.random.choice(n_gIDs)
                idx_n = np.random.choice(n_datamap[get_key(n_gid, n_iou)])
                offset_n = pid_offsets[n_key]

                m3+=1
                method_id = 3

            gid += p_gid_offset
            n_gid += n_gid_offset

            if gid in gid_include and n_gid in gid_include:
                idx_a += offset_a
                idx_p += offset_a
                idx_n += offset_n

                update_file(idx_a, idx_p, idx_n)
                is_valid = True
            else:
                if method_id == 1: m1-=1
                if method_id == 2: m2-=1
                if method_id == 3: m3-=1

    file.close()

    print('method I: ', m1)
    print('method II: ', m2)
    print('method III: ', m3)

def generate_matched_pairs(
        output_dir,
        data_info:dict,
        n_samples,
        pid_offsets,
        gid_offsets,
        gid_include,
        fname_prefix='matched'
):
    data_info_key = list(data_info.keys())

    fname = os.path.join(output_dir, '{:s}.txt'.format(fname_prefix))
    file = open(fname, 'w')

    def update_file(idx_1, idx_2, is_match):
        file.writelines('{:d} {:d} {:d}\n'.format(idx_1, idx_2, is_match))

    for _ in tqdm(range(n_samples), desc='Generating matched pairs'):

        # matched if two patches from either A or B in the same group
        key = np.random.choice(data_info_key)
        gIDs, datamap, _ = data_info[key]
        gid = np.random.choice(gIDs)
        while gid+gid_offsets[key] not in gid_include:
            key = np.random.choice(data_info_key)
            gIDs, datamap, _ = data_info[key]
            gid = np.random.choice(gIDs)

        idx_1 = np.random.choice(datamap[get_key(gid, 0)]) + pid_offsets[key]
        idx_2 = idx_1
        while idx_1 == idx_2:
            iou_id = np.random.choice([0, 1])
            idx_2 = np.random.choice(datamap[get_key(gid, iou_id)]) + pid_offsets[key]

        update_file(idx_1, idx_2, 1)

        if np.random.random() < 0.5:
            # unmatched if two patches from different groups (set)
            key_1, key_2 = np.random.choice(data_info_key, 2, replace=False)
            gIDs_1, datamap_1, _ = data_info[key_1]
            gIDs_2, datamap_2, _ = data_info[key_2]
            gid_1 = np.random.choice(gIDs_1)
            gid_2 = np.random.choice(gIDs_2)
            while gid_1+gid_offsets[key_1] not in gid_include or \
                    gid_2+gid_offsets[key_2] not in gid_include:
                key_1, key_2 = np.random.choice(data_info_key, 2, replace=False)
                gIDs_1, datamap_1, _ = data_info[key_1]
                gIDs_2, datamap_2, _ = data_info[key_2]
                gid_1 = np.random.choice(gIDs_1)
                gid_2 = np.random.choice(gIDs_2)

            iou_1 = np.random.choice([0, 1])
            iou_2 = np.random.choice([0, 1])
            idx_1 = np.random.choice(datamap_1[get_key(gid_1, iou_1)]) + pid_offsets[key_1]
            idx_2 = np.random.choice(datamap_2[get_key(gid_2, iou_2)]) + pid_offsets[key_2]
            update_file(idx_1, idx_2, 0)

        else:
            # unmatched if one patch from {0, 1} and the other from {2, 3}
            key = np.random.choice(data_info_key)
            gIDs, datamap, _ = data_info[key]
            gid = np.random.choice(gIDs)
            while gid + gid_offsets[key] not in gid_include:
                key = np.random.choice(data_info_key)
                gIDs, datamap, _ = data_info[key]
                gid = np.random.choice(gIDs)

            idx_1 = np.random.choice(datamap[get_key(gid, 0)]) + pid_offsets[key]
            idx_2 = idx_1
            while idx_1 == idx_2:
                iou_id = np.random.choice([2, 3])
                idx_2 = np.random.choice(datamap[get_key(gid, iou_id)]) + pid_offsets[key]

            update_file(idx_1, idx_2, 0)
    file.close()

def generate_image_retrieval_samples(
        output_dir,
        data_info:dict,
        pid_offsets,
        gid_offsets,
        gid_include,
        n_query_per_group=1,
        fname_prefix='retrieval'
):
    #data_info_key = list(data_info.keys())

    fname = os.path.join(output_dir, '{:s}.txt'.format(fname_prefix))
    file = open(fname, 'w')

    def update_file(idx, label, is_query):
        file.writelines('{:d} {:s} {:d}\n'.format(idx, label, is_query))

    for key, (gIDs, datamap, _) in data_info.items():
        for gid in tqdm(gIDs, desc='Generating retrieval test set: %s' % key):
            if gid + gid_offsets[key] not in gid_include:
                continue

            used_pid = []
            label_q = '{:s}_{:d}'.format(key, gid)
            for _ in range(n_query_per_group):
                # randomly choose a patch for query
                idx_q = np.random.choice(datamap[get_key(gid, 0)]) + pid_offsets[key]
                while idx_q in used_pid:
                    idx_q = np.random.choice(datamap[get_key(gid, 0)]) + pid_offsets[key]
                used_pid.append(idx_q)
                update_file(idx_q, label_q, 1)

                # randomly choose a patch expected to be retrieved
                idx_r = idx_q
                while idx_r in used_pid:
                    iou = np.random.choice([0, 1])
                    idx_r = np.random.choice(datamap[get_key(gid, iou)]) + pid_offsets[key]
                used_pid.append(idx_r)
                update_file(idx_r, label_q, 0)

                # randomly choose a patch for dummy
                idx_d = idx_r
                iou = 2
                while idx_d in used_pid:
                    iou = np.random.choice([2, 3])
                    idx_d = np.random.choice(datamap[get_key(gid, iou)]) + pid_offsets[key]
                used_pid.append(idx_d)
                label_d = '{:s}_{:d}_{:d}_dummy'.format(key, gid, iou)
                update_file(idx_d, label_d, 0)

        file.close()

if __name__ == '__main__':
    base_dir = '/home/sungsooha/Desktop/Data/ftfy/sem/train'
    n_triplet_samples = 1000000
    n_matched_pairs   =   50000
    n_query_per_group = 1 # must 1

    do_triplet = False
    do_matched = False
    do_retrieval = False

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
    offsets = dict()
    n_acc_patches = 0
    for data_dir in data_dirs:
        data_info[data_dir], n_patches = load_info(base_dir, data_dir)
        offsets[data_dir] = n_acc_patches
        n_acc_patches += n_patches

    n_image_set = len(list(data_info.keys()))
    n_images = 0
    n_patch_groups = 0
    for k, (gIDs, datamap, pid_to_fid) in data_info.items():
        n_images += len(np.unique(list(pid_to_fid.values())))
        n_patch_groups += len(gIDs)
    print("# SEM images: %s" % n_images)
    print("# SEM image sets: %s" % n_image_set)
    print("# SEM patch groups: %s" % n_patch_groups)
    print("# SEM patches: %s" % n_acc_patches)

    # ------------------------------------------------------------------------
    # Generate triplet samples
    # ------------------------------------------------------------------------
    if do_triplet:
        generate_triplet_samples(
            output_dir=base_dir,
            data_info=data_info,
            n_samples=n_triplet_samples,
            pid_offsets=offsets
        )

    # ------------------------------------------------------------------------
    # Generate matched samples
    # ------------------------------------------------------------------------
    if do_matched:
        generate_matched_pairs(
            output_dir=base_dir,
            data_info=data_info,
            n_samples=n_matched_pairs,
            pid_offsets=offsets
        )

    # ------------------------------------------------------------------------
    # Generate retrieval test set
    # ------------------------------------------------------------------------
    if do_retrieval:
        generate_image_retrieval_samples(
            output_dir=base_dir,
            data_info=data_info,
            n_query_per_group=n_query_per_group,
            pid_offsets=offsets
        )