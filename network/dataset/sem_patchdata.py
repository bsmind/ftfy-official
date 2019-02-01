import os
import numpy as np
import tensorflow as tf

from network.dataset.patchdata import PatchDataSampler

from skimage.io import imread
from tqdm import tqdm

class IoUPatchDataSampler(PatchDataSampler):
    def __init__(
            self,
            base_dir,
            base_patch_size=(64,64),
            patches_per_col=8,
            patches_per_row=13,
            mode = True
    ):
        super().__init__(base_dir, base_patch_size, patches_per_row, patches_per_col, mode)

    def _load_info(self, dir_name, fname='info.txt'):
        info_name = os.path.join(self.base_dir, dir_name, fname)
        assert os.path.isfile(info_name), 'Not a file: %s' % info_name

        fid_counter = 0
        fname_to_fid = dict()
        fid_to_gid = dict()
        datamap = dict()
        labels = []
        with open(info_name, 'r') as f:
            for line in f:
                tokens = line.split()
                fname, pId, gId, iouId, cx, cy, factor = tokens
                pId = int(pId)
                gId = int(gId)
                iouId = int(iouId)

                if fname_to_fid.get(fname, None) is None:
                    fname_to_fid[fname] = fid_counter
                    fid = fid_counter
                    fid_counter += 1
                else:
                    fid = fname_to_fid[fname]

                if fid_to_gid.get(fid, None) is None:
                    fid_to_gid[fid] = [gId]
                else:
                    fid_to_gid[fid].append(gId)

                key = '{:d}_{:d}_{:d}'.format(fid, gId, iouId)
                if datamap.get(key, None) is None:
                    datamap[key] = [pId]
                else:
                    datamap[key].append(pId)

                labels.append(gId)

        return fid_to_gid, datamap, np.asarray(labels)

    def load_dataset(self, dir_name, ext='bmp', patch_size=(64, 64), n_channels=1, debug=True):
        assert os.path.exists(os.path.join(self.base_dir, dir_name)) == True, \
            'The dataset directory does not exist: %s' % dir_name

        fnames = self._load_image_fnames(dir_name, ext)
        patches = self._load_patches(dir_name, fnames, patch_size, n_channels)
        fid_to_gid, datamap, labels = self._load_info(dir_name)

        # initialize patch dataset
        self.data['patches'] = patches
        self.data['datamap'] = datamap
        self.data['fid_to_gid'] = fid_to_gid
        self.data['labels'] = labels

        if debug:
            groups = []
            for v in fid_to_gid.values():
                groups += v
            groups = np.array(groups)
            groups = np.unique(groups)

            #output = print
            output = tf.logging.info
            output('-- Dataset loaded : %s' % dir_name)
            output('-- # images       : %s' % len(fid_to_gid))
            output('-- # patches      : %s' % len(patches))
            output('-- # groups       : %s' % len(groups))
            output('-- # unique labels: %s' % len(np.unique(labels)))

    def generate_triplet(self, n_samples):
        self.index_a = []
        self.index_p = []
        self.index_n = []

        fid_to_groups = self.data['fid_to_gid']
        datamap = self.data['datamap']
        u_fid = list(fid_to_groups.keys())
        p_fid = np.ones(len(u_fid), dtype=np.float32)
        def get_key(fid, gid, iouid):
            return '{:d}_{:d}_{:d}'.format(fid, gid, iouid)

        for _ in tqdm(range(n_samples), desc='Generating triplets'):
        #for i_sample in range(n_samples):
            if p_fid.sum() < 2: p_fid[:] = 1
            # 1. randomly pick a fid
            fid_idx = np.random.choice(len(u_fid), replace=False, p=p_fid/p_fid.sum())
            fid_p = u_fid[fid_idx]
            p_fid[fid_idx] = 0
            # 2. randomly pick a gid within the fid
            gid = np.random.choice(fid_to_groups[fid_p])
            # 3. then, randomly pick triplet sample such that
            # - anchor: from iouID = 0
            idx_a = np.random.choice(datamap[get_key(fid_p, gid, 0)])
            fid_n = fid_p
            if np.random.random() < 0.5:
                # within a group
                # - positive and negative: from iouID = {1, 2, 3  (sorted selection}
                p_iou, n_iou = np.random.choice([1, 2, 3], 2, replace=False)
                if n_iou < p_iou: p_iou, n_iou = n_iou, p_iou
                idx_p = np.random.choice(datamap[get_key(fid_p, gid, p_iou)])
                idx_n = np.random.choice(datamap[get_key(fid_p, gid, n_iou)])
            else:
                # over groups
                # - positive is randomly selelcted from the highest iou group
                # - negative is randomly selected from different group
                idx_p = np.random.choice(datamap[get_key(fid_p, gid, 1)])

                fid_idx = np.random.choice(len(u_fid), replace=False, p=p_fid / p_fid.sum())
                fid_n = u_fid[fid_idx]
                p_fid[fid_idx] = 0
                gid = np.random.choice(fid_to_groups[fid_n])
                idx_n = np.random.choice(datamap[get_key(fid_n, gid, 0)])

            self.index_a.append(idx_a)
            self.index_p.append(idx_p)
            self.index_n.append(idx_n)

            # print('sample id: {:d}'.format(i_sample+1))
            # print('file id  : {:d}, {:d}'.format(fid_p, fid_n))
            # print('group id : {:d}'.format(gid))
            # print('triplet  : {:d}, {:d}, {:d}'.format(idx_a, idx_p, idx_n))

        # convert to numpy array
        self.index_a = np.array(self.index_a)
        self.index_p = np.array(self.index_p)
        self.index_n = np.array(self.index_n)

        self.n_triplet_samples = len(self.index_a)
        self.sample_idx = 0

    def generate_match_pairs(self, n_samples):

        matches = []

        fid_to_groups = self.data['fid_to_gid']
        datamap = self.data['datamap']
        u_fid = list(fid_to_groups.keys())
        p_fid = np.ones(len(u_fid), dtype=np.float32)
        def get_key(fid, gid, iouid):
            return '{:d}_{:d}_{:d}'.format(fid, gid, iouid)

        #for i_sample in range(n_samples):
        for _ in tqdm(range(n_samples), desc='Generating match pairs'):
            if p_fid.sum() < 1: p_fid[:] = 1
            # 1. randomly pick a fid
            fid_idx = np.random.choice(len(u_fid), replace=False, p=p_fid/p_fid.sum())
            fid = u_fid[fid_idx]
            p_fid[fid_idx] = 0
            # 2. randomly pick a gid within the fid
            gid_1, gid_2 = np.random.choice(fid_to_groups[fid], 2, replace=False)

            idx_1, idx_2, idx_3 = np.random.choice(
                datamap[get_key(fid, gid_1, 0)], 3, replace=False)
            idx_4 = np.random.choice(datamap[get_key(fid, gid_2, 0)])
            matches.append([idx_1, idx_2, 1])
            matches.append([idx_3, idx_4, 0])

        matches = np.asarray(matches)
        self.data['matches'] = matches
        self.n_matches = len(matches)

def input_fn(
        data_dir,
        base_patch_size, patches_per_row, patches_per_col,
        batch_size, patch_size, n_channels):

    data_sampler = IoUPatchDataSampler(
        base_dir=data_dir,
        base_patch_size=base_patch_size,
        patches_per_row=patches_per_row,
        patches_per_col=patches_per_col,
    )
    output_shape = [*patch_size, n_channels]
    dataset = (
        tf.data.Dataset()
            .from_generator(generator=lambda: data_sampler,
                            output_types=(tf.float32, tf.float32, tf.float32),
                            output_shapes=(output_shape, output_shape, output_shape))
            .shuffle(buffer_size=2*batch_size)
            .batch(batch_size=batch_size)
            .prefetch(1)
    )
    return dataset, data_sampler


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    base_dir = '/home/sungsooha/Desktop/Data/ftfy/austin'
    patch_dir = 'scene_patch' #'campus_patch'

    mode = False

    sampler = IoUPatchDataSampler(base_dir)
    sampler.load_dataset(patch_dir)

    sampler.set_mode(mode)
    if mode:
        sampler.generate_triplet(50)
    else:
        sampler.generate_match_pairs(10)

    for a, p, n in sampler:
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(np.squeeze(a))
        ax[1].imshow(np.squeeze(p))
        if mode:
            ax[2].imshow(np.squeeze(n))
        else:
            idx_1 = int(n[0,0,0])
            idx_2 = int(n[1,0,0])
            is_match = int(n[2,0,0])
            print('{:d}, {:d}, {:s}'.format(
                idx_1, idx_2, 'Y' if is_match == 1 else 'N'
            ))
        plt.show()






















