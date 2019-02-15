import os
import numpy as np
import tensorflow as tf

from network.dataset.sem_patchdata import IoUPatchDataSampler

from skimage.io import imread
from tqdm import tqdm

class IoUPatchDataSamplerExt(IoUPatchDataSampler):
    def __init__(
            self,
            base_dir,
            base_patch_size=(128,128),
            patches_per_col=10,
            patches_per_row=10,
            mode=0
    ):
        super().__init__(base_dir, base_patch_size, patches_per_col, patches_per_row, mode)

        self.key_ap = []
        self.key_n = []

    def _load_triplet_samples(self, dir_name, fname='triplet_5000000.txt'):
        triplet_fname = os.path.join(self.base_dir, dir_name, fname)
        assert os.path.isfile(triplet_fname), 'Cannot find %s!' % fname

        self.index_a = []
        self.index_p = []
        self.index_n = []
        self.key_ap = []
        self.key_n = []

        with open(triplet_fname, 'r') as f:
            for line in tqdm(f, 'Load triplet samples'):
                key, idx_a, idx_p, key_n, idx_n = line.split()
                self.index_a.append(int(idx_a))
                self.index_p.append(int(idx_p))
                self.index_n.append(int(idx_n))
                self.key_ap.append(key)
                self.key_n.append(key_n)

        # convert to numpy array
        self.index_a = np.array(self.index_a)
        self.index_p = np.array(self.index_p)
        self.index_n = np.array(self.index_n)
        self.key_ap = np.array(self.key_ap)
        self.key_n = np.array(self.key_n)

        self.n_triplet_samples = len(self.index_a)
        self.max_n_triplet_samples = self.n_triplet_samples
        self.sample_idx = 0

    def _load_matched_pairs(self, dir_name, fname='matched_50000.txt'):
        matched_fname = os.path.join(self.base_dir, dir_name, fname)
        assert os.path.isfile(matched_fname), 'Cannot find %s!' % fname

        matches = []
        matches_key = []
        with open(matched_fname, 'r') as f:
            for line in tqdm(f, 'Load matched pairs'):
                key_1, idx_1, key_2, idx_2, is_match = line.split()
                matches.append([int(idx_1), int(idx_2), int(is_match)])
                matches_key.append([key_1, key_2])

        matches = np.asarray(matches)
        matches_key = np.asarray(matches_key)
        self.data['matches'] = matches
        self.data['matches_key'] = matches_key
        self.n_matches = len(matches)
        self.max_n_matched_pairs = self.n_matches

    def _load_retrieval_set(self, dir_name, fname='retrieval.txt'):
        retrieval_fname = os.path.join(self.base_dir, dir_name, fname)
        assert os.path.isfile(retrieval_fname), 'Cannot find %s!' % fname

        retrievals = []
        retrievals_key = []
        labels = []
        counter = 0
        with open(retrieval_fname, 'r') as f:
            for line in tqdm(f, 'Load retrieval set'):
                key, idx, label, is_query = line.split()
                retrievals.append([int(idx), counter, int(is_query)])
                retrievals_key.append(key)
                labels.append(label)
                counter += 1

        self.data['retrieval'] = np.asarray(retrievals)
        self.data['retrieval_key'] = np.asarray(retrievals_key)
        self.data['labels'] = np.asarray(labels)
        self.n_retrievals = len(labels)

    def load_dataset(self, dir_name, ext='bmp', patch_size=(128, 128), n_channels=1, debug=True):
        assert isinstance(dir_name, list), 'It expects to get list of directory names'

        def get_n_patches(d):
            file = open(os.path.join(self.base_dir, d, 'patches', 'info.txt'), 'r')
            count = 0
            for _ in file: count += 1
            return count

        patches = dict()
        total_n_patches = 0
        for d in dir_name:
            n_patches = get_n_patches(d)
            fnames = self._load_image_fnames(os.path.join(d, 'patches'), ext)
            patches[d] = self._load_patches(
                os.path.join(d, 'patches'),
                fnames, patch_size, n_channels, n_patches
            )
            total_n_patches += n_patches

        self._load_triplet_samples('.')
        self._load_matched_pairs('.')
        self._load_retrieval_set('.')

        self.data['patches'] = patches

        if debug:
            output = print
            output('-- Dataset loaded : {}'.format(dir_name))
            output('-- # patches      : %s' % total_n_patches)
            output('-- # triplet samples: %s' % self.n_triplet_samples)
            output('-- # matched pairs  : %d' % (self.n_matches//2))
            output('-- # retrieval set  : %d' % self.n_retrievals)

    def train_next(self):
        if self.n_triplet_samples == 0 or self.sample_idx >= self.n_triplet_samples:
            self.sample_idx = 0

            # shuffle
            N = len(self.index_a)
            ind = np.arange(N)
            np.random.shuffle(ind)
            self.index_a = self.index_a[ind]
            self.index_p = self.index_p[ind]
            self.index_n = self.index_n[ind]
            self.key_ap = self.key_ap[ind]
            self.key_n = self.key_n[ind]

            raise StopIteration

        idx_a = self.index_a[self.sample_idx]
        idx_p = self.index_p[self.sample_idx]
        idx_n = self.index_n[self.sample_idx]

        key_ap = self.key_ap[self.sample_idx]
        key_n = self.key_n[self.sample_idx]

        im_a = self.data['patches'][key_ap][idx_a]
        im_p = self.data['patches'][key_ap][idx_p]
        im_n = self.data['patches'][key_n][idx_n]

        self.sample_idx += 1

        return im_a, im_p, im_n

    def test_next(self):
        if self.n_matches == 0 or self.eval_idx >= self.n_matches:
            self.eval_idx = 0
            raise StopIteration

        idx_1, idx_2, is_match = self.data['matches'][self.eval_idx]
        key_1, key_2 = self.data['matches_key'][self.eval_idx]
        im_1 = self.data['patches'][key_1][idx_1]
        im_2 = self.data['patches'][key_2][idx_2]

        keys = list(self.data['patches'].keys())
        idx_key_1 = keys.index(key_1)
        idx_key_2 = keys.index(key_2)


        info = np.zeros_like(im_1)
        info[0,0,0] = idx_1
        info[1,0,0] = idx_2
        info[2,0,0] = is_match

        info[3,0,0] = idx_key_1
        info[4,0,0] = idx_key_2

        self.eval_idx += 1

        return im_1, im_2, info

    def retrieval_next(self):
        if self.n_retrievals == 0 or self.eval_idx >= self.n_retrievals:
            self.eval_idx = 0
            raise StopIteration

        idx, label_idx, is_query = self.data['retrieval'][self.eval_idx]
        key = self.data['retrieval_key'][self.eval_idx]
        im = self.data['patches'][key][idx]

        keys = list(self.data['patches'].keys())
        idx_key = keys.index(key)

        info = np.zeros_like(im)
        info[0,0,0] = is_query
        info[1,0,0] = label_idx
        info[2,0,0] = idx
        info[3,0,0] = idx_key
        self.eval_idx += 1

        return im, im, info

    def get_string_key(self, int_key):
        keys = list(self.data['patches'].keys())
        return keys[int_key]


if __name__ == "__main__":
    base_dir = '/home/sungsooha/Desktop/Data/ftfy/sem/train'

    data_sampler = IoUPatchDataSamplerExt(base_dir)

    data_dirs = []
    for f in os.listdir(base_dir):
        if os.path.isdir(os.path.join(base_dir,f)):
            data_dirs.append(f)
    data_dirs = sorted(data_dirs)

    data_sampler.load_dataset(data_dirs)

    import matplotlib.pyplot as plt
    max_test_count = 10
    count = 0
    # triplet
    print('triplet examples:')
    data_sampler.set_mode(0)
    for a, p, n in data_sampler:
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(np.squeeze(a), cmap='gray')
        ax[1].imshow(np.squeeze(p), cmap='gray')
        ax[2].imshow(np.squeeze(n), cmap='gray')
        plt.show()

        count += 1
        if count == max_test_count:
            break

    # matched
    count = 0
    print('matched pair examples:')
    data_sampler.set_mode(1)
    for a, p, n in data_sampler:
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(np.squeeze(a), cmap='gray')
        ax[1].imshow(np.squeeze(p), cmap='gray')

        idx_1 = int(n[0,0,0])
        idx_2 = int(n[1,0,0])
        is_match = int(n[2,0,0])
        idx_key_1 = int(n[3,0,0])
        idx_key_2 = int(n[4,0,0])

        print("patch id: {}, {}, is_match: {}".format(idx_1, idx_2, is_match))
        print("key: {}, {}".format(
            data_sampler.get_string_key(idx_key_1),
            data_sampler.get_string_key(idx_key_2)
        ))
        plt.show()

        count += 1
        if count == max_test_count:
            break

    # retrieval
    count = 0
    data_sampler.eval_idx = 0
    print('retrieval test examples:')
    data_sampler.set_mode(2)
    for a, p, n in data_sampler:
        fig, ax = plt.subplots()
        ax.imshow(np.squeeze(a), cmap='gray')

        is_query = int(n[0,0,0])
        label_idx = int(n[1,0,0])
        patch_idx = int(n[2,0,0])
        key_idx = int(n[3,0,0])

        print('patch id: {} at {}, {}'.format(
            patch_idx, data_sampler.get_string_key(key_idx), is_query
        ))
        print('label: {}'.format(
            data_sampler.get_labels(label_idx)
        ))
        plt.show()

        count += 1
        if count == max_test_count:
            break
