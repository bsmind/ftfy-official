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
            mode = 0
    ):
        super().__init__(base_dir, base_patch_size, patches_per_row, patches_per_col, mode)

        self.max_n_triplet_samples = 0
        self.max_n_matched_pairs = 0

    def set_n_triplet_samples(self, n):
        self.n_triplet_samples = min(n, self.max_n_triplet_samples)

    def set_n_matched_pairs(self, n):
        self.n_matches = min(2*n, self.max_n_matched_pairs)

    def get_patch_by_retrieval_idx(self, idx):
        patch_idx = self.data['retrieval'][idx][0]
        return self.data['patches'][patch_idx], patch_idx

    def _load_triplet_samples(self, dir_name, fname='triplet_1000000.txt'):
        triplet_fname = os.path.join(self.base_dir, dir_name, fname)
        assert os.path.isfile(triplet_fname), 'Cannot find %s!' % fname

        self.index_a = []
        self.index_p = []
        self.index_n = []

        with open(triplet_fname, 'r') as f:
            for line in tqdm(f, 'Load triplet samples'):
                idx_a, idx_p, idx_n = line.split()
                self.index_a.append(int(idx_a))
                self.index_p.append(int(idx_p))
                self.index_n.append(int(idx_n))

        # convert to numpy array
        self.index_a = np.array(self.index_a)
        self.index_p = np.array(self.index_p)
        self.index_n = np.array(self.index_n)

        self.n_triplet_samples = len(self.index_a)
        self.max_n_triplet_samples = self.n_triplet_samples
        self.sample_idx = 0

    def _load_matched_pairs(self, dir_name, fname='matched_50000.txt'):
        matched_fname = os.path.join(self.base_dir, dir_name, fname)
        assert os.path.isfile(matched_fname), 'Cannot find %s!' % fname

        matches = []
        with open(matched_fname, 'r') as f:
            for line in tqdm(f, 'Load matched pairs'):
                idx_1, idx_2, is_match = line.split()
                matches.append([int(idx_1), int(idx_2), int(is_match)])

        matches = np.asarray(matches)
        self.data['matches'] = matches
        self.n_matches = len(matches)
        self.max_n_matched_pairs = self.n_matches

    def _load_retrieval_set(self, dir_name, fname='retrieval.txt'):
        retrieval_fname = os.path.join(self.base_dir, dir_name, fname)
        assert os.path.isfile(retrieval_fname), 'Cannot find %s!' % fname

        retrievals = []
        labels = []
        counter = 0
        with open(retrieval_fname, 'r') as f:
            for line in tqdm(f, 'Load retrieval set'):
                idx, label, is_query = line.split()
                retrievals.append([int(idx), counter, int(is_query)])
                labels.append(label)
                counter += 1

        self.data['retrieval'] = np.asarray(retrievals)
        self.data['labels'] = np.asarray(labels)
        self.n_retrievals = len(labels)


    def load_dataset(self, dir_name, ext='bmp', patch_size=(64, 64), n_channels=1, debug=True):
        assert os.path.exists(os.path.join(self.base_dir, dir_name)) == True, \
            'The dataset directory does not exist: %s' % dir_name

        fnames = self._load_image_fnames(dir_name, ext)
        n_patches = self.PATCHES_PER_ROW * self.PATCHES_PER_COL * len(fnames)
        patches = self._load_patches(dir_name, fnames, patch_size, n_channels, n_patches)
        self._load_triplet_samples(dir_name)
        self._load_matched_pairs(dir_name)
        self._load_retrieval_set(dir_name)

        # initialize patch dataset
        self.data['patches'] = patches

        if debug:
            output = print
            #output = tf.logging.info
            output('-- Dataset loaded   : %s' % dir_name)
            output('-- # patches        : %s' % len(patches))
            output('-- # triplet samples: %s' % self.n_triplet_samples)
            output('-- # matched pairs  : %d' % (self.n_matches//2))
            output('-- # retrieval set  : %d' % self.n_retrievals)

    def retrieval_next(self):
        if self.n_retrievals == 0 or self.eval_idx >= self.n_retrievals:
            self.eval_idx = 0
            raise StopIteration

        idx, label_idx, is_query = self.data['retrieval'][self.eval_idx]
        im = self.data['patches'][idx]

        info = np.zeros_like(im)
        info[0,0,0] = is_query
        info[1,0,0] = label_idx
        info[2,0,0] = idx
        self.eval_idx += 1

        return im, im, info

    def __next__(self):
        if self.mode == 0:
            return self.train_next()
        elif self.mode == 1:
            return self.test_next()
        elif self.mode == 2:
            return self.retrieval_next()
        else:
            raise ValueError('Unknown mode: ', self.mode)


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

























