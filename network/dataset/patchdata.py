import os
import numpy as np
import tensorflow as tf
from skimage.io import imread
from tqdm import tqdm

from utils.utils import resize

class PatchDataSampler(object):
    def __init__(
            self,
            base_dir,
            base_patch_size=(64, 64),
            patches_per_row=16,
            patches_per_col=16,
            mode = 0
    ):
        self.base_dir = base_dir

        # True for training, False for evaluation
        self.mode = mode

        # a patch image configuration
        # an image contains (row x col) patches
        # where each patch is size of `patch_size`
        self.PATCH_SIZE = base_patch_size
        self.PATCHES_PER_ROW = patches_per_row
        self.PATCHES_PER_COL = patches_per_col

        # the loaded patches
        self.data = dict()
        self.label_to_ind = dict() # aux, label: list of index belonging to the label
        self.ulabels = [] # aux, list of a unique labels containing at least 2 samples)

        # index of samples
        self.sample_idx = 0
        self.n_triplet_samples = 0
        self.index_a = [] # anchor
        self.index_p = [] # positive
        self.index_n = [] # negative

        # for evaluation
        self.eval_idx = 0
        self.n_matches = 0

    def set_mode(self, mode):
        self.mode = mode

    def get_labels(self, indices):
        return self.data['labels'][indices]

    def _load_image_fnames(self, dir_name, ext='bmp'):
        files = []
        # find those files with the specified extension
        dataset_dir = os.path.join(self.base_dir, dir_name)
        for f in os.listdir(dataset_dir):
            if f.endswith(ext):
                files.append(os.path.join(dataset_dir, f))
        return sorted(files)

    def _load_patches(self, dir_name, fnames, patch_size, n_channels, n_patches=None):
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
            patches_row = np.split(im, self.PATCHES_PER_ROW, axis=0)
            for row in patches_row:
                if done: break
                patches = np.split(row, self.PATCHES_PER_COL, axis=1)
                for patch in patches:
                    if done: break
                    if patch_size != self.PATCH_SIZE:
                        patch = resize(patch, patch_size)
                    patch_tensor = patch.reshape(*patch_size, n_channels)
                    if n_patches is None:
                        patches_all.append(patch_tensor)
                    else:
                        patches_all[counter] = patch_tensor
                    counter += 1
                    if counter >= n_patches:
                        done = True
        if n_patches == np.inf:
            patches_all = np.asarray(patches_all)
        return patches_all

    def _load_labels(self, dir_name, fname='info.txt'):
        info_name = os.path.join(self.base_dir, dir_name, fname)
        assert os.path.isfile(info_name), 'Not a file: %s' % info_name

        labels = []
        with open(info_name, 'r') as f:
            for line in f:
                labels.append(int(line.split()[0]))

        return np.asarray(labels)

    def _load_matches(self, dir_name, fname='m50_100000_100000_0.txt'):
        match_name = os.path.join(self.base_dir, dir_name, fname)
        assert os.path.isfile(match_name), 'Not a file %s' % match_name

        # (wrong) read file and keep only 3D point ID and 1 if same, otherwise 0
        # keep only patch ID, and label is 1 if 3D point ID is same; otherwise 0
        matches = []
        with open(match_name, 'r') as f:
            for line in f:
                l = line.split()
                matches.append([int(l[0]), int(l[3]), int(l[1] == l[4])])

        return np.asarray(matches)

    def load_dataset(self, dir_name, ext='bmp', patch_size=(64,64), n_channels=1, debug=True):
        assert os.path.exists(os.path.join(self.base_dir, dir_name)) == True, \
            'The dataset directory does not exist: %s' % dir_name

        fnames = self._load_image_fnames(dir_name, ext)
        labels = self._load_labels(dir_name)
        patches = self._load_patches(dir_name, fnames, patch_size, n_channels)
        matches = self._load_matches(dir_name)

        # initialize patch dataset
        N = min(len(labels), len(patches))
        self.data['patches'] = patches[:N]
        self.data['labels'] = labels[:N]
        self.data['matches'] = matches
        self.n_matches = len(matches)

        # initialize `label_to_ind` hashmap
        self.label_to_ind = dict()
        for idx, label in enumerate(self.data['labels']):
            if self.label_to_ind.get(label, None) is None:
                self.label_to_ind[label] = [idx]
            else:
                self.label_to_ind[label].append(idx)

        # initialize unique labels
        self.ulabels = []
        for label in self.label_to_ind.keys():
            if len(self.label_to_ind[label]) < 2:
                continue
            self.ulabels.append(label)
        self.ulabels = np.array(self.ulabels)

        if debug:
            tf.logging.info('-- Dataset loaded : %s' % dir_name)
            tf.logging.info('-- # images        : %s' % len(fnames))
            tf.logging.info('-- # patches       : %s' % N)
            tf.logging.info('-- # labels        : %s' % N)
            tf.logging.info('-- # unique labels : %s' % len(np.unique(self.data['labels'])))
            tf.logging.info('-- # matches       : %s' % self.n_matches)

    def generate_triplet(self, n_samples):
        self.index_a = []
        self.index_p = []
        self.index_n = []

        label_p = np.ones(len(self.ulabels), dtype=np.float32)
        for _ in tqdm(range(n_samples), desc='Generating triplets'):
            if label_p.sum() < 2.:
                label_p[:] = 1

            # randomly pick one label for anchor and positive
            label_idx = np.random.choice(len(label_p), replace=False, p=label_p/label_p.sum())
            label = self.ulabels[label_idx]
            label_p[label_idx] = 0.

            # randomly pick anchor and positive
            idx_a, idx_p = np.random.choice(self.label_to_ind[label], size=2, replace=False)
            self.index_a.append(idx_a)
            self.index_p.append(idx_p)

            # randomly pick negative
            label_idx = np.random.choice(len(label_p), replace=False, p=label_p/label_p.sum())
            label = self.ulabels[label_idx]
            label_p[label_idx] = 0.
            idx_n = np.random.choice(self.label_to_ind[label], replace=False)
            self.index_n.append(idx_n)

        # convert to numpy array
        self.index_a = np.array(self.index_a)
        self.index_p = np.array(self.index_p)
        self.index_n = np.array(self.index_n)

        # shuffle
        # shuffling is moved to tf.dataset
        self.n_triplet_samples = len(self.index_a)
        # temp_ind = np.arange(self.n_triplet_samples)
        # np.random.shuffle(temp_ind)
        # self.index_a = self.index_a[temp_ind]
        # self.index_p = self.index_p[temp_ind]
        # self.index_n = self.index_n[temp_ind]

        # initialize sample index
        self.sample_idx = 0

    def generate_stats(self):
        mean = np.mean(self.data['patches'])
        std = np.std(self.data['patches'])
        return mean, std

    def normalize_data(self, mean, std):
        for i in tqdm(range(len(self.data['patches'])), desc='Normalizing data'):
            self.data['patches'][i] = (self.data['patches'][i] - mean) / std

    def __iter__(self):
        return self

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

            raise StopIteration

        idx_a = self.index_a[self.sample_idx]
        idx_p = self.index_p[self.sample_idx]
        idx_n = self.index_n[self.sample_idx]

        im_a = self.data['patches'][idx_a]
        im_p = self.data['patches'][idx_p]
        im_n = self.data['patches'][idx_n]

        self.sample_idx += 1

        return im_a, im_p, im_n

    def test_next(self):
        if self.n_matches == 0 or self.eval_idx >= self.n_matches:
            self.eval_idx = 0
            raise StopIteration

        idx_1, idx_2, is_match = self.data['matches'][self.eval_idx]
        im_1 = self.data['patches'][idx_1]
        im_2 = self.data['patches'][idx_2]

        info = np.zeros_like(im_1)
        info[0,0,0] = idx_1
        info[1,0,0] = idx_2
        info[2,0,0] = is_match
        self.eval_idx += 1

        return im_1, im_2, info

    def __next__(self):
        if self.mode == 0:
            return self.train_next()
        return self.test_next()


def input_fn(
        data_dir,
        base_patch_size, patches_per_row, patches_per_col,
        batch_size, patch_size, n_channels):
    data_sampler = PatchDataSampler(
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



