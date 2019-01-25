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
            mode = True
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

    def _load_patches(self, dir_name, fnames, patch_size, n_channels):
        patches_all = []

        for f in tqdm(fnames, desc='Loading dataset %s' % dir_name):
            assert os.path.isfile(f), 'Not a file: %s' % f
            # todo: what if the maximum value is not 255?
            im = imread(f) / 255.
            patches_row = np.split(im, self.PATCHES_PER_ROW, axis=0)
            for row in patches_row:
                patches = np.split(row, self.PATCHES_PER_COL, axis=1)
                for patch in patches:
                    if patch_size != self.PATCH_SIZE:
                        patch = resize(patch, patch_size)
                    patch_tensor = patch.reshape(*patch_size, n_channels)
                    patches_all.append(patch_tensor)

        return np.asarray(patches_all)

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
        patches = self._load_patches(dir_name, fnames, patch_size, n_channels)
        labels = self._load_labels(dir_name)
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
            print('-- Dataset loaded  : %s' % dir_name)
            print('-- # images        : %s' % len(fnames))
            print('-- # patches       : %s' % N)
            print('-- # labels        : %s' % N)
            print('-- # unique labels : %s' % len(np.unique(self.data['labels'])))
            print('-- # matches       : %s' % self.n_matches)

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
        if self.mode:
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

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    base_dir = '/home/sungsooha/Desktop/Data/ftfy/descriptor'
    base_patch_size = (64,64)
    patches_per_row = 16
    patches_per_col = 16

    batch_size = 5
    patch_size = (64, 64)
    n_channels = 1

    dir_name = 'liberty'

    mode = False

    dataset, data_sampler = input_fn(
        data_dir=base_dir,
        base_patch_size=base_patch_size,
        patches_per_row=patches_per_row,
        patches_per_col=patches_per_col,
        batch_size=batch_size, patch_size=patch_size, n_channels=n_channels
    )

    data_iterator = tf.data.Iterator.from_structure(
        dataset.output_types,
        dataset.output_shapes
    )
    dataset_init = data_iterator.make_initializer(dataset)
    batch_data = data_iterator.get_next()

    # first load dataset to use
    data_sampler.load_dataset(dir_name=dir_name)

    # visualization for visual checking
    plt.ion()
    fig, ax = plt.subplots(batch_size, 3)


    handlers = []
    for _ax in ax:
        h1 = _ax[0].imshow(np.zeros((64, 64), dtype=np.float32), cmap='gray', vmin=0, vmax=1)
        h2 = _ax[1].imshow(np.zeros((64, 64), dtype=np.float32), cmap='gray', vmin=0, vmax=1)
        h3 = _ax[2].imshow(np.zeros((64, 64), dtype=np.float32), cmap='gray', vmin=0, vmax=1)
        _ax[0].axis('off')
        _ax[1].axis('off')
        _ax[2].axis('off')
        handlers.append([h1, h2, h3])

    ax[0,0].set_title('anchor')
    ax[0,1].set_title('positive')
    ax[0,2].set_title('negative')

    plt.ion()
    plt.show()

    with tf.Session() as sess:
        # sample generation can be done just one time with very large number
        # or can be generate new samples each epoch
        if mode:
            data_sampler.generate_triplet(99)
        else:
            data_sampler.set_mode(mode)
        sess.run(dataset_init)

        count = 0
        matched = 0
        patch_indices = []

        try:
            while True:
                a, p, n = sess.run([*batch_data])
                count += len(a)

                if mode:
                    for idx in range(len(a)):
                        _h = handlers[idx]
                        _h[0].set_data(np.squeeze(a[idx]))
                        _h[1].set_data(np.squeeze(p[idx]))
                        _h[2].set_data(np.squeeze(n[idx]))
                else:
                    for idx in range(len(a)):
                        _h = handlers[idx]
                        _h[0].set_data(np.squeeze(a[idx]))
                        _h[1].set_data(np.squeeze(p[idx]))

                        _ax = ax[idx][2]
                        [p.remove() for p in reversed(_ax.texts)]
                        _info = n[idx]
                        _idx_1 = int(_info[0,0,0])
                        _idx_2 = int(_info[1,0,0])
                        _is_match = int(_info[2,0,0])
                        _info_str = '{:d}\n{:d}\n{:s}'.format(
                            _idx_1, _idx_2, 'yes' if _is_match == 1 else 'no'
                        )
                        _ax.text( x=0, y=50, s=_info_str, color='white')
                        matched += _is_match
                        patch_indices.append(_idx_1)
                        patch_indices.append(_idx_2)

                plt.pause(0.001)


        except tf.errors.OutOfRangeError:
            print('Exhausted all samples in the dataset: %d' % count)
            if not mode:
                print('Matched pairs: %d' % matched)
                patch_indices = set(patch_indices)
                print('Number of patches: %d' % len(patch_indices))





