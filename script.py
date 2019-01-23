import os
import numpy as np
from skimage.io import imread
from tqdm import tqdm

from utils.utils import resize

import matplotlib.pyplot as plt




class PatchData(object):
    def __init__(self, base_dir, test=False):

        self.base_dir = base_dir
        self.test = test
        self.n = 128

        self.PATCH_SIZE = 64
        # the number of patches per row/column in the image
        # containing all the patches
        self.PATCHES_PER_ROW = 16
        self.PATCHES_PER_IMAGE = self.PATCHES_PER_ROW**2

        # the loaded patches
        self.data = dict()

        # triplet samples
        self._index_a = []
        self._index_p = []
        self._index_n = []

    def _get_data(self):
        return self.data

    def _get_patches(self):
        return self._get_data()['patches']

    def _get_matches(self):
        return self._get_data()['matches']

    def _get_labels(self):
        return self._get_data()['labels']

    def _load_image_fnames(self, dir_name):
        files = []
        # find those files with the specified extension
        dataset_dir = os.path.join(self.base_dir, dir_name)
        for f in os.listdir(dataset_dir):
            if f.endswith('bmp'):
                files.append(os.path.join(dataset_dir, f))
        return sorted(files)

    def _load_patches(self, fnames, patch_size=64, n_channels=1, dir_name=''):
        patches_all = []
        fnames = fnames[:self.n] if self.test else fnames

        for f in tqdm(fnames, desc='Loading dataset %s' % dir_name):
            # pick file name
            assert os.path.isfile(f), 'Not a file: %s' % f
            img = imread(f) / 255.
            patches_row = np.split(img, self.PATCHES_PER_ROW, axis=0)
            for row in patches_row:
                patches = np.split(row, self.PATCHES_PER_ROW, axis=1)
                for patch in patches:
                    # resize the patch
                    if patch_size != self.PATCH_SIZE:
                        patch = resize(patch, (patch_size, patch_size))
                    # convert to tensor
                    patch_tensor = patch.reshape(patch_size, patch_size, n_channels)
                    patches_all.append(patch_tensor)

        patches_all = np.asarray(patches_all) if not self.test \
            else np.asarray(patches_all[:self.n])
        return patches_all

    def _load_labels(self, dir_name):
        info_fname = os.path.join(self.base_dir, dir_name, 'info.txt')
        assert os.path.isfile(info_fname), 'Not a file: %s' % info_fname

        labels = []
        with open(info_fname, 'r') as f:
            for line in f:
                labels.append(int(line.split()[0]))

        return np.asarray(labels) if not self.test \
                else np.asarray(labels[:self.n])

    def _load_matches(self, dir_name):
        fname = os.path.join(self.base_dir, dir_name, 'm50_100000_100000_0.txt')
        assert os.path.isfile(fname), 'Not a file: %s' % fname

        # read file and keep only 3D point ID and 1 if same, otherwise 0
        matches = []
        with open(fname, 'r') as f:
            for line in f:
                l = line.split()
                matches.append([int(l[0]), int(l[3]), int(l[1] == l[4])])

        return np.asarray(matches)

    def load_by_dirname(self, dir_name, patch_size=64, n_channels=1, debug=True):
        assert os.path.exists(os.path.join(self.base_dir, dir_name)) == True, \
                "The dataset directory does not exist: %s" % dir_name

        fnames = self._load_image_fnames(dir_name=dir_name)
        patches = self._load_patches(fnames,
                                     patch_size=patch_size,
                                     n_channels=n_channels,
                                     dir_name=dir_name)
        labels = self._load_labels(dir_name=dir_name)
        matches = self._load_matches(dir_name=dir_name)

        N = min(len(labels), len(patches))
        self.data['patches'] = patches[:N]
        self.data['labels'] = labels[:N]
        self.data['matches'] = matches

        if debug:
            print('-- Dataset loaded: %s' % dir_name)
            print('-- Number of images: %s' % len(fnames))
            print('-- Number of patches: %s' % len(self.data['patches']))
            print('-- Number of labels: %s' % len(self.data['labels']))
            print('-- Number of ulabels: %s' % len(np.unique(labels)))
            print('-- Number of matches: %s' % len(matches))

    def generate_triplet(self, n_samples):
        # label --> list of index belonging to the label
        label_to_ind = dict()
        for idx, label in enumerate(self._get_labels()):
            if label_to_ind.get(label, None) is None:
                label_to_ind[label] = [idx]
            else:
                label_to_ind[label].append(idx)

        # list of a unique labels (skip labels containing less than 2 samples)
        ulabels = []
        for label in label_to_ind.keys():
            if len(label_to_ind[label]) < 2:
                continue
            ulabels.append(label)
        ulabels = np.array(ulabels)
        assert len(ulabels) >= 2, 'At least 2 unique labels needed (%d)!' % len(ulabels)

        # triplet ids
        self._index_a = [] # list of anchor index
        self._index_p = [] # list of positive index
        self._index_n = [] # list of negative index

        # Generating triplets
        label_p = np.ones(len(ulabels), dtype=np.float32)
        for _ in range(n_samples):
            if label_p.sum() < 2:
                label_p[:] = 1.

            # randomly pick one label for anchor and positive
            label_idx = np.random.choice(len(label_p), replace=False, p=label_p/label_p.sum())
            label = ulabels[label_idx]
            label_p[label_idx] = 0.

            # randomly pick ancher and positive
            idx_a, idx_p = np.random.choice(label_to_ind[label], size=2, replace=False)
            self._index_a.append(idx_a)
            self._index_p.append(idx_p)

            # randomly pick negative
            label_idx = np.random.choice(len(label_p), replace=False, p=label_p/label_p.sum())
            label = ulabels[label_idx]
            label_p[label_idx] = 0.
            idx_n = np.random.choice(label_to_ind[label], replace=False)
            self._index_n.append(idx_n)

        # convert to numpy array
        self._index_a = np.array(self._index_a)
        self._index_p = np.array(self._index_p)
        self._index_n = np.array(self._index_n)
        assert len(self._index_a) == len(self._index_p) == len(self._index_n), \
            'Unmatched length of triplet samples among (a, p, n)!'

        # shuffle
        n_samples = len(self._index_a)
        temp_index = np.arange(n_samples)
        np.random.shuffle(temp_index)
        self._index_a = self._index_a[temp_index]
        self._index_p = self._index_p[temp_index]
        self._index_n = self._index_n[temp_index]

    def next_batch(self, batch_size):
        idx_a = self._index_a[:batch_size]
        idx_p = self._index_p[:batch_size]
        idx_n = self._index_n[:batch_size]

        patches = self._get_patches()
        patches_a = patches[idx_a]
        patches_p = patches[idx_p]
        patches_n = patches[idx_n]

        labels = self._get_labels()
        labels_a = labels[idx_a]
        labels_p = labels[idx_p]
        labels_n = labels[idx_n]

        return (patches_a, patches_p, patches_n), (labels_a, labels_p, labels_n)

data_dir = '/home/sungsooha/Desktop/Data/ftfy/descriptor'
filename = 'patches0000.bmp'
patch_size = (64, 64)

patch_data = PatchData(data_dir, test=False)
patch_data.load_by_dirname(dir_name='liberty')
patch_data.generate_triplet(100)

batch_size = 10
batch_images, batch_labels = patch_data.next_batch(batch_size)

im_a, im_p, im_n = batch_images
la_a, la_p, la_n = batch_labels

fig, ax = plt.subplots(batch_size, 3)

for idx in range(batch_size):
    _ax = ax[idx]

    _ax[0].imshow(np.squeeze(im_a[idx]), cmap='gray')
    _ax[0].axis('off')
    _ax[0].set_title(la_a[idx])

    _ax[1].imshow(np.squeeze(im_p[idx]), cmap='gray')
    _ax[1].axis('off')
    _ax[1].set_title(la_p[idx])

    _ax[2].imshow(np.squeeze(im_n[idx]), cmap='gray')
    _ax[2].axis('off')
    _ax[2].set_title(la_n[idx])

plt.show()