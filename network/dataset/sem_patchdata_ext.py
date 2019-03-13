import os
import numpy as np
import tensorflow as tf

from network.dataset.sem_patchdata import IoUPatchDataSampler

class IoUPatchDataSamplerExt(IoUPatchDataSampler):
    def __init__(
            self,
            base_dir, train_fnames:dict, test_fnames:dict=None,
            base_patch_size=(128,128),
            patches_per_col=10,
            patches_per_row=10,
            mode=0,
            eval_mode=True # use train eval. dataset for True
    ):
        super().__init__(base_dir, base_patch_size, patches_per_col, patches_per_row, mode)

        self.train_fnames = train_fnames
        self.test_fnames = test_fnames
        self.eval_mode = eval_mode

        # matched pairs for training
        self.train_matches = None
        self.train_n_matches = 0

        # retrieval examples for training
        self.train_retrievals = None
        self.train_labels = None
        self.train_n_retrievals = 0

        # matched pairs for test
        self.test_matches = None
        self.test_n_matches = 0

        # retrieval examples for test
        self.test_retrievals = None
        self.test_labels = None
        self.test_n_retrievals = 0

    def set_eval_mode(self, eval_mode):
        if eval_mode or self.test_fnames is None:
            self.data['matches'] = self.train_matches
            self.data['retrieval'] = self.train_retrievals
            self.data['labels'] = self.train_labels
            self.max_n_matched_pairs = self.train_n_matches
            self.n_matches = self.train_n_matches
            self.n_retrievals = self.train_n_retrievals
            self.eval_mode = True
        else:
            self.data['matches'] = self.test_matches
            self.data['retrieval'] = self.test_retrievals
            self.data['labels'] = self.test_labels
            self.max_n_matched_pairs = self.test_n_matches
            self.n_matches = self.test_n_matches
            self.n_retrievals = self.test_n_retrievals
            self.eval_mode = False

    def load_dataset(
            self, dir_name,
            ext='bmp', patch_size=(128, 128), n_channels=1,
            patch_dir='patches',
            debug=True
    ):
        assert isinstance(dir_name, list), 'It expects to get list of directory names'

        def get_n_patches(d):
            file = open(os.path.join(self.base_dir, d, patch_dir, 'info.txt'), 'r')
            count = 0
            for _ in file: count += 1
            return count

        patches = []
        total_n_patches = 0
        for d in dir_name:
            n_patches = get_n_patches(d)
            fnames = self._load_image_fnames(os.path.join(d, patch_dir), ext)
            patches.append(self._load_patches(
                os.path.join(d, 'patches'),
                fnames, patch_size, n_channels, n_patches
            ))
            total_n_patches += n_patches
        patches = np.concatenate(patches, axis=0)
        #print(patches.shape, total_n_patches)

        # for training
        self._load_triplet_samples('.', fname=self.train_fnames['triplet'])
        matches = self._load_matched_pairs('.', fname=self.train_fnames['matched'])
        retrievals, labels = self._load_retrieval_set('.', fname=self.train_fnames['retrieval'])

        self.train_matches = matches
        self.train_n_matches = len(matches)

        self.train_retrievals = retrievals
        self.train_labels = labels
        self.train_n_retrievals = len(labels)

        # for testing
        if self.test_fnames is not None:
            matches = self._load_matched_pairs('.', fname=self.test_fnames['matched'])
            retrievals, labels = self._load_retrieval_set('.', fname=self.test_fnames['retrieval'])

            self.test_matches = matches
            self.test_n_matches = len(matches)

            self.test_retrievals = retrievals
            self.test_labels = labels
            self.test_n_retrievals = len(labels)

        self.data['patches'] = patches
        self.set_eval_mode(True)

        if debug:
            #output = print
            output = tf.logging.info
            output('-- Dataset loaded : {}'.format(dir_name))
            output('-- # patches      : %s' % total_n_patches)
            output('-- # triplet samples: %s' % self.n_triplet_samples)
            output('-- Evaluation on Training dataset:')
            output('-- # matched pairs  : %d' % (self.train_n_matches//2))
            output('-- # retrieval set  : %d' % self.train_n_retrievals)
            output('-- Evaluation on Test dataset:')
            output('-- # matched pairs  : %d' % (self.test_n_matches//2))
            output('-- # retrieval set  : %d' % self.test_n_retrievals)


def input_fn(
        data_dir, train_fnames, test_fnames,
        base_patch_size, patches_per_row, patches_per_col,
        batch_size, patch_size, n_channels):

    data_sampler = IoUPatchDataSamplerExt(
        base_dir=data_dir,
        train_fnames=train_fnames,
        test_fnames=test_fnames,
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

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    base_dir = '/home/sungsooha/Desktop/Data/ftfy/sem/train'

    data_sampler = IoUPatchDataSamplerExt(
        base_dir,
        train_fnames=dict(triplet='train_triplet.txt',
                          matched='train_matched.txt',
                          retrieval='train_retrieval.txt'),
        test_fnames=dict(matched='test_matched.txt', retrieval='test_retrieval.txt')
    )

    data_dirs = []
    for f in os.listdir(base_dir):
        if os.path.isdir(os.path.join(base_dir,f)):
            data_dirs.append(f)
    data_dirs = sorted(data_dirs)

    data_sampler.load_dataset(data_dirs)

    import matplotlib.pyplot as plt
    max_test_count = 10
    for iii in range(2):
        data_sampler.set_eval_mode(iii==0)

        # triplet
        print('triplet examples:')
        count = 0
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
        print('matched pair examples:')
        count = 0
        data_sampler.set_mode(1)
        for a, p, n in data_sampler:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(np.squeeze(a), cmap='gray')
            ax[1].imshow(np.squeeze(p), cmap='gray')

            idx_1 = int(n[0,0,0])
            idx_2 = int(n[1,0,0])
            is_match = int(n[2,0,0])

            print("patch id: {}, {}, is_match: {}".format(idx_1, idx_2, is_match))
            plt.show()

            count += 1
            if count == max_test_count:
                break

        # retrieval
        print('retrieval test examples:')
        count = 0
        data_sampler.eval_idx = 0
        data_sampler.set_mode(2)
        for a, p, n in data_sampler:
            fig, ax = plt.subplots()
            ax.imshow(np.squeeze(a), cmap='gray')

            is_query = int(n[0,0,0])
            label_idx = int(n[1,0,0])
            patch_idx = int(n[2,0,0])

            print('patch id: {}, label: {}, is_query: {}'.format(
                patch_idx, data_sampler.get_labels(label_idx), is_query))
            plt.show()

            count += 1
            if count == max_test_count:
                break