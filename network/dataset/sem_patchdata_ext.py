import os
import numpy as np
import tensorflow as tf

from network.dataset.sem_patchdata import IoUPatchDataSampler

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

    def load_dataset(self, dir_name, ext='bmp', patch_size=(128, 128), n_channels=1, debug=True):
        assert isinstance(dir_name, list), 'It expects to get list of directory names'

        def get_n_patches(d):
            file = open(os.path.join(self.base_dir, d, 'patches', 'info.txt'), 'r')
            count = 0
            for _ in file: count += 1
            return count

        patches = []
        total_n_patches = 0
        for d in dir_name:
            n_patches = get_n_patches(d)
            fnames = self._load_image_fnames(os.path.join(d, 'patches'), ext)
            patches.append(self._load_patches(
                os.path.join(d, 'patches'),
                fnames, patch_size, n_channels, n_patches
            ))
            total_n_patches += n_patches
        patches = np.concatenate(patches, axis=0)
        #print(patches.shape, total_n_patches)

        self._load_triplet_samples('.', fname='triplet_1000000.txt')
        self._load_matched_pairs('.')
        self._load_retrieval_set('.')

        self.data['patches'] = patches

        if debug:
            #output = print
            output = tf.logging.info
            output('-- Dataset loaded : {}'.format(dir_name))
            output('-- # patches      : %s' % total_n_patches)
            output('-- # triplet samples: %s' % self.n_triplet_samples)
            output('-- # matched pairs  : %d' % (self.n_matches//2))
            output('-- # retrieval set  : %d' % self.n_retrievals)


def input_fn(
        data_dir,
        base_patch_size, patches_per_row, patches_per_col,
        batch_size, patch_size, n_channels):

    data_sampler = IoUPatchDataSamplerExt(
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
    max_test_count = 100
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

        print("patch id: {}, {}, is_match: {}".format(idx_1, idx_2, is_match))
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

        print('patch id: {}, label: {}, is_query: {}'.format(
            patch_idx, data_sampler.get_labels(label_idx), is_query))
        plt.show()

        count += 1
        if count == max_test_count:
            break