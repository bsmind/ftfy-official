import numpy as np
import tensorflow as tf
from network.dataset.patchdata import input_fn as ubc_input_fn
from network.dataset.sem_patchdata import input_fn as sem_input_fn

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    use_ubc = False
    if use_ubc:
        base_dir = '/home/sungsooha/Desktop/Data/ftfy/descriptor'
        base_patch_size = (64,64)
        patches_per_row = 16
        patches_per_col = 16
        dir_name = 'liberty'
        input_fn = ubc_input_fn
    else:
        base_dir = '/home/sungsooha/Desktop/Data/ftfy/austin'
        base_patch_size = (64,64)
        patches_per_row = 13
        patches_per_col = 8
        dir_name = 'scene_patch'
        input_fn = sem_input_fn

    batch_size = 5
    patch_size = (64, 64)
    n_channels = 1



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
    if not use_ubc:
        data_sampler.generate_match_pairs(1000)

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



