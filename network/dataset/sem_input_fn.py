import tensorflow as tf
import os
#from network.dataset.data_sampler import DataSamplerIoU
from network.dataset.data_sampler_ext import DataSamplerIoUExt
from network.dataset.data_sampler import TestDataSamplerIoU
from utils.utils import get_image_list, load_image

def train_process(anchors, positives, negatives):
    # scale invariance (multi-scale)
    # 1. randomly down sample
    # tf.random_uniform([], minval=, maxval=)
    # tf.nn.avg_pool(value, ksize, strides, padding)
    # tf.image.resize_bilinear()

    # rotation invariance -- random 0.5 probability
    #tf.image.random_flip_up_down(anchors)
    #tf.image.random_flip_left_right(anchors)

    # brightness and saturation
    #tf.image.random_brightness(anchors)
    #tf.image.random_saturation(anchors)

    # after augmentation, it's still in [0, 1]
    #tf.clip_by_value(anchors, 0., 1.)

    return anchors, positives, negatives

def input_fn(data_dir, params, aug_kwarg={}, file_list=None, is_training=True):

    patch_size = params.get('patch_size', None)
    n_channels = params.get('n_channels', None)
    iou_thresholds = params.get('iou_thresholds', [0.7, 0.1])
    n_img_per_iter = params.get('n_img_per_iter', 10)
    n_crops_per_img = params.get('n_crops_per_img', 10)
    batch_size = params.get('batch_size', 100)

    if None in [patch_size, n_channels]:
        raise ValueError('There is at least one undefined dataset parameter!')

    if is_training:
        data = get_image_list(data_dir, file_list)
        data_sampler = DataSamplerIoUExt(
            data=data,
            n_channels=n_channels,
            patch_size=patch_size,
            iou_thresholds=iou_thresholds,
            n_img_per_iter=n_img_per_iter,
            n_crops_per_img=n_crops_per_img,
            aug_kwarg=aug_kwarg
        )
        dataset = (tf.data.Dataset
                   .from_generator(generator=lambda: data_sampler,
                                   output_types=(tf.float32, tf.float32, tf.float32),
                                   output_shapes=(
                                       [n_img_per_iter*n_crops_per_img, *patch_size, n_channels],
                                       [n_img_per_iter*n_crops_per_img, *patch_size, n_channels],
                                       [n_img_per_iter*n_crops_per_img, *patch_size, n_channels],
                                   ))
                   .take(count=10)
                   .flat_map(map_func=lambda x, y, z: tf.data.Dataset.from_tensor_slices((x, y, z)))
                   # .map(preprocessor, 8)
                   .shuffle(buffer_size=batch_size*2)
                   .batch(batch_size=batch_size)
                   .prefetch(1)
                   )
    else:
        data_sampler = TestDataSamplerIoU(patch_size=patch_size)
        dataset = (
            tf.data.Dataset()
                .from_generator(generator=lambda: data_sampler,
                                output_types=(tf.float32, tf.float32, tf.float32),
                                output_shapes=(
                                    [*patch_size, n_channels],
                                    [*patch_size, n_channels],
                                    [*patch_size, n_channels]
                                ))
                .batch(batch_size=batch_size)
                .prefetch(1)
        )

    return dataset, data_sampler



if __name__ == "__main__":
    import network.dataset.sem_dataset as sem

    data_dir = '/home/sungsooha/Desktop/Data/ftfy/data_hxnsem_selected'

    files_by_zoom = sem.filelist_by_zoomfactor(data_dir)
    tr_list, t_list = sem.split_train_test(files_by_zoom, train_ratio=.8)

    img_idx = 0
    fn = t_list[img_idx]
    if fn.startswith(os.sep):
        fn = fn[len(os.sep):]
    img = load_image(os.path.join(data_dir, fn))
    #gen = my_generator(img, (208, 208), stride=10)

    patch_size = (208, 208)
    stride = (10,10)
    nchnnels = 1
    data_sampler = TestDataSamplerIoU(patch_size=patch_size, stride=stride)


    dataset = (
        tf.data.Dataset()
            .from_generator(generator=lambda: data_sampler,
                            output_types=(tf.float32, tf.float32, tf.float32),
                            output_shapes=(
                                [*patch_size, nchnnels],
                                [*patch_size, nchnnels],
                                [*patch_size, nchnnels]
                            ))
            .batch(batch_size=16)
            .prefetch(1)
    )
    data_iterator = tf.data.Iterator.from_structure(
        dataset.output_types,
        dataset.output_shapes
    )
    dataset_init = data_iterator.make_initializer(dataset)
    batch_data = data_iterator.get_next()

    with tf.Session() as sess:
        for i in range(1):
            data_sampler.reset(img)
            sess.run(dataset_init)

            try:
                while True:
                    p, r, c = sess.run([*batch_data])
                    print(p.shape, r[:, 0, 0, 0], c[:, 0, 0, 0])
            except tf.errors.OutOfRangeError:
                pass

            print(['-']*40)

    # test with input_fn above
    # dataset_kwargs = {
    #     'patch_size': (208, 208),
    #     'n_channels': 1,
    #     'iou_thresholds': [0.7, 0.1],
    #     'n_img_per_iter': 20,
    #     'n_crops_per_img': 20,
    #     'batch_size': 5
    # }
    # dataset = input_fn(data_dir, t_list, dataset_kwargs)
    # data_iterator = tf.data.Iterator.from_structure(
    #     dataset.output_types,
    #     dataset.output_shapes
    # )
    # dataset_init = data_iterator.make_initializer(dataset)
    # batch_data = data_iterator.get_next()
    #
    # with tf.Session() as sess:
    #     sess.run(dataset_init)
    #
    #     try:
    #         while True:
    #             a, p, n = sess.run([*batch_data])
    #
    #             fig, ax = plt.subplots(len(a), 3)
    #             for _ax, _a, _p, _n in zip(ax, a, p, n):
    #                 _ax[0].imshow(np.squeeze(_a), vmin=0, vmax=1)
    #                 _ax[1].imshow(np.squeeze(_p), vmin=0, vmax=1)
    #                 _ax[2].imshow(np.squeeze(_n), vmin=0, vmax=1)
    #             plt.show()
    #
    #     except tf.errors.OutOfRangeError:
    #         pass
