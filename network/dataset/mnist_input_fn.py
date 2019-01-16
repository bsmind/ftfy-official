"""Create the input data pipeline using `tf.data`"""

import tensorflow as tf
import numpy as np
import network.dataset.mnist_dataset as mnist_dataset
from network.dataset.data_sampler import TripletDataSampler
from network.dataset.data_sampler import DataSampler

def get_mnist_train_data(data_dir):
    images_file = mnist_dataset.download(data_dir, "train-images-idx3-ubyte")
    labels_file = mnist_dataset.download(data_dir, "train-labels-idx1-ubyte")

    mnist_dataset.check_image_file_header(images_file)
    mnist_dataset.check_labels_file_header(labels_file)

    # get raw image data
    with tf.gfile.Open(images_file, 'rb') as f:
        mnist_dataset.read32(f) # magic
        num_images = mnist_dataset.read32(f) # num_images
        rows = mnist_dataset.read32(f) # rows
        cols = mnist_dataset.read32(f) # cols

        image_data = np.frombuffer(f.read(), dtype=np.uint8).astype(np.float32)
        image_data = image_data.reshape(num_images, rows, cols, 1)
        image_data /= 255

    # get raw label data
    with tf.gfile.Open(labels_file, 'rb') as f:
        mnist_dataset.read32(f) # magic
        mnist_dataset.read32(f) # num_items

        label_data = np.frombuffer(f.read(), dtype=np.uint8).astype(np.int32)

    return image_data, label_data

def get_mnist_test_data(data_dir):
    images_file = mnist_dataset.download(data_dir, "t10k-images-idx3-ubyte")
    labels_file = mnist_dataset.download(data_dir, "t10k-labels-idx1-ubyte")

    mnist_dataset.check_image_file_header(images_file)
    mnist_dataset.check_labels_file_header(labels_file)

    # get raw image data
    with tf.gfile.Open(images_file, 'rb') as f:
        mnist_dataset.read32(f) # magic
        num_images = mnist_dataset.read32(f) # num_images
        rows = mnist_dataset.read32(f) # rows
        cols = mnist_dataset.read32(f) # cols

        image_data = np.frombuffer(f.read(), dtype=np.uint8).astype(np.float32)
        image_data = image_data.reshape(num_images, rows, cols, 1)
        image_data /= 255

    # get raw label data
    with tf.gfile.Open(labels_file, 'rb') as f:
        mnist_dataset.read32(f) # magic
        mnist_dataset.read32(f) # num_items

        label_data = np.frombuffer(f.read(), dtype=np.uint8).astype(np.int32)

    return image_data, label_data

def preprocess_for_train(image, label):
    shape = image.get_shape().as_list()
    image = tf.pad(image, [[2,2], [2,2], [0,0]])
    image = tf.random_crop(image, shape)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    return image, label

def triplet_preprocess_for_train(a, p, n, label):
    a, _ = preprocess_for_train(a, label)
    p, _ = preprocess_for_train(p, label)
    n, _ = preprocess_for_train(n, label)
    return a, p, n, label

def triplet_preprocess_for_train_dummy(a, p, n, label):
    return a, p, n, label

def input_fn(data_dir, is_training, params):
    """Train input function for the MNIST dataset.

    Args:
        data_dir: (string) path to the data directory
        params: (dict) contains hyperparameters of the model

    Returns:
        dataset: (tf.data.Dataset) input data pipeline
    """
    n_class = 10
    img_shape = (28, 28, 1)

    n_class_per_iter = params.get('n_class_per_iter', None)
    n_img_per_class = params.get('n_img_per_class', None)
    n_iter_per_epoch = params.get('n_iter_per_epoch', None)
    batch_size = params.get('batch_size', None)
    is_online = params.get('is_online', None)
    if None in [n_class_per_iter, n_img_per_class, batch_size, n_iter_per_epoch, is_online]:
        raise ValueError('There is at least one undefined dataset parameter!')

    data = get_mnist_train_data(data_dir) if is_training else get_mnist_test_data(data_dir)

    if is_online:
        data_sampler = DataSampler(
            data=data,
            n_class=n_class,
            n_class_per_iter=n_class_per_iter,
            n_img_per_class=n_img_per_class
        )
        dataset = (tf.data.Dataset
            .from_generator(generator=lambda: data_sampler,
                            output_types=(tf.float32, tf.int32),
                            output_shapes=([batch_size, *img_shape], [batch_size]))
            .take(count=n_iter_per_epoch)
            .flat_map(map_func=lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))
            #.map(preprocess_for_train, 8) # this line for the augmentation if available
            .batch(batch_size=batch_size)
            .prefetch(1)
        )
    else:
        preprocessor = triplet_preprocess_for_train if is_training else triplet_preprocess_for_train_dummy

        data_sampler = TripletDataSampler(
            data=data,
            n_class=n_class,
            n_class_per_iter=n_class_per_iter,
            n_img_per_class=n_img_per_class
        )
        dataset = (tf.data.Dataset
            .from_generator(generator=lambda: data_sampler,
                            output_types=(tf.float32, tf.float32, tf.float32, tf.int32),
                            output_shapes=(
                                [batch_size, *img_shape],
                                [batch_size, *img_shape],
                                [batch_size, *img_shape],
                                [batch_size]
                            ))
            .take(count=n_iter_per_epoch)
            .flat_map(map_func=lambda x, y, z, w: tf.data.Dataset.from_tensor_slices((x, y, z, w)))
            .map(preprocessor, 8) # this line for the augmentation if available
            .shuffle(buffer_size=batch_size*3)
            .batch(batch_size=batch_size)
            .prefetch(1)
        )

    # data_iterator = tf.data.Iterator.from_structure(
    #     dataset.output_types,
    #     dataset.output_shapes
    # )
    # init_op = data_iterator.make_initializer(dataset)
    # batch_data = data_iterator.get_next()

    return dataset



# def testrun_mnist_dataset():
#     from network.dataset.data_sampler import DataSampler
#
#     n_class = 10
#     n_class_per_iter = 10
#     n_img_per_class = 16
#     n_iter_per_epoch = 20
#     batch_size = n_class_per_iter * n_img_per_class
#     img_shape = (28, 28, 1)
#
#     train_data = get_mnist_train_data("../../data")
#     train_data_sampler = DataSampler(
#         train_data,
#         n_class=n_class,
#         n_class_per_iter=n_class_per_iter,
#         n_img_per_class=n_img_per_class
#     )
#
#     train_dataset = (tf.data.Dataset
#         .from_generator(generator=lambda: train_data_sampler,
#                         output_types=(tf.float32, tf.int32),
#                         output_shapes=([batch_size, *img_shape], [batch_size]))
#         .take(count=n_iter_per_epoch)
#         .flat_map(map_func=lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))
#         #.map(preprocess_for_train, 8) # this line for the augmentation if available
#         .batch(batch_size=batch_size)
#         .prefetch(1)
#     )
#
#     data_iterator = tf.data.Iterator.from_structure(
#         train_dataset.output_types,
#         train_dataset.output_shapes
#     )
#
#     train_data_init = data_iterator.make_initializer(train_dataset)
#
#     images, labels = data_iterator.get_next()
#
#     with tf.Session() as sess:
#
#         sess.run(train_data_init)
#
#         step = 0
#         try:
#             while True:
#                 x, y = sess.run([images, labels])
#
#                 print("i: {}, img: {}, labels: {}".format(step, x.shape, y))
#
#                 step += 1
#
#         except tf.errors.OutOfRangeError:
#             print("all data are consumed!")


# def testrun_mnist_triplet_dataset():
#     from network.dataset.data_sampler import TripletDataSampler as DataSampler
#     import matplotlib.pyplot as plt
#
#     n_class = 10
#     n_class_per_iter = 10
#     n_img_per_class = 1
#     n_iter_per_epoch = 20
#     batch_size = n_class_per_iter * n_img_per_class
#     img_shape = (28, 28, 1)
#
#     train_data = get_mnist_train_data("../../data")
#     train_data_sampler = DataSampler(
#         train_data,
#         n_class=n_class,
#         n_class_per_iter=n_class_per_iter,
#         n_img_per_class=n_img_per_class
#     )
#
#     train_dataset = (tf.data.Dataset
#         .from_generator(generator=lambda: train_data_sampler,
#                         output_types=(tf.float32, tf.float32, tf.float32),
#                         output_shapes=(
#                             [batch_size, *img_shape],
#                             [batch_size, *img_shape],
#                             [batch_size, *img_shape]))
#         .take(count=n_iter_per_epoch)
#         .flat_map(map_func=lambda x, y, z: tf.data.Dataset.from_tensor_slices((x, y, z)))
#         #.map(preprocess_for_train, 8) # this line for the augmentation if available
#         .batch(batch_size=batch_size)
#         .prefetch(1)
#     )
#
#     data_iterator = tf.data.Iterator.from_structure(
#         train_dataset.output_types,
#         train_dataset.output_shapes
#     )
#
#     train_data_init = data_iterator.make_initializer(train_dataset)
#
#     anchors, positives, negatives = data_iterator.get_next()
#
#     with tf.Session() as sess:
#
#         sess.run(train_data_init)
#
#         step = 0
#         try:
#             while True:
#                 a, p, n = sess.run([anchors, positives, negatives])
#
#                 print("i: {}, a: {}, p: {}, n: {}".format(
#                     step, a.shape, p.shape, n.shape))
#
#                 fig, axes = plt.subplots(batch_size, 3)
#                 for ax, _a, _p, _n in zip(axes, a, p, n):
#                     ax[0].imshow(np.squeeze(_p))
#                     ax[1].imshow(np.squeeze(_a))
#                     ax[2].imshow(np.squeeze(_n))
#                 plt.show()
#
#                 step += 1
#
#         except tf.errors.OutOfRangeError:
#             print("all data are consumed!")


# if __name__ == "__main__":
#     testrun_mnist_triplet_dataset()

