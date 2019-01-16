import numpy as np
import tensorflow as tf
from utils.Param import Param
from network.dataset.sem_data_sampler import train_input_fn, test_input_fn, load_full_dataset

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def test_grid_sampling():
    p = Param()
    p.test_datasets = '../../../data/test_sem_dataset_208.npy'

    with tf.device('/cpu:0'), tf.name_scope('input'):
        test_dataset, test_data_sampler = test_input_fn(
            patch_size=p.patch_size,
            n_channels=p.n_channels,
            batch_size=p.batch_size
        )
        data_iterator = tf.data.Iterator.from_structure(
            test_dataset.output_types,
            test_dataset.output_shapes
        )
        test_dataset_init = data_iterator.make_initializer(test_dataset)
        batch_data = data_iterator.get_next()

    test_filenames, test_xy, test_images = load_full_dataset(p.data_dir, p.test_datasets)

    with tf.Session() as sess:
        im = test_images[0]

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(np.squeeze(im), vmin=0, vmax=1)
        ax[0].axis('off')

        h = ax[1].imshow(np.zeros(p.patch_size, dtype=np.float32), vmin=0, vmax=1)
        ax[1].axis('off')

        rect = Rectangle(
            xy=(0,0), width=p.patch_size[1], height=p.patch_size[0],
            facecolor='none', edgecolor='red', linewidth=1
        )
        ax[0].add_patch(rect)

        plt.ion()
        plt.show()

        test_data_sampler.set_image(im)
        test_data_sampler.set_sampling_method(True, stride=p.stride)
        sess.run(test_dataset_init)
        try:
            while True:
                a, x0, y0 = sess.run([*batch_data])
                x0 = x0.astype(np.int32)
                x0 = x0[:, 0, 0, 0]
                y0 = y0.astype(np.int32)
                y0 = y0[:, 0, 0, 0]
                for _a, _x0, _y0 in zip(a, x0, y0):
                    h.set_data(np.squeeze(_a))
                    rect.set_xy((_x0, _y0))
                    plt.draw()
                    plt.pause(0.1)
        except tf.errors.OutOfRangeError:
            pass

def test_xy_sampling():
    p = Param()
    p.test_datasets = '../../../data/test_sem_dataset_208.npy'

    with tf.device('/cpu:0'), tf.name_scope('input'):
        test_dataset, test_data_sampler = test_input_fn(
            patch_size=p.patch_size,
            n_channels=p.n_channels,
            batch_size=p.batch_size
        )
        data_iterator = tf.data.Iterator.from_structure(
            test_dataset.output_types,
            test_dataset.output_shapes
        )
        test_dataset_init = data_iterator.make_initializer(test_dataset)
        batch_data = data_iterator.get_next()

    test_filenames, test_xy, test_images = load_full_dataset(p.data_dir, p.test_datasets)

    with tf.Session() as sess:
        im = test_images[0]

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(np.squeeze(im), vmin=0, vmax=1)
        ax[0].axis('off')

        h = ax[1].imshow(np.zeros(p.patch_size, dtype=np.float32), vmin=0, vmax=1)
        ax[1].axis('off')

        rect = Rectangle(
            xy=(0,0), width=p.patch_size[1], height=p.patch_size[0],
            facecolor='none', edgecolor='red', linewidth=1
        )
        ax[0].add_patch(rect)

        plt.ion()
        plt.show()

        test_data_sampler.set_image(im)
        test_data_sampler.set_sampling_method(False, xy=test_xy[0])
        sess.run(test_dataset_init)
        try:
            while True:
                a, x0, y0 = sess.run([*batch_data])
                x0 = x0.astype(np.int32)
                x0 = x0[:, 0, 0, 0]
                y0 = y0.astype(np.int32)
                y0 = y0[:, 0, 0, 0]
                for _a, _x0, _y0 in zip(a, x0, y0):
                    h.set_data(np.squeeze(_a))
                    rect.set_xy((_x0, _y0))
                    plt.draw()
                    plt.pause(1)
        except tf.errors.OutOfRangeError:
            pass

def test_random_sampling():
    p = Param()
    p.test_datasets = '../../../data/test_sem_dataset_208.npy'

    with tf.device('/cpu:0'), tf.name_scope('input'):
        test_dataset, test_data_sampler = test_input_fn(
            patch_size=p.patch_size,
            n_channels=p.n_channels,
            batch_size=p.batch_size
        )
        data_iterator = tf.data.Iterator.from_structure(
            test_dataset.output_types,
            test_dataset.output_shapes
        )
        test_dataset_init = data_iterator.make_initializer(test_dataset)
        batch_data = data_iterator.get_next()

    test_filenames, test_xy, test_images = load_full_dataset(p.data_dir, p.test_datasets)

    with tf.Session() as sess:
        im = test_images[0]

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(np.squeeze(im), vmin=0, vmax=1)
        ax[0].axis('off')

        h = ax[1].imshow(np.zeros(p.patch_size, dtype=np.float32), vmin=0, vmax=1)
        ax[1].axis('off')

        rect = Rectangle(
            xy=(0,0), width=p.patch_size[1], height=p.patch_size[0],
            facecolor='none', edgecolor='red', linewidth=1
        )
        ax[0].add_patch(rect)

        plt.ion()
        plt.show()

        test_data_sampler.set_image(im)
        test_data_sampler.set_sampling_method(False, n_samplings=20)
        sess.run(test_dataset_init)
        try:
            while True:
                a, x0, y0 = sess.run([*batch_data])
                x0 = x0.astype(np.int32)
                x0 = x0[:, 0, 0, 0]
                y0 = y0.astype(np.int32)
                y0 = y0[:, 0, 0, 0]
                for _a, _x0, _y0 in zip(a, x0, y0):
                    h.set_data(np.squeeze(_a))
                    rect.set_xy((_x0, _y0))
                    plt.draw()
                    plt.pause(1)
        except tf.errors.OutOfRangeError:
            pass


if __name__ == '__main__':
    #test_grid_sampling()
    test_xy_sampling()
    #test_random_sampling()