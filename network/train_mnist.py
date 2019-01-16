import os
# set tensorflow cpp log level. It is useful to diable some annoying log message,
# but sometime may miss some useful information
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import importlib
import numpy as np
import tensorflow as tf

from network.dataset.data_sampler import TripletDataSampler
from network.dataset.mnist_input_fn import get_mnist_train_data, get_mnist_test_data
from network.dataset.mnist_input_fn import input_fn
from network.model.simple_cnn import Net as model
from network.loss.triplet import batch_offline_triplet_loss, batch_offline_lossless_triplet_loss

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class LRManager:
    def __init__(self, boundaries, values):
        self.boundaries = boundaries
        self.values = values

    def get(self, epoch):
        for b, v in zip(self.boundaries, self.values):
            if epoch < b:
                return v
        return self.values[-1]

# set seed
np.random.seed(0)
tf.set_random_seed(0)

# parameters
data_dir = "../data"
n_class = 10
n_class_per_iter = n_class
n_img_per_class = 10 * n_class
n_iter_per_epoch = 20
batch_size = n_class_per_iter * n_img_per_class
is_online = False # online or offline mining

n_epoch = 500
init_learning_rate = 0.0001
log_every = 50
n_iter_for_embed = 1
tsne_dim = 2

n_feats = 32
margin = 0.3
use_lossless = False
use_custom_lr = False

# if tsne_dim == 3:
#     from mpl_toolkits.mplot3d import Axes3D


# dataset pipeline
dataset_wargs = {
    'n_class_per_iter': n_class_per_iter,
    'n_img_per_class': n_img_per_class,
    'n_iter_per_epoch': n_iter_per_epoch,
    'batch_size': batch_size,
    'is_online': is_online
}
with tf.device('/cpu:0'), tf.name_scope('input'):
    train_dataset = input_fn(data_dir, True, dataset_wargs)
    test_dataset = input_fn(data_dir, False, dataset_wargs)

    data_iterator = tf.data.Iterator.from_structure(
        train_dataset.output_types,
        train_dataset.output_shapes
    )
    train_dataset_init = data_iterator.make_initializer(train_dataset)
    test_dataset_init = data_iterator.make_initializer(test_dataset)
    batch_data = data_iterator.get_next()


# placeholder & variables
is_training = tf.placeholder(tf.bool, (), "is_training")
learning_rate = tf.placeholder(tf.float32, (), "learning_rate")
global_step = tf.train.create_global_step()

# build network and get embeddings
batch_anchors, batch_positives, batch_negatives, batch_labels = batch_data
net = model(n_feats=n_feats, weight_decay=0.0001)
a_embed = net(batch_anchors, is_training, bn_prefix="a")
p_embed = net(batch_positives, is_training, bn_prefix="p")
n_embed = net(batch_negatives, is_training, bn_prefix="n")

# loss
with tf.name_scope('losses'):
    if use_lossless:
        loss = batch_offline_lossless_triplet_loss(a_embed, p_embed, n_embed, n_feats, n_feats)
    else:
        loss = batch_offline_triplet_loss(a_embed, p_embed, n_embed, margin)
    l2_reg = tf.losses.get_regularization_loss()

# optimizer & train operation
#optimizer = tf.train.AdamOptimizer(init_learning_rate)
if use_custom_lr:
    optimizer = tf.train.AdamOptimizer(learning_rate)
else:
    optimizer = tf.train.AdamOptimizer(init_learning_rate)
grads_and_vars = optimizer.compute_gradients(loss + l2_reg)

train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
train_op = tf.group(train_op, *update_ops)

#lr_manager = LRManager([50, 100, 200], [1e-3, 1e-4, 1e-5, 1e-6])
lr_manager = LRManager([100, 200, 300], [1e-3, 1e-4, 1e-5, 1e-6])

# training loop
config = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False,
    intra_op_parallelism_threads=8,
    inter_op_parallelism_threads=0
)
config.gpu_options.allow_growth = True

fig = plt.figure()
plt.ion()

with tf.Session(config=config) as sess:
    # initialize global/local variables
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    for epoch in range(n_epoch):
        print('-' * 40)
        print('Epoch: {:d}'.format(epoch+1))

        # training step
        lr = lr_manager.get(epoch)
        step = 0
        avg_loss = 0
        try:
            # initialize dataset
            sess.run(train_dataset_init)
            while True:
                outputs = sess.run(
                    [train_op, loss, l2_reg],
                    feed_dict={
                        learning_rate: lr,
                        is_training: True,
                    }
                )
                step += 1
                avg_loss += outputs[1]
                if step % log_every == 0:
                    print("step {:d}: loss: {:.6f}, l2_reg: {:.6f}".format(
                        step, outputs[1], outputs[2]
                    ))
        except tf.errors.OutOfRangeError:
            pass

        avg_loss /= step
        print("avg loss: ", avg_loss)

        # todo: save checkpoint

        if epoch % 10:
            continue



        # validation step
        step = 0
        val_embeds = []
        val_labels = []
        try:
            # initialize dataset
            sess.run(test_dataset_init)
            while True:
                outputs = sess.run(
                    [a_embed, batch_labels],
                    feed_dict={is_training: False}
                )
                if step < n_iter_for_embed:
                    val_embeds.append(outputs[0])
                    val_labels.append(outputs[1])
                step += 1

        except tf.errors.OutOfRangeError:
            pass

        val_embeds = np.concatenate(val_embeds, axis=0)
        val_labels = np.concatenate(val_labels, axis=0)

        tsne_model = TSNE(tsne_dim)
        val_embeds = tsne_model.fit_transform(val_embeds)

        fig.clear()
        for l in range(10):
            c_feats = val_embeds[val_labels == l]
            plt.scatter(c_feats[:, 0], c_feats[:, 1], label=l)
        plt.legend()

        plt.draw()
        plt.pause(0.001)
        if epoch == 0:
            plt.show()




# # MNIST meta data
# n_class = 10
# data_dir = "../data"
#
# params = {
#     'n_class': n_class,
#     'img_shape': (28, 28, 1),
#
#     'is_online': False,
#     'n_class_per_iter': 10,
#     'n_img_per_class': 2,
#     'batch_size': 20,
#     'n_iter_per_epoch': 500,
# }
#
# #with tf.device('/cpu:0'), tf.name_scope('input'):
# train_dataset = input_fn(data_dir, True, params)
#
# data_iterator = tf.data.Iterator.from_structure(
#     train_dataset.output_types,
#     train_dataset.output_shapes
# )
# train_input_init_op = data_iterator.make_initializer(train_dataset)
# train_batch_data = data_iterator.get_next()
#     # # get data from data iterator
#     # batch_anchors, batch_positives, batch_negatives = data_iterator.get_next()
#     # tf.summary.image('anchors', batch_anchors)
#     # tf.summary.image('positives', batch_positives)
#     # tf.summary.image('negatives', batch_negatives)
# # batch_anchors = tf.placeholder(tf.float32, (None, 28, 28, 1))
# # batch_positives = tf.placeholder(tf.float32, (None, 28, 28, 1))
# # batch_negatives = tf.placeholder(tf.float32, (None, 28, 28, 1))
#
# # define useful scalars
# # learning_rate = tf.placeholder(tf.float32, shape=(), name='learning_rate')
# init_learning_rate = 0.001
# is_training = tf.placeholder(tf.bool, [], name='is_training')
# global_step = tf.train.create_global_step()
# # tf.summary.scalar('lr', learning_rate)
#
#
#
# # get features
# batch_anchors, batch_positives, batch_negatives = train_batch_data
# # with tf.device('/GPU:0'):
# # define optimizer
# optimizer = tf.train.AdamOptimizer(init_learning_rate)
# # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
#
# # build the net
# # model = importlib.import_module('models.{}'.format(FLAGS.model))
# net = model(n_feats=2, weight_decay=0.0001)
#
# a_embed = net(batch_anchors, is_training, bn_prefix="a")
# p_embed = net(batch_positives, is_training, bn_prefix="p")
# n_embed = net(batch_negatives, is_training, bn_prefix="n")
# # tf.summary.histogram('a-embed', a_embed)
# # tf.summary.histogram('p-embed', p_embed)
# # tf.summary.histogram('n-embed', n_embed)
#
# # summary variable defined in net
# # for var in tf.trainable_variables():
# #     print(var.name, var.shape)
# #    tf.summary.histogram(var.name, var)
#
# with tf.name_scope('losses'):
#     loss = batch_offline_triplet_loss(batch_anchors, batch_positives, batch_negatives, 1.0)
#     l2_reg = tf.losses.get_regularization_loss()
#
# # with tf.name_scope('metrics') as scope:
# #     pass
#
# # compute grad
# grads_and_vars = optimizer.compute_gradients(loss + l2_reg)
#
# # summary grads
# # for g, v in grads_and_vars:
# #     tf.summary.histogram(v.name + '/grad', g)
#
# # run train_op and update_op together
# train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
# update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
# train_op = tf.group(train_op, *update_ops)
#
# # build summary
# # ...
#
# # init op
# init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
#
# # prepare for the logdir
# # if not tf.gfile.Exists(logdir):
# #     tf.gfile.MakeDirs(logdir)
#
# # saver
# #saver = tf.train.Saver(max_to_keep=n_epoch)
#
# # summary writer
# # train_writer = tf.summary.FileWriter(
# #     os.path.join(logdir, 'train'),
# #     tf.get_default_graph()
# # )
#
# # session
# config = tf.ConfigProto(
#     allow_soft_placement=True,
#     log_device_placement=False,
#     intra_op_parallelism_threads=8,
#     inter_op_parallelism_threads=0
# )
# config.gpu_options.allow_growth = True
#
# with tf.Session(config=config) as sess:
#     sess.run(init_op)
#
# #     # restore
# #     # if restore:
# #     #     saver.restore(sess, restore)
# #
#     # start train
#     n_epoch = 100
#     log_every = 100
#     for e in range(n_epoch):
#         print('-' * 40)
#         print('Epoch: {:d}'.format(e+1))
#
#         # training loop
#         i = 0
#         avg_loss = 0
#         avg_l2_reg = 0
#         try:
#             sess.run(train_input_init_op)
#             while True:
#                 outputs = sess.run(
#                     [train_op, loss, l2_reg],
#                     feed_dict={
#                         # batch_anchors: np.random.random((100, 28, 28, 1)),
#                         # batch_positives: np.random.random((100, 28, 28, 1)),
#                         # batch_negatives: np.random.random((100, 28, 28, 1)),
#                         is_training: True
#                     }
#                 )
#
#                 avg_loss += outputs[1]
#                 avg_l2_reg += outputs[2]
#                 i += 1
#
#                 if i % log_every == 0:
#                     print("@{:d}: loss: {:.3f}, l2_reg: {:.3f}".format(i, outputs[1], outputs[2]))
#
#         except tf.errors.OutOfRangeError:
#             pass
#
#         print("Avg. loss: {:.3f}, Avg. L2 reg: {:.3f}".format(
#             avg_loss / i, avg_l2_reg / i
#         ))
# #
# #
# #         # save checkpoint
# #         # ...
# #
# #         # val loop
# #         # ...
# #
#         print('-' * 40)
