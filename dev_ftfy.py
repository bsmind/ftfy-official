import os
import pickle
import tensorflow as tf
import numpy as np

from utils.Param import FTFYParam
from network.dataset.ftfy_patchdata import input_fn
from network.model_fn import ftfy_model_fn
from network.train import FTFYEstimator


param = FTFYParam(ftfy_scope='ftfy', feat_scope='triplet-net', log_dir='./log/sem_ftfy')
param.is_ftfy_model = False
param.model_path = './log/sem/ckpt'
param.train_log_every = 100
param.batch_size = 4
feat_trainable = False
param.optimizer_name = 'Grad'
param.learning_rate = 0.1
param.obj_scale = 1.0
param.noobj_scale = 1.0
param.coord_scale = 1.0

# only for sem dataset
data_dirs = []
for data_dir in os.listdir(param.data_dir):
    if os.path.isdir(os.path.join(param.data_dir, data_dir)):
        data_dirs.append(data_dir)
data_dirs = sorted(data_dirs)

# ----------------------------------------------------------------------------
# Data pipeline
# ----------------------------------------------------------------------------
tf.logging.info("Preparing data pipeline ...")
with tf.device('/cpu:0'), tf.name_scope('input'):
    dataset, data_sampler = input_fn(
        param.data_dir,
        batch_size=param.batch_size,
        cellsz=param.src_cellsz,
        n_parameters=param.n_parameters,
        src_size=param.src_size,
        tar_size=param.tar_size,
        n_channels=param.n_channels
    )
    data_iterator = tf.data.Iterator.from_structure(
        dataset.output_types,
        dataset.output_shapes
    )
    dataset_init = data_iterator.make_initializer(dataset)
    batch_data = data_iterator.get_next()

data_sampler.load_dataset(
    data_dirs, param.src_dir, param.tar_dir,
    src_ext=param.src_ext, src_size=param.src_size, n_src_channels=param.n_channels,
    src_per_col=param.src_patches_per_col, src_per_row=param.src_patches_per_row,
    tar_ext=param.tar_ext, tar_size=param.tar_size, n_tar_channels=param.n_channels,
    tar_per_col=param.tar_patches_per_col, tar_per_row=param.tar_patches_per_row
)

# ----------------------------------------------------------------------------
# Compute data stat
# ----------------------------------------------------------------------------
tf.logging.info('Loading training stats: %s' % param.train_datasets)
try:
    file = open(os.path.join(param.log_dir, 'stats_%s.pkl' % param.train_datasets), 'rb')
    mean, std = pickle.load(file)
except FileNotFoundError:
    tf.logging.info("Calculating train data stats (mean, std)")
    mean, std = data_sampler.generate_stats()
    pickle.dump(
        [mean, std],
        open(os.path.join(param.log_dir, 'stats_%s.pkl' % param.train_datasets), 'wb')
    )
tf.logging.info('Mean: {:.5f}'.format(mean))
tf.logging.info('Std : {:.5f}'.format(std))
data_sampler.normalize_data(mean, std)

# ----------------------------------------------------------------------------
# Build network
# ----------------------------------------------------------------------------
tf.logging.info("Creating the model ...")
sources, targets, labels, bboxes = batch_data
spec = ftfy_model_fn(sources, targets, labels, bboxes,
                     mode='TRAIN', **param.get_model_kwargs(feat_trainable=feat_trainable))
# 20-th epoch, logged with 5 interval
estimator = FTFYEstimator(spec, **param.get_ckpt_kwargs(20//5))

# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
tf.logging.info('='*50)
tf.logging.info('Start training ...')
tf.logging.info('='*50)
for epoch in range(param.n_epoch):
    tf.logging.info('-'*50)
    tf.logging.info('TRAIN {:d}, {:s} start ...'.format(epoch, param.train_datasets))
    loss = estimator.train(
        dataset_initializer=dataset_init,
        log_every=param.train_log_every
    )
    tf.logging.info('-'*50)

    tf.logging.info('-' * 50)
    tf.logging.info('TEST {:d}, {:s} start ...'.format(epoch, param.train_datasets))
    logits, bboxes = estimator.run(dataset_init)
    tf.logging.info('-' * 50)
    break


# ph_src = tf.placeholder(tf.float32, (None, 256, 256, 1), "source")
# ph_tar = tf.placeholder(tf.float32, (None, 128, 128, 1), "target")
# ph_labels = tf.placeholder(tf.float32, (None, 16, 16, 5), "labels")
# batch_size = 16
#
# spec = ftfy_model_fn(
#     sources=ph_src, targets=ph_tar, labels=ph_labels, bboxes=None,
#     feat_name='ftfy', feat_trainable=False, feat_shared_batch_layers=True, feat_scope='triplet-net',
#     ftfy_ver='v0', ftfy_scope='ftfy', cell_size=8, n_bbox_estimators=2, n_parameters=5,
#     loss_name='rms', obj_scale=1.0, noobj_scale=0.5, coord_scale=5.0,
#     use_regularization_loss=False, optimizer_name='Momentum', optimizer_kwargs=None,
#     mode='TRAIN'
# )
#
# for v in tf.trainable_variables():
#     print(v)
#
# # initialize
# config = tf.ConfigProto(
#     allow_soft_placement=True,
#     log_device_placement=False,
#     intra_op_parallelism_threads=8,
#     inter_op_parallelism_threads=0
# )
# config.gpu_options.allow_growth = True
#
# vars_triplet = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='triplet-net')
# saver = tf.train.Saver(vars_triplet)
#
# with tf.Session(config=config) as sess:
#     sess.run(tf.global_variables_initializer())
#     sess.run(tf.local_variables_initializer())
#
#     ckpt = tf.train.get_checkpoint_state('./log/sem/ckpt')
#     saver.restore(sess, ckpt.model_checkpoint_path)
#
#     test_src = np.random.random((batch_size, 256, 256, 1)).astype(np.float32)
#     test_tar = np.random.random((batch_size, 128, 128, 1)).astype(np.float32)
#     test_labels = np.random.random((batch_size, 16, 16, 5)).astype(np.float32)
#
#     #feed_dict = dict(triplet_spec.test_feed_dict)
#     feed_dict = dict(spec.test_feed_dict)
#     feed_dict[ph_src] = test_src
#     feed_dict[ph_tar] = test_tar
#     feed_dict[ph_labels] = test_labels
#     #feed_dict[ph_is_training] = False
#
#     _, src_feat, tar_feat, logits_output, obj_loss_output, noobj_loss_output, coord_loss_output = \
#         sess.run(
#         [
#             spec.train_op,
#             spec.src_feat, spec.tar_feat, spec.logits,
#             spec.obj_loss, spec.noobj_loss, spec.coord_loss
#         ],
#         feed_dict=feed_dict
#     )
#
#     print(src_feat.shape)
#     print(tar_feat.shape)
#     print(logits_output.shape)
#     print(obj_loss_output, noobj_loss_output, coord_loss_output)