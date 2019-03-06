#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import os
#import matplotlib.pyplot as plt

import pickle
import tensorflow as tf
import numpy as np

from utils.Param import FTFYParam
from network.dataset.ftfy_patchdata import input_fn
from network.model_fn import ftfy_model_fn
from network.train import FTFYEstimator

from utils.eval import calc_iou_k, ftfy_retrieval_accuracy


# set seed for reproduction
np.random.seed(2019)
tf.set_random_seed(2019)


# parameters (adjust as needed)
log_dir = './log/sem_ftfy_full_2'
param = FTFYParam(ftfy_scope='ftfy', feat_scope='triplet-net', log_dir=log_dir)
param.is_ftfy_model = False
param.batch_size = 8 # 32 for v100, 8 for ws
feat_trainable = True
param.optimizer_name = 'Grad'
param.obj_scale = 1.0
param.noobj_scale = 0.5
param.coord_scale = 5.0
param.decay_steps = 20000 # 60000 for v100
param.train_log_every = 1000

param.n_epoch = 100
n_max_tests = 5000 # 5000


is_sem = True
if is_sem:
    #param.data_dir = './Data/sem/train'
    param.data_dir = '/home/sungsooha/Desktop/Data/ftfy/sem/train'
    param.learning_rate = 0.01
    
    n_max_steps = 1000 # 0 for v100, sem
else:
    param.data_dir = './Data/austin'
    param.model_path = './log/campus' # './log/sem'
    param.learning_rate = 0.005
    
    param.src_dir = 'campus_sources'
    param.tar_dir = 'campus_patch'
    param.train_datasets = 'campus'
    param.tar_patches_per_row = 13
    param.tar_patches_per_col = 6
    param.train_log_every = 1000
    
    n_max_steps = 0 # 0 for v100, sem


# In[ ]:
if is_sem:
    # only for sem dataset
    data_dirs = []
    for data_dir in os.listdir(param.data_dir):
        if os.path.isdir(os.path.join(param.data_dir, data_dir)):
            data_dirs.append(data_dir)
    data_dirs = sorted(data_dirs)
else:
    data_dirs = ['.']


# ### data pipeline
# In[ ]:
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


# ### compute data statistics
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


# ### build model
tf.logging.info("Creating the model ...")
sources, targets, labels, bboxes = batch_data
spec = ftfy_model_fn(sources, targets, labels, bboxes,
                     mode='TRAIN', **param.get_model_kwargs(feat_trainable=feat_trainable))
# 20-th epoch, logged with 5 interval
estimator = FTFYEstimator(spec, **param.get_ckpt_kwargs(20//5))


# ### Training
top_k = [1, 5, 10, 20, 30]
iou_thrs = [0.7, 0.8, 0.9]


all_loss = []
all_accuracy = []
all_d_mean = []
all_d_std = []

tf.logging.info('='*50)
tf.logging.info('Start training ...')
tf.logging.info('='*50)
for epoch in range(param.n_epoch):
    tf.logging.info('-'*50)
    tf.logging.info('TRAIN {:d}, {:s} start ...'.format(epoch, param.train_datasets))
    data_sampler.set_mode(True)
    loss = estimator.train(
        dataset_initializer=dataset_init,
        log_every=param.train_log_every,
        n_max_steps=n_max_steps
    )
    tf.logging.info('-'*50)

    tf.logging.info('-' * 50)
    tf.logging.info('TEST {:d}, {:s} start ...'.format(epoch, param.train_datasets))
    data_sampler.set_mode(False)
    pred_confidences, pred_bboxes, bboxes = estimator.run(
        dataset_init,top_k=top_k[-1], n_max_test=n_max_tests)
    iou_k = calc_iou_k(pred_bboxes, bboxes)
    accuracy = ftfy_retrieval_accuracy(iou_k, top_k, iou_thrs)

    pred_bboxes = pred_bboxes[:, 0]
    d_bboxes = np.abs(pred_bboxes - bboxes)
    src_h, src_w = param.src_size
    d_bboxes[..., 0] *= src_w
    d_bboxes[..., 1] *= src_h
    d_bboxes[..., 2] *= src_w
    d_bboxes[..., 3] *= src_h
    d_cx_mean, d_cx_std = np.mean(d_bboxes[..., 0]), np.std(d_bboxes[..., 0])
    d_cy_mean, d_cy_std = np.mean(d_bboxes[..., 1]), np.std(d_bboxes[..., 1])
    d_w_mean, d_w_std = np.mean(d_bboxes[..., 2]), np.std(d_bboxes[..., 2])
    d_h_mean, d_h_std = np.mean(d_bboxes[..., 3]), np.std(d_bboxes[..., 3])


    all_loss.append(loss)
    all_accuracy.append(accuracy)
    all_d_mean.append(np.array([d_cx_mean, d_cy_mean, d_w_mean, d_h_mean]))
    all_d_std.append(np.array([d_cx_std, d_cy_std, d_w_std, d_h_std]))

    tf.logging.info('Avg. Retrieval Accuracy:\n {}'.format(accuracy))
    tf.logging.info('For the best (@k=1), [mean, std]')
    tf.logging.info('d_cx: {:.3f}, {:.3f}'.format(d_cx_mean, d_cx_std))
    tf.logging.info('d_cy: {:.3f}, {:.3f}'.format(d_cy_mean, d_cy_std))
    tf.logging.info('d_w : {:.3f}, {:.3f}'.format(d_w_mean, d_w_std))
    tf.logging.info('d_h : {:.3f}, {:.3f}'.format(d_h_mean, d_h_std))
    tf.logging.info('-' * 50)

    # save checkpoint
    if epoch % param.save_every == 0 or epoch+1 == param.n_epoch:
        estimator.save(param.ftfy_scope, global_step=epoch)

        out_dir = os.path.join(param.log_dir, 'metrics_{}_{}.npy'.format(
            param.train_datasets, param.train_datasets
        ))
        metric = dict(
            loss=all_loss,
            accuracy=all_accuracy,
            d_mean=all_d_mean,
            d_std=all_d_std
        )
        np.save(out_dir, metric)




# plt.plot(all_loss)


# In[ ]:


# all_accuracy = np.asarray(all_accuracy)
# all_accuracy = np.squeeze(all_accuracy)
# plt.plot(all_accuracy)


# In[ ]:


# all_d_cx_mean = np.squeeze(np.asarray(all_d_cx_mean))
# all_d_cy_mean = np.squeeze(np.asarray(all_d_cy_mean))
# all_d_w_mean = np.squeeze(np.asarray(all_d_w_mean))
# all_d_h_mean = np.squeeze(np.asarray(all_d_h_mean))

# all_d_cx_std = np.squeeze(np.asarray(all_d_cx_std))
# all_d_cy_std = np.squeeze(np.asarray(all_d_cy_std))
# all_d_w_std = np.squeeze(np.asarray(all_d_w_std))
# all_d_h_std = np.squeeze(np.asarray(all_d_h_std))


# In[ ]:


# fig, ax = plt.subplots(2, 2)
# ax = ax.ravel()
#
# N = len(all_d_cx_mean)
# ax[0].errorbar(range(N), all_d_cx_mean, all_d_cx_std, linestyle='None', marker='^')
# ax[1].errorbar(range(N), all_d_cy_mean, all_d_cy_std, linestyle='None', marker='^')
# ax[2].errorbar(range(N), all_d_w_mean, all_d_w_std, linestyle='None', marker='^')
# ax[3].errorbar(range(N), all_d_h_mean, all_d_h_std, linestyle='None', marker='^')


# In[ ]:


# save results
# out_dir = os.path.join(param.log_dir, 'metrics_{}_{}.npy'.format(
#     param.train_datasets, param.train_datasets
# ))
# metric = dict(
#     loss=np.array(all_loss),
#     accuracy=all_accuracy,
#     d_cx_mean=all_d_cx_mean,
#     d_cx_std=all_d_cx_std,
#     d_cy_mean=all_d_cy_mean,
#     d_cy_std=all_d_cy_std,
#     d_w_mean=all_d_w_mean,
#     d_w_std=all_d_w_std,
#     d_h_mean=all_d_h_mean,
#     d_h_std=all_d_h_std
# )
# np.save(out_dir, metric)


# In[ ]:





# In[ ]:




