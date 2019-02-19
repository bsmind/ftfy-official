import os

import tensorflow as tf
import numpy as np
import pickle

from utils.Param import get_default_param
from utils.eval import fpr, retrieval_recall_K

from network.model_fn import triplet_model_fn
from network.dataset.sem_patchdata import input_fn
from network.dataset.sem_patchdata_ext import input_fn as sem_input_fn
from network.train import TripletEstimator

from skimage.io import imsave

tf.logging.set_verbosity(tf.logging.INFO)
np.random.seed(2019)
tf.set_random_seed(2019)

param = get_default_param(mode='AUSTIN', log_dir=None)
param.log_dir = './log/human'
param.data_dir = '/home/sungsooha/Desktop/Data/ftfy/austin'

#sem_data_dir = './Data/sem/train'
sem_data_dir = '/home/sungsooha/Desktop/Data/ftfy/sem/train'
sem_test_datasets = []
for f in os.listdir(sem_data_dir):
    if os.path.isdir(os.path.join(sem_data_dir,f)):
        sem_test_datasets.append(f)
sem_test_datasets = sorted(sem_test_datasets)
print(sem_test_datasets)

param.train_datasets = 'human_patch'
param.test_datasets = 'sem' #'human_patch'
param.batch_size = 128
param.model_path = './log/human/ckpt'

do_test_fpr = True
do_test_retrieval = True
do_collect_retrieval_5 = True

print('Preparing data pipeline ...')
with tf.device('/cpu:0'), tf.name_scope('input'):
    # test_dataset, test_data_sampler = input_fn(
    #     data_dir=param.data_dir,
    #     base_patch_size=param.base_patch_size,
    #     patches_per_row=param.patches_per_row,
    #     patches_per_col=param.patches_per_col,
    #     batch_size=param.batch_size,
    #     patch_size=param.patch_size,
    #     n_channels=param.n_channels
    # )
    test_dataset, test_data_sampler = sem_input_fn(
        data_dir=sem_data_dir,
        base_patch_size=param.base_patch_size,
        patches_per_row=10,
        patches_per_col=10,
        batch_size=param.batch_size,
        patch_size=param.patch_size,
        n_channels=param.n_channels
    )
    data_iterator = tf.data.Iterator.from_structure(
        test_dataset.output_types,
        test_dataset.output_shapes
    )
    test_dataset_init = data_iterator.make_initializer(test_dataset)
    batch_data = data_iterator.get_next()

print('load data ...')
test_data_sampler.load_dataset(
    dir_name=sem_test_datasets,
    ext='bmp',
    patch_size=param.patch_size,
    n_channels=param.n_channels,
    debug=True
)

print('Loading training stats: %s' % param.train_datasets)
file = open(os.path.join(param.log_dir, 'stats_%s.pkl' % param.train_datasets), 'rb')
mean, std = pickle.load(file)
print('Mean: {:.5f}'.format(mean))
print('Std : {:.5f}'.format(std))
test_data_sampler.normalize_data(mean, std)

print('Creating the model ...')
anchors, positives, negatives = batch_data
spec = triplet_model_fn(
    anchors, positives, negatives,
    n_feats=param.n_features,
    mode='TEST', cnn_name=param.cnn_name, shared_batch_layers=True,
    name=param.project_name
)
estimator = TripletEstimator(spec, model_path=param.model_path)

if do_test_fpr:
    print('Test for FPR95 ...')
    test_data_sampler.set_mode(1)
    test_match = estimator.run_match(test_dataset_init)
    fpr95 = fpr(test_match.labels, test_match.scores, recall_rate=0.95)
    print('FPR95: {:.5f}'.format(fpr95))

if do_test_retrieval or do_collect_retrieval_5:
    print('Test retrieval ...')
    test_data_sampler.set_mode(2)
    test_rrr = estimator.run_retrieval(test_dataset_init)

    ind = test_rrr.index
    labels = test_data_sampler.get_labels(ind)
    rrr, rrr_col = retrieval_recall_K(
        features=test_rrr.features,
        labels=labels,
        is_query=test_rrr.scores,
        K=[1, 5, 10, 20, 30],
        collect_top_5=do_collect_retrieval_5
    )

    print('Retrieval: {}'.format(rrr))

    if do_collect_retrieval_5:
        failed_ind,  = np.where(rrr_col[:, 6]==0)

        count = 0
        output_dir = os.path.join(param.log_dir, param.test_datasets)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        file = open(os.path.join(output_dir, 'retrieval_fail.txt'), 'w')
        for idx in failed_ind:
            q_idx = ind[rrr_col[idx, 0]]
            top5_ind = ind[rrr_col[idx, 1:6]]

            patch_set = []
            #patch_info = []

            patch, patch_idx = test_data_sampler.get_patch_by_retrieval_idx(q_idx)
            patch = np.squeeze(patch)
            patch = patch * std + mean
            patch = (patch - patch.min()) / patch.ptp()
            patch_set.append(patch)

            # todo: this doesn't work for sem patch dataset
            patch_gid = patch_idx // 78
            irow = (patch_idx%78) // 6
            icol = (patch_idx%78) % 6
            #patch_info.append((patch_gid, irow, icol))
            file.write("{:d} {:d} {:d} {:d}".format(
                count, patch_gid, irow, icol
            ))

            for i in top5_ind:
                patch, patch_idx = test_data_sampler.get_patch_by_retrieval_idx(i)
                patch = np.squeeze(patch)
                patch = patch * std + mean
                patch = (patch - patch.min()) / patch.ptp()
                patch_set.append(patch)

                # todo: this doesn't work for sem patch dataset
                patch_gid = patch_idx // 78
                irow = (patch_idx % 78) // 6
                icol = (patch_idx % 78) % 6
                #patch_info.append((patch_gid, irow, icol))
                file.write(" {:d} {:d} {:d} {:d}".format(
                    i, patch_gid, irow, icol
                ))
            file.write("\n")
            patch_set = np.hstack(patch_set)
            patch_set = patch_set * 255.
            patch_set = patch_set.astype(np.uint8)

            imsave(os.path.join(output_dir, 'fail_{:d}.bmp'.format(count)), patch_set)

            count += 1
        file.close()
            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots(1, 6)
            # for _ax, p, info in zip(ax, patches, patch_info):
            #     _ax.imshow(np.squeeze(p), cmap='gray')
            #     _ax.axis('off')
            #     _ax.set_title("{}".format(info))
            # plt.show()

            #q_label = labels[q_idx]
            #top5_labels = labels[top5_ind]
            # print('Query: {}, top 5: {}'.format(
            #     q_label, top5_labels
            # ))



        #retrieval_data = test_data_sampler.data['retrieval']


# out_dir = os.path.join(param.log_dir, 'metrics_{}_{}.npy'.format(
#     param.train_datasets, param.test_datasets
# ))
# metric = dict(
#     loss=None,
#     fpr95=fpr95,
#     retrieval=rrr
# )
# np.save(out_dir, metric)




















