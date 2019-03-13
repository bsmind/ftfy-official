"""
Training triplet network with natural image set
This is slightly modifed from v100_train_triplet_network_natural.py

Author(s):
    Sungsoo Ha (sungsooha@bnl.gov)

Last Modified:
    March 13, 2019
"""
import os

import tensorflow as tf
import numpy as np
import pickle

from utils.Param import get_default_param
from utils.eval import fpr, retrieval_recall_K

from network.model_fn import triplet_model_fn
from network.dataset.sem_patchdata_ext import input_fn
from network.train import TripletEstimator

# In[ ]:
# set seed for reproduction
np.random.seed(2019)
tf.set_random_seed(2019)

# In[ ]:
# parameters (adjust as needed)
log_dir = './log/campus2'
param = get_default_param(mode='AUSTIN', log_dir=log_dir)

# In[ ]:
data_dir = '/home/sungsooha/Desktop/Data/ftfy/austin'
train_datasets = 'campus_patch'

# In[ ]:
param.data_dir = data_dir
param.train_datasets = 'campus_patch'  # we will define sem dataset separately
param.test_datasets = None  # 'human_patch'
param.batch_size = 64  # 64 for v100
param.n_epoch = 30 #100
param.n_triplet_samples = 500000
param.train_log_every = 5000

# ### data pipeline
# In[ ]:
tf.logging.info("Preparing data pipeline ...")
with tf.device('/cpu:0'), tf.name_scope('input'):
    dataset, data_sampler = input_fn(
        data_dir=data_dir,
        train_fnames=dict(triplet='triplet_' + train_datasets + '.txt',
                          matched='train_matched_' + train_datasets + '.txt',
                          retrieval='train_retrieval_' + train_datasets + '.txt'),
        test_fnames=dict(matched='test_matched_' + train_datasets + '.txt',
                         retrieval='test_retrieval_' + train_datasets + '.txt'),
        base_patch_size=param.base_patch_size,
        patches_per_row=13,
        patches_per_col=6,
        batch_size=param.batch_size,
        patch_size=param.patch_size,
        n_channels=param.n_channels
    )
    data_iterator = tf.data.Iterator.from_structure(
        dataset.output_types,
        dataset.output_shapes
    )
    dataset_init = data_iterator.make_initializer(dataset)
    batch_data = data_iterator.get_next()

# ### load data
# In[ ]:
data_sampler.load_dataset(
    dir_name=[train_datasets],
    ext='bmp',
    patch_size=param.patch_size,
    n_channels=param.n_channels,
    patch_dir='.',
    debug=True
)


# ### compute data statistics
tf.logging.info('Loading training stats: %s' % param.train_datasets)
try:
    file = open(os.path.join(param.log_dir, 'stats_%s.pkl' % param.train_datasets), 'rb')
    mean, std = pickle.load(file)
except:
    tf.logging.info('Calculating train data stats (mean, std)')
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
anchors, positives, negatives = batch_data
spec = triplet_model_fn(
    anchors, positives, negatives, n_feats=param.n_features,
    mode='TRAIN', cnn_name=param.cnn_name, loss_name=param.loss_name,
    optimizer_name=param.optimizer_name,
    margin=param.margin,
    use_regularization_loss=param.use_regularization,
    learning_rate=param.learning_rate,
    shared_batch_layers=True,
    name='triplet-net'
)
estimator = TripletEstimator(spec, save_dir=param.log_dir)

# ### Training
K = [1, 5, 10, 20, 30]

all_loss = []  # avg. loss over epochs
train_fpr95 = []  # fpr95 with training dataset
train_retrieval = []  # retrieval with training dataset
test_fpr95 = []
test_retrieval = []

tf.logging.info('=' * 50)
tf.logging.info('Start training ...')
tf.logging.info('=' * 50)
for epoch in range(param.n_epoch):
    tf.logging.info('-' * 50)
    tf.logging.info('TRAIN {:d}, {:s} start ...'.format(epoch, param.train_datasets))
    data_sampler.set_mode(0)
    data_sampler.set_n_triplet_samples(param.n_triplet_samples)
    #data_sampler.set_n_triplet_samples(5000)
    loss = estimator.train(
        dataset_initializer=dataset_init,
        log_every=param.train_log_every
    )
    all_loss.append(loss)
    tf.logging.info('-' * 50)

    # for evaluation with training dataset
    tf.logging.info('-' * 50)
    tf.logging.info('TEST {:d}, {:s} TRAIN start ...'.format(epoch, param.train_datasets))
    data_sampler.set_eval_mode(True)
    data_sampler.set_mode(1)
    #data_sampler.set_n_matched_pairs(5000)
    test_match = estimator.run_match(dataset_init)
    fpr95 = fpr(test_match.labels, test_match.scores, recall_rate=0.95)
    train_fpr95.append(fpr95)
    tf.logging.info('FPR95: {:.5f}'.format(fpr95))

    data_sampler.set_mode(2)
    test_rrr = estimator.run_retrieval(dataset_init)
    rrr = retrieval_recall_K(
        features=test_rrr.features,
        labels=data_sampler.get_labels(test_rrr.index),
        is_query=test_rrr.scores,
        K=K
    )[0]
    train_retrieval.append(rrr)
    tf.logging.info('Retrieval: {}'.format(rrr))
    tf.logging.info('-' * 50)

    # for evaluation with test dataset
    tf.logging.info('-' * 50)
    tf.logging.info('TEST {:d}, {:s} TEST start ...'.format(epoch, param.train_datasets))
    data_sampler.set_eval_mode(False)
    data_sampler.set_mode(1)
    #data_sampler.set_n_matched_pairs(5000)
    test_match = estimator.run_match(dataset_init)
    fpr95 = fpr(test_match.labels, test_match.scores, recall_rate=0.95)
    test_fpr95.append(fpr95)
    tf.logging.info('FPR95: {:.5f}'.format(fpr95))

    data_sampler.set_mode(2)
    test_rrr = estimator.run_retrieval(dataset_init)
    rrr = retrieval_recall_K(
        features=test_rrr.features,
        labels=data_sampler.get_labels(test_rrr.index),
        is_query=test_rrr.scores,
        K=K
    )[0]
    test_retrieval.append(rrr)
    tf.logging.info('Retrieval: {}'.format(rrr))
    tf.logging.info('-' * 50)

    # save checkpoint
    if epoch % param.save_every == 0 or epoch + 1 == param.n_epoch:
        estimator.save(param.project_name, global_step=epoch)

# save results
out_dir = os.path.join(param.log_dir, 'metrics_train_{:s}.npy'.format(train_datasets))
metric = dict(
    loss=np.array(all_loss),
    fpr95=np.array(train_fpr95),
    retrieval=np.asarray(train_retrieval)
)
np.save(out_dir, metric)

out_dir = os.path.join(param.log_dir, 'metrics_test_{:s}.npy'.format(train_datasets))
metric = dict(
    loss=np.array(all_loss),
    fpr95=np.array(test_fpr95),
    retrieval=np.asarray(test_retrieval)
)
np.save(out_dir, metric)
