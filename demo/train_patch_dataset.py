import tensorflow as tf
import numpy as np

from utils.Param import get_default_param
from utils.viz import ScalarPlot
from utils.eval import fpr, retrieval_recall_K
from network.model_fn import triplet_model_fn
from network.dataset.patchdata import input_fn
from network.train import TripletEstimator



# set seed for reproduction
np.random.seed(2019)
tf.set_random_seed(2019)

# parameters
param = get_default_param(mode='UBC', log_dir='../log/ubc_test')

# data pipeline
tf.logging.info("Preparing data pipeline ...")
with tf.device('/cpu:0'), tf.name_scope('input'):
    train_dataset, train_data_sampler = input_fn(
        data_dir=param.data_dir,
        base_patch_size=param.base_patch_size,
        patches_per_row=param.patches_per_row,
        patches_per_col=param.patches_per_col,
        batch_size=param.batch_size,
        patch_size=param.patch_size,
        n_channels=param.n_channels
    )
    test_dataset, test_data_sampler = input_fn(
        data_dir=param.data_dir,
        base_patch_size=param.base_patch_size,
        patches_per_row=param.patches_per_row,
        patches_per_col=param.patches_per_col,
        batch_size=param.batch_size,
        patch_size=param.patch_size,
        n_channels=param.n_channels
    )
    data_iterator = tf.data.Iterator.from_structure(
        train_dataset.output_types,
        train_dataset.output_shapes
    )
    train_dataset_init = data_iterator.make_initializer(train_dataset)
    test_dataset_init = data_iterator.make_initializer(test_dataset)
    batch_data = data_iterator.get_next()

train_data_sampler.load_dataset(
    dir_name=param.train_datasets,
    ext='bmp',
    patch_size=param.patch_size,
    n_channels=param.n_channels,
    debug=True
)
test_data_sampler.load_dataset(
    dir_name=param.test_datasets,
    ext='bmp',
    patch_size=param.patch_size,
    n_channels=param.n_channels,
    debug=True
)
test_data_sampler.set_mode(False)

#todo: compute mean, std and normalize train (test) dataset
#train_data_sampler.generate_triplet(param.n_triplet_samples)
mean, std = train_data_sampler.generate_stats()
print('-- Mean: {:.3f}'.format(mean))
print('-- Std : {:.3f}'.format(std))
train_data_sampler.normalize_data(mean, std)
test_data_sampler.normalize_data(mean, std)

train_data_sampler.generate_triplet(param.n_triplet_samples)
#train_data_sampler.generate_triplet(50000)

# build model
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

# visualization
train_scalar_plot = ScalarPlot(3, [1, 1, 5],
                         legends=[['triplet-loss'], ['fpr95'], ['@1', '@5', '@10', '@100', '@1000']])

test_scalar_plot = ScalarPlot(2, [1, 5],
                         legends=[['fpr95'], ['@1', '@5', '@10', '@100', '@1000']])

# start training
tf.logging.info('='*80)
tf.logging.info('Start training ...')
tf.logging.info('='*80)
for epoch in range(param.n_epoch):
    tf.logging.info('-'*80)
    tf.logging.info('TRAIN {:d}, {:s} start ...'.format(epoch, param.train_datasets))
    train_data_sampler.set_mode(True)
    # if epoch%10 == 0:
    #     train_data_sampler.generate_triplet(10000)
    loss = estimator.train(
        dataset_initializer=train_dataset_init,
        log_every=param.train_log_every
    )
    train_scalar_plot.update(0, [loss])
    tf.logging.info('-'*80)

    # For evaluation.. FPR95.. and Recall@K
    tf.logging.info('-'*80)
    tf.logging.info('TEST {:d}, {:s} start ...'.format(epoch, param.train_datasets))
    train_data_sampler.set_mode(False)
    test_match = estimator.run_match(train_dataset_init)
    fpr95 = fpr(test_match.labels, test_match.scores, recall_rate=0.95)
    rrr = retrieval_recall_K(
        features=test_match.features,
        labels=train_data_sampler.get_labels(test_match.index),
        K=[1, 5, 10, 100, 1000]
    )
    tf.logging.info('FPR95: {:.3f}'.format(fpr95))
    tf.logging.info('Retrieval Recall Rate: {}'.format(rrr))
    train_scalar_plot.update(1, [fpr95])
    train_scalar_plot.update(2, rrr)
    tf.logging.info('-'*80)

    # For evaluation.. FPR95.. and Recall@K
    tf.logging.info('-'*80)
    tf.logging.info('TEST {:d}, {:s} start ...'.format(epoch, param.test_datasets))
    test_match = estimator.run_match(train_dataset_init)
    fpr95 = fpr(test_match.labels, test_match.scores, recall_rate=0.95)
    rrr = retrieval_recall_K(
        features=test_match.features,
        labels=train_data_sampler.get_labels(test_match.index),
        K=[1, 5, 10, 100, 1000]
    )
    tf.logging.info('FPR95: {:.3f}'.format(fpr95))
    tf.logging.info('Retrieval Recall Rate: {}'.format(rrr))
    test_scalar_plot.update(0, [fpr95])
    test_scalar_plot.update(1, rrr)
    tf.logging.info('-'*80)

    if epoch % param.save_every == 0:
        estimator.save(param.project_name, global_step=epoch)

tf.logging.info('='*80)
tf.logging.info('End training ...')
tf.logging.info('='*80)

train_scalar_plot.hold()
test_scalar_plot.hold()