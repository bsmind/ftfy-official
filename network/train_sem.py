import os
# set tensorflow cpp log level. It is useful to diable some annoying log message,
# but sometime may miss some useful information
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf

from network.dataset.sem_input_fn import input_fn
import network.dataset.sem_dataset as sem
#from network.model.simple_cnn import Net as model
from network.model.ftfy_cnn import Net as model
from network.loss.triplet import batch_offline_triplet_loss

from utils.eval import Evaluator
from utils.viz import RetrievalPlot

import matplotlib.pyplot as plt



# set seed
np.random.seed(0)
tf.set_random_seed(0)


# parameters
data_dir = '/home/sungsooha/Desktop/Data/ftfy/data_hxnsem_selected'
patch_size = (208, 208)
n_channels = 1
iou_thresholds = [0.7, 0.6]
n_img_per_iter = 10
n_crops_per_img = 20
batch_size = 16

n_epoch = 10
init_learning_rate = 0.0001
log_every = 50
n_iter_for_embed = 20
tsne_dim = 2

n_retrieval_test = 10
n_queries = batch_size * 3

top_K = 5
fig_nrow = min(batch_size, 3)
fig_ncol = top_K + 1


n_feats = 32
margin = 0.3

# for evaluation
down_factors = [2, 3, 4, 5]

# dataset pipeline
files_by_zoom = sem.filelist_by_zoomfactor(data_dir)
tr_list, t_list = sem.split_train_test(files_by_zoom, train_ratio=.8)
# tr_list = tr_list[:10]
# t_list = t_list[:10]

dataset_wargs = {
    'patch_size': patch_size,
    'n_channels': n_channels,
    'iou_thresholds': iou_thresholds,
    'n_img_per_iter': n_img_per_iter,
    'n_crops_per_img': n_crops_per_img,
    'batch_size': batch_size
}
with tf.device('/cpu:0'), tf.name_scope('input'):
    # for training/evalution dataset
    train_dataset,_,_ = input_fn(data_dir, tr_list, dataset_wargs, is_training=True)
    test_dataset, test_data_sampler, test_data = input_fn(data_dir, t_list, dataset_wargs,
                                                          is_training=False)

    data_iterator = tf.data.Iterator.from_structure(
        train_dataset.output_types,
        train_dataset.output_shapes
    )
    train_dataset_init = data_iterator.make_initializer(train_dataset)
    test_dataset_init = data_iterator.make_initializer(test_dataset)
    batch_data = data_iterator.get_next()

    # for evaluation
    eval = Evaluator(test_data, patch_size)


# placeholder & variables
is_training = tf.placeholder(tf.bool, (), "is_training")
global_step = tf.train.create_global_step()

# build network and get embeddings
batch_anchors, batch_positives, batch_negatives = batch_data
net = model(n_feats=n_feats, weight_decay=0.0001)
a_embed = net(batch_anchors, is_training, bn_prefix="a")
p_embed = net(batch_positives, is_training, bn_prefix="p")
n_embed = net(batch_negatives, is_training, bn_prefix="n")

# loss
with tf.name_scope('losses'):
    loss = batch_offline_triplet_loss(a_embed, p_embed, n_embed, margin)
    l2_reg = tf.losses.get_regularization_loss()

# optimizer & train operation
optimizer = tf.train.AdamOptimizer(init_learning_rate)

grads_and_vars = optimizer.compute_gradients(loss + l2_reg)
train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
train_op = tf.group(train_op, *update_ops)

# training loop
config = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False,
    intra_op_parallelism_threads=8,
    inter_op_parallelism_threads=0
)
config.gpu_options.allow_growth = True



plot = RetrievalPlot(
    n_examples=3,
    n_queries=3,
    top_k=top_K,
    n_scalars=1,
    colors = ['yellow', 'red', 'orange', 'magenta'],
    image_size=(850, 1280),
    patch_size=patch_size
)

with tf.Session(config=config) as sess:
    # initialize global/local variables
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    eval_accuracy = []
    for epoch in range(n_epoch):
        # validation step
        '''
        Validate the trained model:
            - Methods:
                1. randomly pick an image to build a database
                2. randomly pick image patches (n) from the image
                3. calculate top K image patches
                4. repeat 1-3 (m) times
            - Measurement:
                For given a database and an image patch, we regard the test is succeed if one of
                top K image patches has IoU > 0.7.
                    retrieving_accuracy = (# success) / (n*m)
            - Visualization:
                Show the image with top K bounding boxes and image patches, 
                and queried image patch. 
        '''
        print('-' * 40)
        print('Validation Epoch: {:d}'.format(epoch))
        accuracy = []
        for eval_iter in range(n_retrieval_test):
            val_image = eval.get_image()
            eval.reset_db()

            for val_iter in range(2):
                build_db = True if val_iter == 0 else False
                try:
                    # initialize dataset
                    test_data_sampler.reset(val_image,
                                            stride=(patch_size[0]//8, patch_size[1]//8),
                                            patch_size=patch_size,
                                            random_sampling=not build_db)
                    sess.run(test_dataset_init)
                    step = 0
                    while True:
                        _a_embed, _a_img, _r, _c = sess.run(
                            [a_embed, batch_anchors, batch_positives, batch_negatives],
                            feed_dict={is_training: False})
                        _r = _r.astype(np.int32)
                        _r = _r[:, 0, 0, 0]
                        _c = _c.astype(np.int32)
                        _c = _c[:, 0, 0, 0]

                        eval.add_item(_a_embed, _r, _c, build_db, _a_img)
                        step += 1
                        if not build_db and step * batch_size > n_queries:
                            break
                except tf.errors.OutOfRangeError:
                    pass

            # compute accuracy
            eval.fit()
            acc = eval.get_accuracy(top_k=top_K, iou_threshold=0.7)
            accuracy.append(acc)

            # update plot
            plot.update(
                eval_iter % 3,
                np.squeeze(val_image),
                eval.get_top_k_pos(top_K),
                patch_size
            )

        accuracy = np.array(accuracy)
        accuracy = accuracy.sum() / n_retrieval_test
        eval_accuracy.append(accuracy)
        eval.reset()
        print('avg. accuracy: {:.3f}'.format(eval_accuracy[-1]))
        plot.update_scalar([eval_accuracy])


        #training step
        print('-' * 40)
        print('Training Epoch: {:d}'.format(epoch))
        step = 0
        avg_loss = 0
        try:
            # initialize dataset
            sess.run(train_dataset_init)
            while True:
                outputs = sess.run(
                    [train_op, loss, l2_reg],
                    feed_dict={is_training: True}
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



plt.ioff()
plt.show()
