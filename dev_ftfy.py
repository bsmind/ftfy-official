import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)
import numpy as np

from network.model_fn import triplet_model_fn
from network.model.ftfy import Net as ftfy_net
from network.loss.ftfy import loss

ph_src = tf.placeholder(tf.float32, (None, 256, 256, 1), "source")
ph_tar = tf.placeholder(tf.float32, (None, 128, 128, 1), "target")
ph_is_training = tf.placeholder(tf.bool, (), "ftfy_is_training")
ph_labels = tf.placeholder(tf.float32, (None, 16, 16, 5), "labels")

# triplet network for feature extractor
triplet_spec = triplet_model_fn(
    ph_src, ph_tar, ph_tar,
    n_feats=128,
    mode='TEST',
    cnn_name='ftfy',
    shared_batch_layers=True,
    trainable=False,
    include_fc=False,
    name='triplet-net'
)

# todo: build ftfy network
ftfy_builder = ftfy_net(
    name='ftfy',
    cell_size=8,
    n_bbox_estimators=2,
    n_parameters=5
)
logits = ftfy_builder(triplet_spec.a_feat, triplet_spec.p_feat, ph_is_training, trainable=True)
obj_loss, noobj_loss, coord_loss = loss(logits, ph_labels, 2, 5)

# initialize
config = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False,
    intra_op_parallelism_threads=8,
    inter_op_parallelism_threads=0
)
config.gpu_options.allow_growth = True

vars_triplet = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='triplet-net')
saver = tf.train.Saver(vars_triplet)

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    ckpt = tf.train.get_checkpoint_state('./log/sem/ckpt')
    saver.restore(sess, ckpt.model_checkpoint_path)

    test_src = np.random.random((5, 256, 256, 1)).astype(np.float32)
    test_tar = np.random.random((5, 128, 128, 1)).astype(np.float32)
    test_labels = np.random.random((5, 16, 16, 5)).astype(np.float32)

    feed_dict = dict(triplet_spec.test_feed_dict)
    feed_dict[ph_src] = test_src
    feed_dict[ph_tar] = test_tar
    feed_dict[ph_labels] = test_labels
    feed_dict[ph_is_training] = False

    src_feat, tar_feat, logits_output, obj_loss_output, noobj_loss_output, coord_loss_output = \
        sess.run(
        [triplet_spec.a_feat, triplet_spec.p_feat, logits, obj_loss, noobj_loss, coord_loss],
        feed_dict=feed_dict
    )

    print(src_feat.shape)
    print(tar_feat.shape)
    print(logits_output.shape)
    print(obj_loss_output, noobj_loss_output, coord_loss_output)