import tensorflow as tf
from network.model.ftfy_cnn import Net
from network.train import TripletSpec
from network.loss.triplet import batch_offline_triplet_loss

def triplet_model_fn(
        anchors, positives, negatives, n_feats=32, name='triplet-net',
        mode = 'TRAIN', margin = 0.3, use_regularization_loss = True, learning_rate = 0.0001
):
    if mode not in ['TRAIN', 'TEST']:
        raise ValueError("mode must be one of [TRAIN, TEST].")

    is_training = tf.placeholder(tf.bool, (), "is_training")

    build_model = Net(n_feats=n_feats,  reuse=tf.AUTO_REUSE, name=name)
    a_feat = build_model(anchors, is_training=is_training, bn_prefix="anchor")
    p_feat = build_model(positives, is_training=is_training, bn_prefix="positive")
    n_feat = build_model(negatives, is_training=is_training, bn_prefix="negative")

    triplet_loss = None
    regularization_loss = None
    train_op = None
    if mode is 'TRAIN':
        with tf.name_scope('loss'):
            triplet_loss = batch_offline_triplet_loss(a_feat, p_feat, n_feat, margin)
            if use_regularization_loss:
                regularization_loss = tf.losses.get_regularization_loss()
                loss = triplet_loss + regularization_loss
            else:
                loss = triplet_loss

        global_step = tf.train.create_global_step()
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        grads_and_vars = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(train_op, *update_ops)

    return TripletSpec(
        anchors=anchors,
        positives=positives,
        negatives=negatives,
        a_feat=a_feat,
        p_feat=p_feat,
        n_feat=n_feat,
        train_op=train_op,
        triplet_loss=triplet_loss,
        regularization_loss=regularization_loss,
        train_feed_dict={is_training: True},
        test_feed_dict={is_training: False}
    )