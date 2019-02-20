import tensorflow as tf

from network.model.simple_cnn import Net as simplet_net
from network.model.ftfy_cnn import Net as ftfy_net
from network.model.spreadout_cnn import Net as spread_net

from network.train import TripletSpec
from network.loss.triplet import batch_offline_triplet_loss, spreadout_triplet_loss

MODE = ['TRAIN', 'TEST']
LOSS = ['triplet', 'spreadout']
OPTIMIZER = ['Adam', 'Momentum']
CNN = ['simple', 'ftfy', 'spread']

def get_cnn_model(cnn_name):
    if cnn_name == 'simple':
        return simplet_net
    elif cnn_name == 'ftfy':
        return ftfy_net
    elif cnn_name == 'spread':
        return spread_net
    else:
        raise ValueError('Unknown cnn model: %s' % cnn_name)

def get_triplet_loss(loss_name, a, p, n, **kwargs):
    if loss_name == 'triplet':
        margin = kwargs.get('margin', None)
        loss = batch_offline_triplet_loss(a, p, n, margin)
    elif loss_name == 'spreadout':
        loss = spreadout_triplet_loss(a, p, n)
    else:
        raise ValueError('Unknown loss name: %s' % loss_name)
    return loss

def get_optimizer(optimizer_name, **kwargs):
    if optimizer_name == 'Adam':
        learning_rate = kwargs.get('learning_rate', None)
        return tf.train.AdagradOptimizer(learning_rate=learning_rate)
    elif optimizer_name == 'Momentum':
        global_step = tf.train.get_global_step()
        decay_steps = 10000
        start_learning_rate = 0.1
        learning_rate = tf.train.exponential_decay(
            learning_rate=start_learning_rate,
            global_step=global_step,
            decay_steps=decay_steps,
            decay_rate=0.96,
            staircase=True
        )
        return tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=0.9
        )
    else:
        raise ValueError('Unknown optimizer: %s' % optimizer_name)

def triplet_model_fn(
        anchors, positives, negatives, n_feats=128,
        mode='TRAIN', cnn_name='spread', loss_name='triplet', optimizer_name='Adam',
        margin=None, # todo: replace with loss_kwargs
        use_regularization_loss = False,
        learning_rate = 0.0001, # todo: replace with opt_kwargs
        shared_batch_layers = True,
        trainable=True, include_fc=True,
        name='triplet-net'
):
    assert mode in MODE, 'Unknown mode: %s' % mode
    assert cnn_name in CNN, 'Unknown cnn: %s' % cnn_name
    tf.logging.info('CNN: %s' % cnn_name)

    if mode == 'TRAIN':
        assert loss_name in LOSS, 'Unknown loss: %s' % loss_name
        assert optimizer_name in OPTIMIZER, 'Unknown optimizer: %s' % optimizer_name
        tf.logging.info('LOSS: %s' % loss_name)
        tf.logging.info('OPTIMIZER: %s' % optimizer_name)

    is_training = tf.placeholder(tf.bool, (), "is_training")

    Net = get_cnn_model(cnn_name)
    builder = Net(n_feats=n_feats, weight_decay=0.0001, reuse=tf.AUTO_REUSE, name=name)
    a_feat = builder(anchors, is_training=is_training, trainable=trainable, include_fc=include_fc,
                     bn_prefix="" if shared_batch_layers else "a_")
    p_feat = builder(positives, is_training=is_training, trainable=trainable, include_fc=include_fc,
                     bn_prefix="" if shared_batch_layers else "p_")
    n_feat = builder(negatives, is_training=is_training, trainable=trainable, include_fc=include_fc,
                     bn_prefix="" if shared_batch_layers else "n_")

    triplet_loss = None
    regularization_loss = None
    train_op = None
    global_step = None
    if mode is 'TRAIN':
        with tf.name_scope('triplet-loss'):
            triplet_loss = get_triplet_loss(loss_name, a_feat, p_feat, n_feat, margin=margin)
            if use_regularization_loss:
                regularization_loss = tf.losses.get_regularization_loss()
                loss = triplet_loss + regularization_loss
            else:
                loss = triplet_loss

        global_step = tf.train.create_global_step()
        optimizer = get_optimizer(optimizer_name, learning_rate=learning_rate)

        grads_and_vars = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(train_op, *update_ops)

        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        #     train_op = optimizer.minimize(loss, global_step=global_step)

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
        train_feed_dict={is_training: True and trainable},
        test_feed_dict={is_training: False},
        global_step=global_step,
        net=builder
    )

def ftfy_model_fn(

):
    pass


