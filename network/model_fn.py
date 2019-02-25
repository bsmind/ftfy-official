import tensorflow as tf

from network.model.simple_cnn import Net as simplet_net
from network.model.ftfy_cnn import Net as ftfy_net
from network.model.spreadout_cnn import Net as spread_net
from network.model.ftfy import Net as FTFY

from network.train import TripletSpec, FTFYSpec
from network.loss.triplet import batch_offline_triplet_loss, spreadout_triplet_loss
from network.loss.ftfy import loss as ftfy_loss, inference as ftfy_pred

MODE = ['TRAIN', 'TEST']
LOSS = ['triplet', 'spreadout']
OPTIMIZER = ['Adam', 'Momentum', 'Grad']
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
    learning_rate = kwargs.pop('learning_rate', 0.001)
    if optimizer_name == 'Adam':
        return tf.train.AdagradOptimizer(learning_rate=learning_rate)
    elif optimizer_name == 'Grad':
        decay_steps = kwargs.pop('decay_steps', 10000)
        decay_rate = kwargs.pop('decay_rate', 0.9)
        learning_rate = tf.train.exponential_decay(
            learning_rate=learning_rate,
            global_step=tf.train.get_global_step(),
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=True
        )
        return tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    elif optimizer_name == 'Momentum':
        decay_steps = kwargs.pop('decay_steps', 10000)
        decay_rate = kwargs.pop('decay_rate', 0.9)
        momentum = kwargs.pop('momentum', 0.9)
        learning_rate = tf.train.exponential_decay(
            learning_rate=learning_rate,
            global_step=tf.train.get_global_step(),
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=True
        )
        return tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum)
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
    # inputs
    sources, targets, labels=None, bboxes=None,
    # feature extractor
    feat_name='ftfy', feat_trainable=False,
    feat_shared_batch_layers=True, feat_scope='triplet-net',
    # ftfy
    ftfy_ver='v0', ftfy_scope='ftfy',
    tar_cell_size=(8,8), src_cell_size=(16,16), n_bbox_estimators=2, n_parameters=5,
    # loss and optimizer
    loss_name='rms', obj_scale=1.0, noobj_scale=0.5, coord_scale=5.0,
    use_regularization_loss=False,
    optimizer_name='Momentum', optimizer_kwargs=None,
    mode='TRAIN'
):
    assert mode in MODE, 'Unknown mode: %s' % mode
    assert feat_name in CNN, 'Unknown feature: %s' % feat_name
    tf.logging.info('FEATURE: %s' % feat_name)

    if mode == 'TRAIN':
        # todo: assert for loss_name
        assert optimizer_name in OPTIMIZER, 'Unknown optimizer: %s' % optimizer_name
        tf.logging.info('LOSS: %s' % loss_name)
        tf.logging.info('OPTIMIZER: %s' % optimizer_name)

    feat_is_training = tf.placeholder(tf.bool, (), "feat_is_training")
    ftfy_is_training = tf.placeholder(tf.bool, (), "ftfy_is_training")

    feat_net = get_cnn_model(feat_name)
    feat_builder = feat_net(reuse=tf.AUTO_REUSE, name=feat_scope)
    src_feat = feat_builder(sources, is_training=feat_is_training, trainable=feat_trainable,
                            include_fc=False, bn_prefix="" if feat_shared_batch_layers else "a_")
    tar_feat = feat_builder(targets, is_training=feat_is_training, trainable=feat_trainable,
                            include_fc=False, bn_prefix="" if feat_shared_batch_layers else "p_")

    # todo: use ftfy_ver, if other versions are available...
    ftfy_builder = FTFY(name=ftfy_scope,
                        cell_size=tar_cell_size,
                        n_bbox_estimators=n_bbox_estimators,
                        n_parameters=n_parameters)

    logits = ftfy_builder(src_feat, tar_feat, ftfy_is_training, trainable=True)
    pred_confidence, pred_bboxes = ftfy_pred(logits, n_bbox_estimators, n_parameters,
                                             *src_cell_size)

    obj_loss, noobj_loss, coord_loss, total_loss = None, None, None, None
    regularization_loss = None
    train_op = None
    global_step = None
    if mode is 'TRAIN':
        # todo: get loss function, if other losses are available
        # note that variable_scope for ftfy_loss is defined internally
        obj_loss, noobj_loss, coord_loss = ftfy_loss(
            logits, labels, n_bbox_estimators, n_parameters, *src_cell_size
        )
        obj_loss = obj_scale * obj_loss
        noobj_loss = noobj_scale * noobj_loss
        coord_loss = coord_scale * coord_loss
        total_loss = obj_loss + noobj_loss + coord_loss
        if use_regularization_loss:
            regularization_loss = tf.losses.get_regularization_loss()
            total_loss += regularization_loss

        global_step = tf.train.create_global_step()
        optimizer = get_optimizer(optimizer_name, **optimizer_kwargs)

        grads_and_vars = optimizer.compute_gradients(total_loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(train_op, *update_ops)

    return FTFYSpec(
        sources, targets, labels, bboxes,
        src_feat, tar_feat, logits,
        pred_confidence, pred_bboxes,
        train_op=train_op,
        obj_loss=obj_loss, noobj_loss=noobj_loss, coord_loss=coord_loss,
        regularization_loss=regularization_loss, total_loss=total_loss,
        global_step=global_step,
        train_feed_dict={feat_is_training: True and feat_trainable, ftfy_is_training: True},
        test_feed_dict={feat_is_training: False, ftfy_is_training: False},
        feat_net=feat_builder, ftfy_net=ftfy_builder
    )






















