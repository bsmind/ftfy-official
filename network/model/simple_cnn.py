import tensorflow as tf
from network.model.base import BaseNet

class Net(BaseNet):
    def __init__(self, n_feats, weight_decay=0.0001, reuse=None, name="simple_cnn"):
        super().__init__(name)

        self.weight_decay = weight_decay
        self.n_feats = n_feats
        self.reuse = reuse

    def call(self, images, is_training, **kwargs):
        bn_prefix = kwargs.pop("bn_prefix", "")

        w_args = {
            'kernel_initializer': tf.initializers.variance_scaling(2),
            'kernel_regularizer': lambda w: self.weight_decay * tf.nn.l2_loss(w)
        }
        conv_args = {
            'padding': 'same',
            **w_args
        }
        fc_args = {
            **w_args
        }
        bn_args = {
            'training': is_training
        }
        pool_args = {
            'pool_size': 2,
            'strides': 2,
            'padding': 'same'
        }

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            #net = tf.layers.batch_normalization(images, **bn_args, name=bn_prefix + '_bn0')

            net = tf.layers.conv2d(images,  32, 3, 1, name='conv1_1', **conv_args)
            net = tf.layers.batch_normalization(net, **bn_args, name=bn_prefix + '_bn1_1')
            net = tf.nn.relu(net)

            net = tf.layers.conv2d(net, 32, 3, 1, name='conv1_2', **conv_args)
            net = tf.layers.batch_normalization(net, **bn_args, name=bn_prefix + '_bn1_2')
            net = tf.nn.relu(net)
            #net = tf.layers.max_pooling2d(net, 2, 2, padding='same')
            net = tf.layers.average_pooling2d(net, **pool_args)

            net = tf.layers.conv2d(net   ,  64, 3, 1, name='conv2_1', **conv_args)
            net = tf.layers.batch_normalization(net, **bn_args, name=bn_prefix + '_bn2_1')
            net = tf.nn.relu(net)

            net = tf.layers.conv2d(net   ,  64, 3, 1, name='conv2_2', **conv_args)
            net = tf.layers.batch_normalization(net, **bn_args, name=bn_prefix + '_bn2_2')
            net = tf.nn.relu(net)
            #net = tf.layers.max_pooling2d(net, 2, 2, padding='same')
            net = tf.layers.average_pooling2d(net, **pool_args)

            net = tf.layers.conv2d(net, 128, 3, 1, name='conv3_1', **conv_args)
            net = tf.layers.batch_normalization(net, **bn_args, name=bn_prefix + '_bn3_1')
            net = tf.nn.relu(net)

            net = tf.layers.conv2d(net   , 128, 3, 1, name='conv3_2', **conv_args)
            net = tf.layers.batch_normalization(net, **bn_args, name=bn_prefix + '_bn3_2')
            net = tf.nn.relu(net)
            #net = tf.layers.max_pooling2d(net, 2, 2, padding='same')
            net = tf.layers.average_pooling2d(net, **pool_args)

            # net = tf.layers.conv2d(net, 256, 3, 1, name='conv4_1', **conv_args)
            # net = tf.layers.conv2d(net   , 256, 3, 1, name='conv4_2', **conv_args)
            # net = tf.layers.batch_normalization(net, **bn_args, name=bn_prefix + '_bn4')
            # net = tf.nn.relu(net)

            net = tf.layers.flatten(net)
            net = tf.layers.dropout(net, rate=0.5, training=is_training)
            net = tf.layers.dense(net, self.n_feats, name='fc1', **fc_args)
            net = tf.layers.batch_normalization(net, **bn_args, name=bn_prefix + '_bn4')
            net = tf.nn.l2_normalize(net, axis=1)
            #net = tf.nn.sigmoid(net)

        return net

if __name__ == "__main__":
    import numpy as np
    tf.reset_default_graph()

    anchors = tf.placeholder(tf.float32, (None, 28, 28, 1), "anchors")
    positives = tf.placeholder(tf.float32, (None, 28, 28, 1), "positives")
    negatives = tf.placeholder(tf.float32, (None, 28, 28, 1), "negatives")
    is_training = tf.placeholder(tf.bool, [], "is_training")

    n_feats = 2
    weight_decay = 0.0001

    net = Net(n_feats, weight_decay)

    a_embed = net(anchors, is_training, bn_prefix = "a")
    p_embed = net(positives, is_training, bn_prefix = "p")
    n_embed = net(negatives, is_training, bn_prefix = "n")

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    sess = tf.Session()
    sess.run(init_op)

    writer = tf.summary.FileWriter(
        "../../logs",
        sess.graph
    )

    for _ in range(10000):
        sess.run([a_embed, p_embed, n_embed], feed_dict={
            anchors: np.random.random((100, 28, 28, 1)),
            positives: np.random.random((100, 28, 28, 1)),
            negatives: np.random.random((100, 28, 28, 1)),
            is_training: False
        })


    writer.close()



