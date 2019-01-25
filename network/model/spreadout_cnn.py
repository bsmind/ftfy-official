'''CNN from spread-out local feature paper

Learning Spread-out Local Feature Descriptors
Xu Zhang et al.
ICCV, 2017

'''
import tensorflow as tf
from network.model.base import BaseNet

class Net(BaseNet):
    def __init__(self, n_feats, weight_decay, reuse=None, name='spreadout_cnn'):
        super().__init__(name)
        self.weight_decay = weight_decay
        self.n_feats = n_feats
        self.reuse = reuse

        self.layer_names = ['conv_1', 'conv_2', 'conv_3', 'fc_1']
        self.layers = []

    def call(self, images, is_training, **kwargs):
        bn_prefix = kwargs.pop("bn_prefix", "")
        # tf.initializers.variance_scaling(2),
        w_args = {
            'kernel_initializer': tf.truncated_normal_initializer(stddev=.1),
            #'kernel_regularizer': lambda w: self.weight_decay * tf.nn.l2_loss(w)
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

        layers = dict()
        with tf.variable_scope(self.name, reuse=self.reuse):
            net = tf.layers.conv2d(images, 32, 7, 1, name='conv_1', **conv_args)
            net = tf.layers.batch_normalization(net, **bn_args, name=bn_prefix + 'bn1')
            net = tf.nn.relu(net)
            layers['conv_1'] = tf.identity(net)

            net = tf.layers.max_pooling2d(net, **pool_args)

            net = tf.layers.conv2d(net, 64, 6, 1, name='conv_2', **conv_args)
            net = tf.layers.batch_normalization(net, **bn_args, name=bn_prefix + 'bn2')
            net = tf.nn.relu(net)
            layers['conv_2'] = tf.identity(net)

            net = tf.layers.max_pooling2d(net, **pool_args)

            net = tf.layers.conv2d(net, 128, 5, 1, name='conv_3', **conv_args)
            net = tf.layers.batch_normalization(net, **bn_args, name=bn_prefix + 'bn3')
            net = tf.nn.relu(net)
            layers['conv_3'] = tf.identity(net)

            net = tf.layers.max_pooling2d(net, **pool_args)

            net = tf.layers.flatten(net)
            #net = tf.layers.dropout(net, rate=0.5, training=is_training)
            net = tf.layers.dense(net, self.n_feats, name='fc_1', **fc_args)
            net = tf.nn.l2_normalize(net, axis=1)

            layers['fc_1'] = tf.identity(net)

        self.layers.append(layers)
        return net

























