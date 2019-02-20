import tensorflow as tf
from network.model.base import BaseNet

class Net(BaseNet):
    def __init__(self, n_feats, weight_decay=0.0001, reuse=None, name="ftfy_cnn"):
        super().__init__(name)

        self.weight_decay = weight_decay
        self.n_feats = n_feats
        self.reuse = reuse

        self.created = False

    def call(self, images, is_training, **kwargs):
        bn_prefix = kwargs.get("bn_prefix", "")
        trainable = kwargs.get("trainable", True)

        w_args = {
            'kernel_initializer': tf.initializers.variance_scaling(2),
            'kernel_regularizer': lambda w: self.weight_decay * tf.nn.l2_loss(w)
        }
        conv_args = {
            'padding': 'same',
            'trainable': trainable,
            **w_args
        }
        fc_args = {
            **w_args,
            'trainable': trainable
        }
        bn_args = {
            'training': is_training,
            'trainable': trainable
        }
        pool_args = {
            'pool_size': 2,
            'strides': 2,
            'padding': 'same'
        }


        with tf.variable_scope(self.name, reuse=self.reuse):
            ##
            net = tf.layers.conv2d(images,  32, 3, 1, name='conv1', **conv_args)
            net = tf.layers.batch_normalization(net, **bn_args, name=bn_prefix + '_bn1')
            net = tf.nn.leaky_relu(net)

            if not self.created:
                self.conv1 = tf.identity(net)

            ##
            net = tf.layers.conv2d(net,  64, 3, 1, name='conv2', **conv_args)
            net = tf.layers.batch_normalization(net, **bn_args, name=bn_prefix + '_bn2')
            net = tf.nn.leaky_relu(net)

            net = tf.layers.average_pooling2d(net, **pool_args)
            if not self.created:
                self.conv2 = tf.identity(net)

            ##
            net = tf.layers.conv2d(net, 128, 3, 1, name='conv3', **conv_args)
            net = tf.layers.batch_normalization(net, **bn_args, name=bn_prefix + '_bn3')
            net = tf.nn.leaky_relu(net)
            if not self.created:
                self.conv3 = tf.identity(net)

            net = tf.layers.conv2d(net, 64, 1, 1, name='conv4', **conv_args)
            net = tf.layers.batch_normalization(net, **bn_args, name=bn_prefix + '_bn4')
            net = tf.nn.leaky_relu(net)
            if not self.created:
                self.conv4 = tf.identity(net)

            net = tf.layers.conv2d(net, 128, 3, 1, name='conv5', **conv_args)
            net = tf.layers.batch_normalization(net, **bn_args, name=bn_prefix + '_bn5')
            net = tf.nn.leaky_relu(net)

            net = tf.layers.average_pooling2d(net, **pool_args)
            net_4 = tf.identity(net)
            if not self.created:
                self.conv5 = tf.identity(net)

            ##
            net = tf.layers.conv2d(net, 256, 3, 1, name='conv6', **conv_args)
            net = tf.layers.batch_normalization(net, **bn_args, name=bn_prefix + '_bn6')
            net = tf.nn.leaky_relu(net)
            if not self.created:
                self.conv6 = tf.identity(net)

            net = tf.layers.conv2d(net, 128, 1, 1, name='conv7', **conv_args)
            net = tf.layers.batch_normalization(net, **bn_args, name=bn_prefix + '_bn7')
            net = tf.nn.leaky_relu(net)
            if not self.created:
                self.conv7 = tf.identity(net)

            net = tf.layers.conv2d(net, 256, 3, 1, name='conv8', **conv_args)
            net = tf.layers.batch_normalization(net, **bn_args, name=bn_prefix + '_bn8')
            net = tf.nn.leaky_relu(net)

            net = tf.layers.average_pooling2d(net, **pool_args)
            net_2 = tf.identity(net)
            if not self.created:
                self.conv8 = tf.identity(net)

            ##
            net = tf.layers.conv2d(net, 512, 3, 1, name='conv9', **conv_args)
            net = tf.layers.batch_normalization(net, **bn_args, name=bn_prefix + '_bn9')
            net = tf.nn.leaky_relu(net)
            if not self.created:
                self.conv9 = tf.identity(net)

            net = tf.layers.conv2d(net, 256, 1, 1, name='conv10', **conv_args)
            net = tf.layers.batch_normalization(net, **bn_args, name=bn_prefix + '_bn10')
            net = tf.nn.leaky_relu(net)
            if not self.created:
                self.conv10 = tf.identity(net)

            net = tf.layers.conv2d(net, 512, 3, 1, name='conv11', **conv_args)
            net = tf.layers.batch_normalization(net, **bn_args, name=bn_prefix + '_bn11')
            net = tf.nn.leaky_relu(net)
            net = tf.layers.average_pooling2d(net, **pool_args)
            if not self.created:
                self.conv11 = tf.identity(net)

            ##
            merged = tf.concat([
                net,
                tf.space_to_depth(net_2, block_size=2),
                tf.space_to_depth(net_4, block_size=4)
            ], axis=3, name='merged')
            if not self.created:
                self.merged = tf.identity(merged)

            ##
            net = tf.layers.conv2d(merged, 1024, 3, 1, name='conv12', **conv_args)
            net = tf.layers.batch_normalization(net, **bn_args, name=bn_prefix + '_bn12')
            net = tf.nn.leaky_relu(net)
            if not self.created:
                self.conv12 = tf.identity(net)

            net = tf.layers.conv2d(net, 512, 1, 1, name='conv13', **conv_args)
            net = tf.layers.batch_normalization(net, **bn_args, name=bn_prefix + '_bn13')
            net = tf.nn.leaky_relu(net)
            if not self.created:
                self.conv13 = tf.identity(net)

            ###
            net = tf.layers.flatten(net)
            net = tf.layers.dropout(net, rate=0.5, training=is_training)
            net = tf.layers.dense(net, self.n_feats, name='fc1', **fc_args)
            net = tf.layers.batch_normalization(net, **bn_args, name=bn_prefix + '_bn14')
            net = tf.nn.l2_normalize(net, axis=1)
            if not self.created:
                self.fc1 = tf.identity(net)
            self.created = True

        return net




