import tensorflow as tf

class Net(object):
    def __init__(self,
                 name='ftfy',
                 cell_size=(8, 8),
                 n_bbox_estimators=2,
                 n_parameters=5,
                 weight_decay=0.0001):
        self.name = name

        self.cell_size = cell_size
        self.n_bbox_estimators = n_bbox_estimators
        self.n_parameters = n_parameters
        self.weight_decay = weight_decay

        self.w_args = {
            'kernel_initializer': tf.initializers.variance_scaling(2),
            'kernel_regularizer': lambda w: self.weight_decay * tf.nn.l2_loss(w)
        }

    def _filter(self, features, trainable, is_training):
        conv_args = {
            'padding': 'same',
            'trainable': trainable,
            **self.w_args
        }
        bn_args = {
            'training': is_training,
            'trainable': trainable
        }
        with tf.variable_scope('filter_generator'):
            features_reshaped = tf.reshape(
                tf.transpose(features, [0, 3, 1, 2]),
                [-1, self.cell_size[0], self.cell_size[1], 1]
            )

            h = tf.layers.conv2d(
                features_reshaped, 512, self.cell_size, self.cell_size, **conv_args, name='conv1')
            h = tf.layers.batch_normalization(h, **bn_args, name='bn1')
            h = tf.nn.leaky_relu(h)

            h = tf.layers.conv2d(h, 256, 1, 1, **conv_args, name='conv2')
            h = tf.layers.batch_normalization(h, **bn_args, name='bn2')
            h = tf.nn.leaky_relu(h)

            filters = tf.transpose(
                tf.reshape(h, [-1, 512, 1, 1, 256]),
                [0, 2, 3, 1, 4],
                name='filters'
            )
        return filters

    def _matching(self, features, filters, trainable, is_training):
        bn_args = {
            'training': is_training,
            'trainable': trainable
        }
        with tf.variable_scope('matching'):
            filters_reshaped = tf.reshape(tf.transpose(filters, [1, 2, 0, 3, 4]), [1, 1, -1, 256])

            feat_shape = tf.shape(features)
            gy, gx = feat_shape[1], feat_shape[2]
            features_reshaped = tf.reshape(tf.transpose(features, [1, 2, 0, 3]), [1, gy, gx, -1])

            feat_f = tf.nn.depthwise_conv2d(
                features_reshaped,
                filter=filters_reshaped,
                strides=[1, 1, 1, 1],
                padding='SAME'
            )
            feat_f = tf.reshape(feat_f, [gy, gx, -1, 512, 256])
            feat_f = tf.reduce_sum(tf.transpose(feat_f, [2, 0, 1, 3, 4]), axis=3)

            feat_f = tf.layers.batch_normalization(feat_f, **bn_args, name='bn')
            feat_f = tf.nn.leaky_relu(feat_f)
        return feat_f

    def _bbox_prediction(self, features, trainable, is_training):
        conv_args = {
            'padding': 'same',
            'trainable': trainable,
            **self.w_args
        }
        bn_args = {
            'training': is_training,
            'trainable': trainable
        }

        with tf.variable_scope('bbox_prediction'):
            h = tf.layers.conv2d(features, 512, 3, **conv_args, name='conv1')
            h = tf.layers.batch_normalization(h, **bn_args, name='bn1')
            h = tf.nn.leaky_relu(h)

            h = tf.layers.conv2d(h, 256, 3, **conv_args, name='conv2')
            h = tf.layers.batch_normalization(h, **bn_args, name='bn2')
            h = tf.nn.leaky_relu(h)

            h = tf.layers.conv2d(h, 128, 1, **conv_args, name='conv3')
            h = tf.layers.batch_normalization(h, **bn_args, name='bn3')
            h = tf.nn.leaky_relu(h)

            h = tf.layers.conv2d(h, 256, 3, **conv_args, name='conv4')
            h = tf.layers.batch_normalization(h, **bn_args, name='bn4')
            h = tf.nn.leaky_relu(h)

            h = tf.layers.conv2d(h, 128, 1, **conv_args, name='conv5')
            h = tf.layers.batch_normalization(h, **bn_args, name='bn5')
            h = tf.nn.leaky_relu(h)

            h = tf.layers.conv2d(h, 256, 3, **conv_args, name='conv6')
            h = tf.layers.batch_normalization(h, **bn_args, name='bn6')
            h = tf.nn.leaky_relu(h)

            N = self.n_parameters * self.n_bbox_estimators
            h = tf.layers.conv2d(h, N, 1, **conv_args, name='conv7')
            h = tf.nn.sigmoid(h)
        return h


    def __call__(self, src, tar, is_training, **kwargs):
        trainable = kwargs.get('trainable', True)

        # filter generator
        filters = self._filter(tar, trainable, is_training)

        # filtering (matching)
        with tf.control_dependencies([src, filters]):
            src_filtered = self._matching(src, filters, trainable, is_training)

        # predict bboxes
        logits = self._bbox_prediction(src_filtered, trainable, is_training)
        return logits