import tensorflow as tf
import numpy as np
from network.model_fn import triplet_model_fn

class TripletNet(object):
    def __init__(self, model_path, epoch=None, **model_kwargs):
        '''
        Build triplet network with restored parameters in the `model_path`.

        Note that if parameters in model_kwargs didn't match with checkpoint, it will fail to
        restore the checkpoint.

        Args:
            model_path: (str), path to a directory where checkpoints are placed
            epoch: (int), if it is not None and it is valid, select epoch-th checkpoint
            **model_kwargs: model argument including
                mean: (float) mean of training dataset
                std : (float) standard deviation of training dataset
                patch_size: (int) input patch (image) size
                n_channels: (int) the number of channels
                n_feats   : (int) the number of feature dimension,
                cnn_name  : (str) CNN model name in ['ftfy', 'simple', 'spread'],
                shared_batch_layers: (bool) used shared batch layers if True
                name               : (str) model name
        '''
        self.mean = model_kwargs.pop('mean', 0.)
        self.std  = model_kwargs.pop('std', 1.)
        self.patch_size = model_kwargs.pop('patch_size', None)
        self.n_channels = model_kwargs.pop('n_channels', None)
        assert self.patch_size is not None, 'Missing input patch_size!'
        assert self.n_channels is not None, 'Missing the number of image channel!'

        input_shape = (None, self.patch_size, self.patch_size, self.n_channels)
        self.anchors   = tf.placeholder(tf.float32, input_shape, 'ph_a')
        self.positives = tf.placeholder(tf.float32, input_shape, 'ph_p')
        self.negatives = tf.placeholder(tf.float32, input_shape, 'ph_n')

        self.spec = triplet_model_fn(self.anchors, self.positives, self.negatives,
                                     mode='TEST', **model_kwargs)

        config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            intra_op_parallelism_threads=8,
            inter_op_parallelism_threads=0
        )
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.saver = tf.train.Saver()

        # initialize
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt is None:
            raise ValueError('Cannot find a checkpoint: %s' % model_path)
        model_checkpoint_path = ckpt.model_checkpoint_path
        if epoch is not None:
            all_model_checkpoint_paths = ckpt.all_model_checkpoint_path
            if 0 <= epoch and epoch < len(all_model_checkpoint_paths):
                model_checkpoint_path = all_model_checkpoint_paths[epoch]
        self.saver.restore(self.sess, model_checkpoint_path)

    def normalize(self, images):
        for i in range(len(images)):
            images[i] = (images[i] - self.mean) / self.std
        return images

    def get_feature(self, images, batch_size=48, is_normalized=False):
        input_shape = images.shape
        assert len(input_shape) == 4, 'Required to have 4-dimension input.'
        assert input_shape[1] == self.patch_size and input_shape[2] == self.patch_size, \
            'Unmatched input image size, must {:d} x {:d}'.format(self.patch_size, self.patch_size)
        assert input_shape[3] == self.n_channels, 'Unmatched number of channels, must %d' % self.n_channels

        # normalize
        if not is_normalized:
            images = self.normalize(images)
        image_features = []

        n_images = input_shape[0]
        batch_iters = (n_images + batch_size) // batch_size
        for i_batch in range(batch_iters):
            start = i_batch*batch_size
            end = min(n_images, (i_batch+1)*batch_size)

            feed_dict = dict(self.spec.test_feed_dict)
            feed_dict[self.anchors] = images[start: end]
            features = self.sess.run(self.spec.a_feat, feed_dict=feed_dict)
            image_features.append(features)

            if end == n_images:
                break

        image_features = np.concatenate(image_features, axis=0)
        return image_features

