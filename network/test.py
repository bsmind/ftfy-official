import tensorflow as tf
import numpy as np
from network.model_fn import triplet_model_fn
from network.model_fn import ftfy_model_fn

tf.logging.set_verbosity(tf.logging.INFO)

def tf_normalize(x, size=(128,128), mean=0., std=1., align_corners=True):
    x = tf.image.resize_bilinear(x, size, align_corners)
    x_min = tf.reduce_min(x, axis=[1, 2, 3], keepdims=True)
    x_max = tf.reduce_max(x, axis=[1, 2, 3], keepdims=True)
    x = (x - x_min) / (x_max - x_min)
    x = (x - mean) / std
    return x

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
        self.n_channels = model_kwargs.pop('n_channels', 1)

        input_shape = (None, self.patch_size, self.patch_size, self.n_channels)
        self.anchors   = tf.placeholder(tf.float32, input_shape, 'ph_a')
        self.positives = tf.placeholder(tf.float32, input_shape, 'ph_p')
        self.negatives = tf.placeholder(tf.float32, input_shape, 'ph_n')

        self.is_normalized = False
        self.a = self.anchors
        self.p = self.positives
        self.n = self.negatives
        if self.patch_size is None or self.patch_size != 128:
            self.a = tf_normalize(self.anchors, mean=self.mean, std=self.std)
            self.p = tf_normalize(self.positives, mean=self.mean, std=self.std)
            self.n = tf_normalize(self.negatives, mean=self.mean, std=self.std)
            self.is_normalized = True

        self.spec = triplet_model_fn(self.a, self.p, self.n, mode='TEST', **model_kwargs)
        self.patch_size = 128

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
            all_model_checkpoint_paths = ckpt.all_model_checkpoint_paths
            if 0 <= epoch and epoch < len(all_model_checkpoint_paths):
                model_checkpoint_path = all_model_checkpoint_paths[epoch]
        self.saver.restore(self.sess, model_checkpoint_path)

    def normalize(self, images):
        for i in range(len(images)):
            images[i] = (images[i] - self.mean) / self.std
        return images

    def get_feature(self, images, batch_size=48):
        input_shape = images.shape
        assert len(input_shape) == 4, 'Required to have 4-dimension input.'
        assert input_shape[3] == self.n_channels, 'Unmatched number of channels, must %d' % self.n_channels

        # normalize
        if not self.is_normalized:
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

    def get_input(self, images, batch_size=48):
        input_shape = images.shape
        assert len(input_shape) == 4, 'Required to have 4-dimension input.'
        assert input_shape[3] == self.n_channels, 'Unmatched number of channels, must %d' % self.n_channels

        raw_inputs = []

        n_images = input_shape[0]
        batch_iters = (n_images + batch_size) // batch_size
        for i_batch in range(batch_iters):
            start = i_batch*batch_size
            end = min(n_images, (i_batch+1)*batch_size)

            feed_dict = dict(self.spec.test_feed_dict)
            feed_dict[self.anchors] = images[start: end]
            raw_input = self.sess.run(self.a, feed_dict=feed_dict)
            raw_inputs.append(raw_input)

            if end == n_images:
                break

        raw_inputs = np.concatenate(raw_inputs, axis=0)
        return raw_inputs

class FTFYNet(object):
    def __init__(self, model_path, epoch=None, **model_kwargs):
        self.mean = model_kwargs.pop('mean', 0)
        self.std  = model_kwargs.pop('std', 1)
        self.n_channels = model_kwargs.pop('n_channels', 1)
        self.src_size = model_kwargs.pop('src_size', (256, 256))
        self.src_cell_size = model_kwargs.pop('src_cell_size', (16, 16))

        input_shape = (None, None, None, self.n_channels)
        self.src = tf.placeholder(tf.float32, input_shape, 'ph_a')
        self.tar = tf.placeholder(tf.float32, input_shape, 'ph_p')

        self.is_normalized = True
        self.norm_src = tf_normalize(self.src, self.src_size, mean=self.mean, std=self.std)
        self.norm_tar = tf_normalize(self.tar, (128, 128), mean=self.mean, std=self.std)

        # todo: parsing model arguments to properly build network graph
        # todo: currently, using the default setting
        self.spec = ftfy_model_fn(
            sources=self.norm_src, targets=self.norm_tar,
            src_cell_size=self.src_cell_size, mode='TEST'
        )

        config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            intra_op_parallelism_threads=8,
            inter_op_parallelism_threads=0
        )
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        # initialize
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        self.saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt is None:
            raise ValueError('Cannot find a checkpoint: %s' % model_path)
        model_checkpoint_path = ckpt.model_checkpoint_path
        if epoch is not None:
            all_model_checkpoint_paths = ckpt.all_model_checkpoint_paths
            if epoch < len(all_model_checkpoint_paths):
                model_checkpoint_path = all_model_checkpoint_paths[epoch]
        self.saver.restore(self.sess, model_checkpoint_path)


    def run(self, src, tar, top_k=0):

        feed_dict = dict(self.spec.test_feed_dict)
        feed_dict[self.src] = src
        feed_dict[self.tar] = tar

        confidence, bboxes = self.sess.run(
            [self.spec.pred_confidence, self.spec.pred_bboxes], feed_dict)

        batch_size = confidence.shape[0]
        batch_ind = np.argsort(confidence)
        for i in range(batch_size):
            ind = batch_ind[i]
            ind = ind[::-1]
            confidence[i] = confidence[i, ind]
            bboxes[i] = bboxes[i, ind]

        if top_k > 0:
            confidence = confidence[:, :top_k]
            bboxes = bboxes[:, :top_k]

        return confidence, bboxes


    def get_input(self, images, is_src:bool, batch_size=48):
        input_shape = images.shape
        assert len(input_shape) == 4, 'Required to have 4-dimension input.'
        assert input_shape[3] == self.n_channels, 'Unmatched number of channels, must %d' % self.n_channels

        raw_inputs = []

        n_images = input_shape[0]
        batch_iters = (n_images + batch_size) // batch_size
        for i_batch in range(batch_iters):
            start = i_batch*batch_size
            end = min(n_images, (i_batch+1)*batch_size)

            feed_dict = dict(self.spec.test_feed_dict)
            if is_src:
                feed_dict[self.src] = images[start: end]
                raw_input = self.sess.run(self.norm_src, feed_dict=feed_dict)
            else:
                feed_dict[self.tar] = images[start: end]
                raw_input = self.sess.run(self.norm_tar, feed_dict=feed_dict)

            raw_inputs.append(raw_input)

            if end == n_images:
                break

        raw_inputs = np.concatenate(raw_inputs, axis=0)
        return raw_inputs


