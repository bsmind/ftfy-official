import tensorflow as tf
import numpy as np

class Estimator(object):
    '''train and evaluate network'''
    def __init__(
            self, data_dir, train_list, input_fn, dataset_warg, aug_warg,
            triplet_model_fn, triplet_model_name,
            triplet_loss_fn,
            loss_wargs,
            optimizer_fn, optimizer_wargs
    ):
        self.data_dir = data_dir

        # build input pipeline
        self._build_data_pipeline(train_list, input_fn, dataset_warg, aug_warg)

        # placeholder & variables
        self.is_training = tf.placeholder(tf.bool, (), "is_training")

        # build network
        self._build_network(triplet_model_fn, triplet_model_name)

        # losses
        self.triplet_loss = self._get_triplet_loss(triplet_loss_fn, loss_wargs)
        self.l2_reg_loss = tf.losses.get_regularization_loss()
        self.loss = self.triplet_loss + self.l2_reg_loss

        # set train pipeline
        self.train_op = self._get_train_op(optimizer_fn, optimizer_wargs)

        # create session
        config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            intra_op_parallelism_threads=8,
            inter_op_parallelism_threads=0
        )
        config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())


    def _build_data_pipeline(self, train_list, input_fn, dataset_wargs, aug_wargs):
        if train_list is None or len(train_list) == 0:
            raise ValueError("train list must include more than 1 elements.")

        self.train_dataset = None
        self.test_dataset = None
        with tf.device('/cpu:0'), tf.name_scope('input'):
            self.train_dataset, _ = input_fn(
                data_dir=self.data_dir,
                params=dataset_wargs,
                aug_kwarg=aug_wargs,
                file_list=train_list,
                is_training=True
            )
            self.test_dataset, self.test_data_sampler = input_fn(
                self.data_dir,
                dataset_wargs,
                is_training=False
            )

            data_iterator = tf.data.Iterator.from_structure(
                self.train_dataset.output_types,
                self.train_dataset.output_shapes
            )
            self.train_dataset_init = data_iterator.make_initializer(self.train_dataset)
            self.test_dataset_init = data_iterator.make_initializer(self.test_dataset)

            self.batch_data = data_iterator.get_next()

    def _build_network(self, model_fn, model_name):
        anchors, positives, negatives = self.batch_data
        with tf.name_scope(model_name):
            self.a_feat = model_fn(anchors, self.is_training, bn_prefix="a")
            self.p_feat = model_fn(positives, self.is_training, bn_prefix="p")
            self.n_feat = model_fn(negatives, self.is_training, bn_prefix="n")

    def _get_triplet_loss(self, loss_fn, loss_wargs):
        margin = loss_wargs.get('margin', 0.3)
        return loss_fn(self.a_feat, self.p_feat, self.n_feat, margin)

    def _get_train_op(self, optimizer_fn, optimizer_wargs):
        global_step = optimizer_wargs.pop('global_step', None)
        if global_step is None:
            global_step = tf.train.create_global_step()
        optimizer = optimizer_fn(**optimizer_wargs)

        grads_and_vars = optimizer.compute_gradients(self.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        return tf.group(train_op, *update_ops)

    def run_train(self, log_every=50):
        step = 0
        avg_loss = 0
        try:
            self.sess.run(self.train_dataset_init)
            while True:
                outputs = self.sess.run(
                    [self.train_op, self.triplet_loss, self.l2_reg_loss],
                    {self.is_training: True}
                )
                step += 1
                avg_loss += outputs[1]
                if step % log_every == 0:
                    print("step {:d}, triplet loss: {:.6f}, l2: {:.6f}".format(
                        step, outputs[1], outputs[2]
                    ))
        except tf.errors.OutOfRangeError:
            pass

        return avg_loss / step

    def get_features(self,
                     image, patch_size, stride,
                     max_n_feats=None,
                     random_sampling=False,
                     augmentor=None
    ):
        self.test_data_sampler.reset(
            image,
            patch_size=patch_size,
            stride=stride,
            random_sampling=random_sampling,
            augmentor=augmentor
        )
        self.sess.run(self.test_dataset_init)

        features = []
        rows = []
        cols = []
        patches = []

        n_feats = 0
        try:
            while True:
                feat, img, r, c = self.sess.run(
                    [self.a_feat, *self.batch_data],
                    feed_dict={self.is_training: False}
                )
                r = r.astype(np.int32)
                r = r[:, 0, 0, 0]
                c = c.astype(np.int32)
                c = c[:, 0, 0, 0]

                n_feats += len(feat)
                features.append(feat)
                rows.append(r)
                cols.append(c)
                patches.append(img)
                if max_n_feats is not None and n_feats >= max_n_feats:
                    break
        except tf.errors.OutOfRangeError:
            pass

        features = np.concatenate(features, axis=0)
        rows = np.concatenate(rows, axis=0)
        cols = np.concatenate(cols, axis=0)
        patches = np.concatenate(patches, axis=0)

        return features, rows, cols, patches
