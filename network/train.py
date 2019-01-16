import tensorflow as tf
import numpy as np

class TripletSpec(object):
    def __init__(
            self,
            anchors, positives, negatives,
            a_feat, p_feat, n_feat,
            train_op = None,
            triplet_loss = None,
            regularization_loss = None,
            train_feed_dict = None,
            test_feed_dict = None
    ):
        self.anchors = anchors
        self.positives = positives
        self.negatives = negatives
        self.a_feat = a_feat
        self.p_feat = p_feat
        self.n_feat = n_feat

        # for training
        self.train_op = train_op
        self.triplet_loss = triplet_loss
        self.regularization_loss = regularization_loss

        # feed_dict
        self.train_feed_dict = train_feed_dict
        self.test_feed_dict = test_feed_dict

class TripletOutputSpec(object):
    def __init__(self, features, x0, y0, images):
        self.features = features
        self.x0 = x0
        self.y0 = y0
        self.images = images

class TripletEstimator(object):
    '''DNN estimator for triplet network'''
    def __init__(self, spec: TripletSpec, restore=None):
        self.spec = spec

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

        if restore is not None:
            raise NotImplementedError("restore the model from the latest checkpoint")

    def train(self, dataset_initializer=None, log_every=0):
        if self.spec.train_op is None or self.spec.triplet_loss is None:
            raise ValueError("train_op and triplet_loss are undefined!")

        train_step = [self.spec.train_op, self.spec.triplet_loss]
        if self.spec.regularization_loss is not None:
            train_step.append(self.spec.regularization_loss)

        step = 0
        if dataset_initializer is not None:
            # initialize tensorflow data pipeline
            self.sess.run(dataset_initializer)

        try:
            while True:
                outputs = self.sess.run(train_step, feed_dict=self.spec.train_feed_dict)
                step += 1
                if log_every > 0 and step % log_every == 0:
                    tf.logging.info(
                        "step: {:d}, triplet_loss: {:.6f}, regularization_loss: {:.6f}".format(
                            step,
                            outputs[1],
                            outputs[2] if len(outputs) > 2 else 0
                        ))
        except tf.errors.OutOfRangeError:
            tf.logging.info("Exhausted all data in the dataset ({:d})!".format(step))

    def run(self, dataset_initializer=None, collect_image=False):
        fetches = [
            self.spec.a_feat,   # feature of input image
            self.spec.positives,# sampling location x0 and y0 (trick)
            self.spec.negatives,
        ]
        if collect_image:
            fetches.append(self.spec.anchors)

        if dataset_initializer is not None:
            # initialize tensorflow data pipeline
            self.sess.run(dataset_initializer)

        features, x0, y0, images = [], [], [], []
        try:
            while True:
                outputs = self.sess.run(fetches, feed_dict=self.spec.test_feed_dict)
                features.append(outputs[0])

                _x0 = outputs[1]
                _x0 = _x0.astype(np.int32)
                _x0 = _x0[:, 0, 0, 0]
                x0.append(_x0)

                _y0 = outputs[2]
                _y0 = _y0.astype(np.int32)
                _y0 = _y0[:, 0, 0, 0]
                y0.append(_y0)

                if collect_image:
                    images.append(outputs[3])
        except tf.errors.OutOfRangeError:
            pass

        features = np.concatenate(features, axis=0)
        x0 = np.concatenate(x0, axis=0)
        y0 = np.concatenate(y0, axis=0)
        if collect_image:
            images = np.concatenate(images, axis=0)

        return TripletOutputSpec(features, x0, y0, images)



