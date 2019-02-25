import os
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
            test_feed_dict = None,
            global_step = None,
            net=None
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
        self.global_step = global_step

        # feed_dict
        self.train_feed_dict = train_feed_dict
        self.test_feed_dict = test_feed_dict

        # for the accesss to intermediate tensors
        self.net = net

class TripletOutputSpec(object):
    def __init__(
            self,
            features,
            x0=None, y0=None, images=None,
            index=None, labels=None, scores=None
    ):
        self.features = features
        self.x0 = x0
        self.y0 = y0
        self.images = images
        self.index = index
        self.labels = labels
        self.scores = scores

    def to_dict(self):
        return {
            'features': self.features,
            'x0': self.x0,
            'y0': self.y0,
            'images': self.images
        }

class TripletEstimator(object):
    '''DNN estimator for triplet network'''
    def __init__(self, spec: TripletSpec, model_path=None, epoch=None, save_dir=None):
        self.spec = spec

        config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            intra_op_parallelism_threads=8,
            inter_op_parallelism_threads=0
        )
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        self.saver = tf.train.Saver(max_to_keep=100)
        if save_dir is not None:
            self.save_dir = save_dir

        # initialize
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        if model_path is not None:
            ckpt = tf.train.get_checkpoint_state(model_path)
            if ckpt is None:
                raise ValueError('Cannot find a checkpoint: %s' % model_path)
            model_checkpoint_path = ckpt.model_checkpoint_path
            if epoch is not None:
                all_model_checkpoint_paths = ckpt.all_model_checkpoint_path
                if 0 <= epoch and epoch < len(all_model_checkpoint_paths):
                    model_checkpoint_path = all_model_checkpoint_paths[epoch]
            self.saver.restore(self.sess, model_checkpoint_path)

    def train(self, dataset_initializer=None, log_every=0):
        if self.spec.train_op is None or self.spec.triplet_loss is None:
            raise ValueError("train_op and triplet_loss are undefined!")

        train_step = [self.spec.train_op, self.spec.triplet_loss]
        if self.spec.regularization_loss is not None:
            train_step.append(self.spec.regularization_loss)

        step = 0
        avg_triplet_loss = 0
        if dataset_initializer is not None:
            # initialize tensorflow data pipeline
            self.sess.run(dataset_initializer)

        try:
            while True:
                outputs = self.sess.run(train_step, feed_dict=self.spec.train_feed_dict)
                avg_triplet_loss += outputs[1]
                step += 1
                if log_every > 0 and step % log_every == 0:
                    tf.logging.info(
                        "step: {:d}, triplet_loss: {:.6f}, regularization_loss: {:.6f}".format(
                            step,
                            outputs[1],
                            outputs[2] if len(outputs) > 2 else 0
                        ))
        except tf.errors.OutOfRangeError:
            avg_triplet_loss /= step
            tf.logging.info("Avg. triplet loss: {:.6f}".format(avg_triplet_loss))
            tf.logging.info("Exhausted all data in the dataset ({:d})!".format(step))

        return avg_triplet_loss

    def run(self, dataset_initializer=None, collect_image=False):
        """
        get feature vectors
            a_feat: feature vector
            positives: (trick) x0 location
            negatives: (trick) y0 location
            anchors: input image, if collect_image is True

        Args:
            dataset_initializer:
            collect_image:

        Returns:

        """
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

    def run_match(self, dataset_initializer=None, info_parser=None):
        """
        get feature vectors for matching evaluation...
            a_feat: feature vector for image 1
            p_feat: feature vector for image 2
            negatives: information of indices of image 1 and 2, and is_match
                is_math is 1 if two images are matched; otherwise 0
        """
        fetches = [
            self.spec.a_feat,  # feature of input image 1,
            self.spec.p_feat,  # feature of input image 2,
            self.spec.negatives, # information (idx_1, idx_2, is_match)
        ]

        if dataset_initializer is not None:
            # initialize tensorflow data pipeline
            self.sess.run(dataset_initializer)

        all_feat_1, all_feat_2, all_idx_1, all_idx_2, all_is_match = [], [], [], [], []
        all_dist = []
        count = 0
        try:
            while True:
                feat_1, feat_2, info = self.sess.run(fetches, feed_dict=self.spec.test_feed_dict)

                count += len(feat_1)

                # compute euclien distance
                dist = np.sqrt(np.sum((feat_1 - feat_2)**2, axis=1))

                # parsing info
                idx_1 = info[:, 0, 0, 0].astype(np.int)
                idx_2 = info[:, 1, 0, 0].astype(np.int)
                is_match = info[:, 2, 0, 0].astype(np.int)

                all_feat_1.append(feat_1)
                all_feat_2.append(feat_2)
                all_idx_1.append(idx_1)
                all_idx_2.append(idx_2)
                all_is_match.append(is_match)
                all_dist.append(dist)

        except tf.errors.OutOfRangeError:
            tf.logging.info('Exhausted dataset for run_match: %d' % count)
            pass

        features = np.concatenate(all_feat_1 + all_feat_2, axis=0)
        ind = np.concatenate(all_idx_1 + all_idx_2, axis=0)
        all_is_match = np.concatenate(all_is_match, axis=0)
        all_dist = np.concatenate(all_dist, axis=0)

        return TripletOutputSpec(
            features, index=ind, labels=all_is_match, scores=all_dist
        )

    def run_retrieval(self, dataset_initializer=None):
        fetches = [
            self.spec.a_feat,  # feature of input image
            self.spec.negatives  # information (is_query, label_idx, patch_idx)
        ]

        if dataset_initializer is not None:
            self.sess.run(dataset_initializer)

        all_feat, all_label_ind, all_is_query, all_patch_ind = [], [], [], []
        count = 0
        try:
            while True:
                feat, info = self.sess.run(fetches, feed_dict=self.spec.test_feed_dict)
                count += len(feat)

                # parsing info
                is_query = info[:, 0, 0, 0].astype(np.int)
                label_idx = info[:, 1, 0, 0].astype(np.int)
                # patch_idx = info[:, 2, 0, 0].astype(np.int)

                all_feat.append(feat)
                all_label_ind.append(label_idx)
                all_is_query.append(is_query)
                # all_patch_ind.append(all_patch_ind)

        except tf.errors.OutOfRangeError:
            tf.logging.info('Exhausted dataset for run_retrieval: %d' % count)

        all_feat = np.concatenate(all_feat, axis=0)
        all_label_ind = np.concatenate(all_label_ind, axis=0)
        all_is_query = np.concatenate(all_is_query, axis=0)

        # all_patch_ind = np.concatenate(all_patch_ind, axis=0)
        return TripletOutputSpec(all_feat,
                                 index=all_label_ind,
                                 scores=all_is_query)
      
    def save(self, name, global_step=None):
        if self.saver is not None:
            save_path = os.path.join(self.save_dir, name)
            save_path = self.saver.save(self.sess, save_path, global_step=global_step)
            tf.logging.info('Save checkpoint @ {}'.format(save_path))

class FTFYSpec(object):
    def __init__(
            self,
            sources, targets, labels, bboxes,
            src_feat, tar_feat, logits,
            pred_confidene, pred_bboxes,
            train_op=None,
            obj_loss=None, noobj_loss=None, coord_loss=None,
            regularization_loss=None, total_loss=None,
            global_step=None,
            train_feed_dict=None,
            test_feed_dict=None,
            feat_net=None, ftfy_net=None
    ):
        self.sources = sources
        self.targets = targets
        self.labels = labels
        self.bboxes = bboxes

        self.src_feat = src_feat
        self.tar_feat = tar_feat
        self.logits = logits

        self.pred_confidence = pred_confidene
        self.pred_bboxes = pred_bboxes

        self.train_op = train_op
        self.obj_loss = obj_loss
        self.noobj_loss = noobj_loss
        self.coord_loss = coord_loss
        self.regularization_loss = regularization_loss
        self.total_loss = total_loss
        self.global_step = global_step

        self.train_feed_dict = train_feed_dict
        self.test_feed_dict = test_feed_dict

        self.feat_net = feat_net
        self.ftfy_net = ftfy_net

class FTFYEstimator(object):
    def __init__(
            self,
            spec:FTFYSpec,

            feat_model_path=None, feat_scope=None, feat_epoch=None,
            ftfy_model_path=None, ftfy_scope=None, ftfy_epoch=None,
            save_dir=None
    ):
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

        if feat_model_path is not None:
            feat_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=feat_scope)
            saver = tf.train.Saver(feat_vars)
            ckpt = tf.train.get_checkpoint_state(feat_model_path)
            if ckpt is None:
                raise ValueError('Cannot find a checkpoint: %s' % feat_model_path)
            model_checkpoint_path = ckpt.model_checkpoint_path
            if feat_epoch is not None:
                all_model_checkpoint_paths = ckpt.all_model_checkpoint_paths
                if 0 <= feat_epoch and feat_epoch < len(all_model_checkpoint_paths):
                    model_checkpoint_path = all_model_checkpoint_paths[feat_epoch]
            saver.restore(self.sess, model_checkpoint_path)

        self.saver = tf.train.Saver(max_to_keep=100)
        if save_dir is not None:
            self.save_dir = save_dir
            
        # todo: resotre from ftfy checkpoint

    def train(self, dataset_initializer=None, log_every=0, n_max_steps=0):
        if self.spec.train_op is None or self.spec.total_loss is None:
            raise ValueError("train_op and total_loss are undefined!")

        train_step = [self.spec.train_op, self.spec.total_loss]
        loss_names = ['total loss']
        if self.spec.obj_loss is not None:
            train_step.append(self.spec.obj_loss)
            loss_names.append('obj loss')
        if self.spec.noobj_loss is not None:
            train_step.append(self.spec.noobj_loss)
            loss_names.append('noobj loss')
        if self.spec.coord_loss is not None:
            train_step.append(self.spec.coord_loss)
            loss_names.append('coord loss')
        if self.spec.regularization_loss is not None:
            train_step.append(self.spec.regularization_loss)
            loss_names.append('L2 reg')

        step = 0
        avg_losses = np.zeros(len(loss_names), dtype=np.float32)
        if dataset_initializer is not None:
            self.sess.run(dataset_initializer)

        try:
            while True:
                outputs = self.sess.run(train_step, feed_dict=self.spec.train_feed_dict)
                for i in range(len(loss_names)):
                    avg_losses[i] += outputs[i+1]

                step += 1
                if log_every > 0 and step % log_every == 0:
                    tf.logging.info("step: {:d}".format(step))
                    for name, value in zip(loss_names, avg_losses):
                        tf.logging.info("Avg. {:s}: {:.6f}".format(name, value/step))

                if n_max_steps > 0 and step >= n_max_steps:
                    tf.logging.info('Terminated with step: {:d}'.format(step))
                    break

        except tf.errors.OutOfRangeError:
            tf.logging.info("Exhausted all data in the dataset ({:d})!".format(step))

        avg_losses /= step
        tf.logging.info("Final avg. losses:")
        for name, value in zip(loss_names, avg_losses):
            tf.logging.info("Avg. {:s}: {:.6f}".format(name, value))

        return avg_losses[0]

    def run(self, dataset_initializer=None, top_k=0, n_max_test=1000):

        fetches = [self.spec.pred_confidence, self.spec.pred_bboxes, self.spec.bboxes]

        if dataset_initializer is not None:
            self.sess.run(dataset_initializer)

        step = 0
        all_confidences, all_pred_bboxes, all_bboxes = [], [], []

        try:
            while True:
                confidence, pred_bboxes, bboxes \
                    = self.sess.run(fetches, feed_dict=self.spec.test_feed_dict)

                batch_size = confidence.shape[0]

                batch_ind = np.argsort(confidence)
                for i in range(batch_size):
                    ind = batch_ind[i]
                    ind = ind[::-1]
                    confidence[i] = confidence[i, ind]
                    pred_bboxes[i] = pred_bboxes[i, ind, :]

                if top_k > 0:
                    confidence = confidence[:, :top_k]
                    pred_bboxes = pred_bboxes[:, :top_k]

                all_confidences.append(confidence)
                all_pred_bboxes.append(pred_bboxes)
                all_bboxes.append(bboxes[..., 1:])
                step += batch_size

                if step >= n_max_test:
                    tf.logging.info("Terminate with {:d}".format(step))
                    break

        except tf.errors.OutOfRangeError:
            tf.logging.info("Exhausted all data in the dataset ({:d})!".format(step))

        all_confidences = np.concatenate(all_confidences, axis=0)
        all_pred_bboxes = np.concatenate(all_pred_bboxes, axis=0)
        all_bboxes = np.concatenate(all_bboxes, axis=0)

        return all_confidences, all_pred_bboxes, all_bboxes
            
    def save(self, name, global_step=None):
        if self.saver is not None:
            save_path = os.path.join(self.save_dir, name)
            save_path = self.saver.save(self.sess, save_path, global_step=global_step)
            tf.logging.info('Save checkpoint @ {}'.format(save_path))
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            

