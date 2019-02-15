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
    def __init__(self, spec: TripletSpec, model_path=None, save_dir=None):
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
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)

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

    def run_match(self, dataset_initializer=None):
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
            self.spec.a_feat, # feature of input image
            self.spec.negatives # information (is_query, label_idx, patch_idx)
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
                #patch_idx = info[:, 2, 0, 0].astype(np.int)
                
                all_feat.append(feat)
                all_label_ind.append(label_idx)
                all_is_query.append(is_query)
                #all_patch_ind.append(all_patch_ind)
                
        except tf.errors.OutOfRangeError:
            tf.logging.info('Exhausted dataset for run_retrieval: %d' % count)

        all_feat = np.concatenate(all_feat, axis=0)
        all_label_ind = np.concatenate(all_label_ind, axis=0)
        all_is_query = np.concatenate(all_is_query, axis=0)
        #all_patch_ind = np.concatenate(all_patch_ind, axis=0)
        return TripletOutputSpec(all_feat, 
                                 index=all_label_ind, 
                                 scores=all_is_query)
            
    def save(self, name, global_step=None):
        if self.saver is not None:
            save_path = os.path.join(self.save_dir, name)
            save_path = self.saver.save(self.sess, save_path, global_step=global_step)
            tf.logging.info('Save checkpoint @ {}'.format(save_path))
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            

