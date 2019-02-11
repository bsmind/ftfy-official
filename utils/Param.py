import os
import logging
import numpy as np

class Param(object):
    def __init__(self, project_name='triplet-net', log_dir=None):
        if log_dir is not None:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            # get TF logger
            log = logging.getLogger('tensorflow')
            log.setLevel(logging.INFO)

            # create formatter and add it to the handlers
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

            # create file handler which logs even debug messages
            fh = logging.FileHandler(os.path.join(log_dir, 'tensorflow.log'))
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            log.addHandler(fh)

        self.project_name = project_name
        self.log_dir = log_dir
        self.save_every = 5
        self.model_path = None

        # data
        self.data_dir = '/home/sungsooha/Desktop/Data/ftfy/data_hxnsem_selected'
        self.train_datasets = '../data/train_sem_dataset_208.npy'
        self.test_datasets = '../data/test_sem_dataset_208.npy'

        self.cnn_name = 'ftfy'
        self.loss_name = 'triplet'
        self.optimizer_name = 'Adam'

        '''Image size
        patch_size: (height, width) input image size of DNN
        n_channels: # channels (1: gray scale, 3: RGB)
        '''
        self.patch_size = (208, 208)
        self.n_channels = 1

        '''training dataset
        n_img_per_iter: # images used by data_sampler per iteration
        n_crops_per_img: # crops per an image
        n_iter: # iterations of data_sampler
        batch_size: batch size

        Each iteration, data_sampler generator (N = n_img_per_iter * n_crops_per_img)
        per iteration. It loops n_iter times. Finally, we will have (N * n_iter) triplet samples 
        and build a dataset using those samples. Each call of the dataset will return batch_size
        number of samples, at most. 
        '''
        self.n_img_per_iter = 10
        self.n_crops_per_img = 10
        self.n_iter = 30
        self.batch_size = 16

        self.base_patch_size = (64,64)
        self.patches_per_row = 16
        self.patches_per_col = 16
        self.n_triplet_samples = 1000000
        self.n_match_pairs = 20000

        '''test dataset
        '''
        self.stride = (208 // 16, 208 // 16)
        self.top_k = 5
        self.iou_threshold = 0.7

        '''triplet network parameters
        '''
        self.n_features = 128 # 32 (seems better with 128 or higher?)
        #self.margin = 0.3 # to use hard-margin
        self.margin = None # to use soft-margin

        '''train parameters
        '''
        self.n_epoch = 500
        self.use_regularization = False # l2-norm
        self.learning_rate = 0.0001
        self.train_log_every = 50

        '''augmentation
        '''
        self.down_factors = np.array([1, 2, 4, 8])

def get_default_param(mode, log_dir):
    p = Param(log_dir=log_dir)

    if mode == 'BNL':
        return p
    elif mode == 'UBC':
        p.data_dir = '/home/sungsooha/Desktop/Data/ftfy/descriptor'
        p.train_datasets = 'notredame'
        p.test_datasets = 'liberty'
        p.patch_size = (64, 64)
        p.n_channels = 1
        p.batch_size = 128
        p.n_features = 128
        p.margin = None
        p.n_epoch = 500
        p.use_regularization = False
        p.learning_rate = 0.01
        p.train_log_every = 50
        #p.cnn_name = 'spread'
        p.cnn_name = 'ftfy'
        #p.loss_name = 'triplet'#''spreadout'
        p.loss_name = 'spreadout'
        p.optimizer_name = 'Momentum'
        return p
    elif mode == 'AUSTIN':
        p.data_dir = '/home/sungsooha/Desktop/Data/ftfy/austin'
        p.train_datasets = 'campus_patch'
        p.test_datasets = 'scene_patch'
        p.patch_size = (128, 128)
        p.n_channels = 1
        p.base_patch_size = (128, 128)
        p.patches_per_row = 13
        p.patches_per_col = 6
        p.batch_size = 16 # 128
        p.n_features = 128
        p.margin = None
        p.n_epoch = 100
        p.use_regularization = False
        p.learning_rate = 0.01
        p.train_log_every = 100
        #p.cnn_name = 'spread'
        p.cnn_name = 'ftfy'
        #p.loss_name = 'triplet'#''spreadout'
        p.loss_name = 'spreadout'
        p.optimizer_name = 'Momentum'
        return p
    else:
        return p