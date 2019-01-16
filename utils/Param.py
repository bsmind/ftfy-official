import os
import logging
import numpy as np

class Param(object):
    def __init__(self, log_dir=None):
        if log_dir is not None:
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
        self.log_dir = log_dir

        self.data_dir = '/home/sungsooha/Desktop/Data/ftfy/data_hxnsem_selected'
        self.train_datasets = '../data/train_sem_dataset_208.npy'
        self.test_datasets = '../data/test_sem_dataset_208.npy'

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
        self.n_iter = 10
        self.batch_size = 16

        '''test dataset
        '''
        self.stride = (208 // 4, 208 // 4)
        self.top_k = 5
        self.iou_threshold = 0.7

        '''triplet network parameters
        '''
        self.n_features = 32
        self.margin = 0.3

        '''train parameters
        '''
        self.n_epoch = 500
        self.use_regularization = True
        self.learning_rate = 0.0001
        self.train_log_every = 50

        '''augmentation
        '''
        self.down_factors = np.array([2, 4, 8])