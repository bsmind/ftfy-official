import os
import logging
import numpy as np

def set_logger(log_dir):
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

class Param(object):
    '''Parameters for triplet'''
    def __init__(self, project_name='triplet-net', log_dir=None):
        if log_dir is not None:
            set_logger(log_dir)

        # --------------------------------------------------------------------
        # Basic configuration
        # --------------------------------------------------------------------
        # overall scope in tensorflow graph
        self.project_name = project_name
        # path for logging
        self.log_dir = log_dir
        # checkpoint interval
        self.save_every = 5
        # full path to an existing model (checkpoint)
        self.model_path = None

        # --------------------------------------------------------------------
        # Data configuration
        # --------------------------------------------------------------------
        # base directory for data
        self.data_dir = '/home/sungsooha/Desktop/Data/ftfy/austin'
        # directory name for train datasets under `data_dir`
        self.train_datasets = 'campus_patch'
        # directory name for test datasets under `data_dir`
        self.test_datasets = 'human_patch'
        # input image (patch) size
        self.patch_size = (128, 128)
        # the number of channels
        self.n_channels = 1
        # base patch size (patch size in database, will be deprecated)
        self.base_patch_size = (128,128)
        # the number of patch rows in a patch set image
        self.patches_per_row = 13
        # the number of patch cols in a patch row
        self.patches_per_col = 6

        # --------------------------------------------------------------------
        # Model configuration
        # --------------------------------------------------------------------
        # CNN model name
        self.cnn_name = 'ftfy'           # ['simple', 'ftfy', 'spread']
        # loss function name
        self.loss_name = 'spreadout'     # ['triplet', 'spreadout']
        # optimizer name
        self.optimizer_name = 'Momentum' # ['Adam', 'Momentum']

        # --------------------------------------------------------------------
        # Train configuration
        # --------------------------------------------------------------------
        # batch size
        self.batch_size = 16
        # the number of triplet samples used for training
        self.n_triplet_samples = 1000000
        # feature dimension
        self.n_features = 128
        # triplet loss margin
        self.margin = None # for `triplet` loss, None for soft-margin, 0.3 for hard-margin
        # the number of epochs
        self.n_epoch = 500
        # l2 regularization
        self.use_regularization = False # set True to apply
        # initial learning rate
        self.learning_rate = 0.1
        # decay step
        self.decay_steps = 10000 # used in `Momentum` for learning rate control
        # decay rate
        self.decay_rate = 0.96 # used in `Momentum` for learning rate control
        # momentum
        self.momentum = 0.9 # for `Momentum`
        # log interval during training
        self.train_log_every = 50

        # --------------------------------------------------------------------
        # Test configuration
        # --------------------------------------------------------------------
        # the number of matched pairs for testing
        self.n_match_pairs = 20000





        '''FTFY'''
        # self.cellsz = (16,16)
        # self.src_size=(256, 256)
        # self.tar_size=(128, 128)
        # self.n_parameters = 5

class FTFYParam(object):
    '''Parameters for ftfy'''
    def __init__(self, ftfy_scope='ftfy', feat_scope='triplet-net', log_dir=None):
        if log_dir is not None:
            set_logger(log_dir)

        # --------------------------------------------------------------------
        # Basic configuration
        # --------------------------------------------------------------------
        # ftfy network scope in tensorflow graph
        self.ftfy_scope = ftfy_scope
        # feature extractor scope in tensorflow graph
        self.feat_scope = feat_scope
        # path for logging
        self.log_dir = log_dir
        # checkpoint interval
        self.save_every = 5
        # full path to an existing model (checkpoint)
        self.is_ftfy_model = False # False if `model_path` is only for feature extractor
        self.model_path = None

        # --------------------------------------------------------------------
        # Data configuration
        # --------------------------------------------------------------------
        # base directory for data
        self.data_dir = '/home/sungsooha/Desktop/Data/ftfy/sem/train'
        # directory name for train datasets under `data_dir`
        self.train_datasets = 'sem'
        # directory name for test datasets under `data_dir`
        #self.test_datasets = 'human_patch'
        # input image (patch) size
        self.src_size = (256, 256) # for source image
        self.tar_size = (128, 128) # for target image
        # the number of channels
        self.n_channels = 1
        # the number of patches in a patch set image
        self.src_dir = 'sources' # ['sources', 'sources_square']
        self.src_ext = 'bmp'
        self.src_patches_per_row = 10 # for source
        self.src_patches_per_col = 10 # for source
        self.tar_dir = 'patches'
        self.tar_ext = 'bmp'
        self.tar_patches_per_row = 10 # for target
        self.tar_patches_per_col = 10 # for target

        # --------------------------------------------------------------------
        # Model configuration
        # --------------------------------------------------------------------
        # CNN model name
        self.cnn_name = 'ftfy'           # ['simple', 'ftfy', 'spread']
        # loss function name
        self.loss_name = 'rms'
        self.obj_scale = 1.0
        self.noobj_scale = 0.5
        self.coord_scale = 5.0
        # optimizer name
        self.optimizer_name = 'Momentum' # ['Adam', 'Momentum']

        # --------------------------------------------------------------------
        # Train configuration
        # --------------------------------------------------------------------
        # cell size (source and target feature dimension)
        self.src_cellsz = (16,16) # 256//16
        self.tar_cellsz = (8,8)   # 128//16
        # number parameters to define a bbox
        self.n_parameters = 5 # (confidence, cx, cy, w, h)
        # number of bbox estimators per a cell (source feature cell)
        self.n_bbox_estimators = 2
        # batch size
        self.batch_size = 16
        # the number of epochs
        self.n_epoch = 500
        # l2 regularization
        self.use_regularization = False # set True to apply
        # initial learning rate
        self.learning_rate = 0.1
        # decay step
        self.decay_steps = 10000 # used in `Momentum` for learning rate control
        # decay rate
        self.decay_rate = 0.96 # used in `Momentum` for learning rate control
        # momentum
        self.momentum = 0.9 # for `Momentum`
        # log interval during training
        self.train_log_every = 50

    def get_optimizer_kwargs(self):
        if self.optimizer_name in ['Momentum', 'Grad']:
            return dict(
                learning_rate=self.learning_rate,
                decay_steps=self.decay_steps,
                decay_rate=self.decay_rate,
                momentum=self.momentum
            )
        elif self.optimizer_name == 'Adam':
            return dict(learning_rate=self.learning_rate)
        else:
            raise ValueError("Unknown optimizer %s" % self.optimizer_name)

    def get_model_kwargs(self, feat_trainable=False, feat_shared_batch_layers=True):
        """

        Args:
            feat_trainable: (bool)
                True: for end-to-end training
                False: will use pre-trained model for feature extractor
                       requires to provide proper pre-triained model

            feat_shared_batch_layers: (bool)
                True: used shared batch normalization layers
                False: this is for cross-domain input images where input images may likely have
                    different characteristics.

        Returns: (dict) model function arguments

        """
        return dict(
            # feature extractor
            feat_name=self.cnn_name,
            feat_trainable=feat_trainable,
            feat_shared_batch_layers=feat_shared_batch_layers,
            feat_scope=self.feat_scope,
            # ftfy
            ftfy_ver='v0', # for future usages
            ftfy_scope=self.ftfy_scope,
            src_cell_size=self.src_cellsz,
            tar_cell_size=self.tar_cellsz,
            n_bbox_estimators=self.n_bbox_estimators,
            n_parameters=self.n_parameters,
            loss_name='rms', # for future usages
            obj_scale=self.obj_scale,
            noobj_scale=self.noobj_scale,
            coord_scale=self.coord_scale,
            use_regularization_loss=self.use_regularization,
            optimizer_name=self.optimizer_name,
            optimizer_kwargs=self.get_optimizer_kwargs()
        )

    def get_ckpt_kwargs(self, epoch=None):
        if self.is_ftfy_model:
            return dict(
                ftfy_model_path=self.model_path,
                ftfy_epoch=epoch,
                save_dir=self.log_dir
            )
        else:
            return dict(
                feat_model_path=self.model_path,
                feat_epoch=epoch,
                feat_scope=self.feat_scope,
                save_dir=self.log_dir
            )



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