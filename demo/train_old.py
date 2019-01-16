import tensorflow as tf
import numpy as np

import network.dataset.sem_dataset as sem
#from network.dataset.sem_input_fn import input_fn, get_image_list
from network.dataset.sem_data_sampler import DataSamplerIoU
from network.model.ftfy_cnn import Net as model
from network.loss.triplet import batch_offline_triplet_loss
from network.train import Estimator
from utils.eval import Evaluator
from utils.viz import RetrievalPlot
from utils.utils import downsample

def load_dataset(dataset_dirs, train_ratio=0.8):
    filenames = []
    patch_xy = []
    for dataset_dir in dataset_dirs:
        dataset = np.load(dataset_dir).item()
        for f, xy_list in dataset.items():
            filenames.append(f)
            patch_xy.append(xy_list[0])

    ind = np.arange(len(filenames), dtype=np.int32)
    np.random.shuffle(ind)
    n_train = int(len(filenames) * train_ratio)

    train_filenames = [filenames[i] for i in ind[:n_train]]
    train_patch_xy = [patch_xy[i] for i in ind[:n_train]]#patch_xy[ind[:n_train]]
    test_filenames = [filenames[i] for i in ind[n_train:]]#filenames[ind[n_train:]]
    test_patch_xy = [patch_xy[i] for i in ind[n_train:]]#patch_xy[ind[n_train:]]
    return (train_filenames, train_patch_xy), (test_filenames, test_patch_xy)

# set seed for reproduction
np.random.seed(2019)
tf.set_random_seed(2019)

# train/evalutation parameters
data_dir = '/home/sungsooha/Desktop/Data/ftfy/data_hxnsem_selected'
dataset_dirs = [
    '../data/sem_dataset_208_a.npy',
    '../data/sem_dataset_208_b.npy'
]
patch_size = (208, 208)
n_channels = 1
n_img_per_iter = 10
n_crops_per_img = 10
batch_size = 16

# augmentation
down_factors = [2, 4, 8]

# image retrieval test parameters
n_tests = 10 # i.e. number of images for the test
n_queries_per_img = batch_size * 3
top_k = 5
test_iou_threshold = 0.7
test_down_factors = [2, 4, 8]

# plot manager parameters
n_examples = 3
n_queries = 3
repeat_plot = True

# triplet model parameters
n_feats = 32
margin = 0.3

# optimizer parameters and training parameters
learning_rate = 0.0001
n_epoch = 500
log_every = 50 # log every 50 iter per epoch

# parsing image files for train/evaluation
train_dataset, test_dataset = load_dataset(dataset_dirs, train_ratio=0.8)
print(len(train_dataset[0]), len(test_dataset[0]))
#tr_list, t_list = sem.split_train_test(sem.filelist_by_zoomfactor(data_dir), train_ratio=.8)

# dataset_wargs = {
#     'patch_size': patch_size,
#     'n_channels': n_channels,
#     'iou_thresholds': iou_thresholds,
#     'n_img_per_iter': n_img_per_iter,
#     'n_crops_per_img': n_crops_per_img,
#     'batch_size': batch_size
# }
# aug_wargs = {
#     'down_factors': down_factors
# }

# create estimator object
# tf.reset_default_graph() # reset graph
# est = Estimator(
#     data_dir=data_dir, train_list=tr_list,
#     input_fn=input_fn, dataset_warg=dataset_wargs, aug_warg=aug_wargs,
#     triplet_model_fn=model(n_feats=n_feats),
#     triplet_model_name='ftfy_triplet',
#     triplet_loss_fn=batch_offline_triplet_loss,
#     loss_wargs=dict(margin=margin),
#     optimizer_fn=tf.train.AdamOptimizer,
#     optimizer_wargs=dict(learning_rate=learning_rate)
# )

# evaluator
# eval = Evaluator(get_image_list(data_dir, t_list), patch_size)

# plot manager
# plot = RetrievalPlot(
#     n_examples=n_examples,
#     n_queries=n_queries,
#     top_k=top_k,
#     n_scalars=1,
#     n_lines=[len(test_down_factors) + 1],
#     colors=['yellow', 'red', 'orange', 'magenta'],
#     image_size=(850, 1280),
#     patch_size=patch_size
# )

# training/evaluate
# for epoch in range(n_epoch):
#
#     print('-' * 80)
#     print('Validation Epoch: {:d}'.format(epoch))
#     avg_acc = np.zeros((len(test_down_factors)+1), dtype=np.float32)
#     eval.reset()
#     for i_test in range(n_tests):
#         eval.reset_db()
#         val_image = eval.get_image()
#         # build database
#         features, rows, cols, _ = est.get_features(
#             image=val_image,
#             patch_size=patch_size,
#             stride=(patch_size[0]//8, patch_size[1]//8),
#             max_n_feats=None,
#             random_sampling=False,
#             augmentor=None
#         )
#         eval.add_item(features, rows, cols, is_db=True, q_img=None)
#
#         # collect query examples
#         for i_f, f in enumerate(test_down_factors):
#             augmentor = lambda x: x if f == 1 else downsample(x, patch_size, (f, f, 1))
#             eval.reset_q()
#             features, rows, cols, imgs = est.get_features(
#                 image=val_image,
#                 patch_size=patch_size,
#                 stride=(patch_size[0]//8, patch_size[1]//8),
#                 max_n_feats=n_queries_per_img,
#                 random_sampling=True,
#                 augmentor=augmentor
#             )
#             eval.add_item(features, rows, cols, is_db=False, q_img=imgs)
#
#             eval.fit()
#             acc = eval.get_accuracy(top_k=top_k, iou_threshold=test_iou_threshold)
#             avg_acc[i_f] += acc
#             avg_acc[-1] += acc
#
#             # update plot
#             # todo: optimize for image rendering as it will be same
#             if not repeat_plot and i_test > n_examples:
#                 continue
#
#             plot.update(
#                 i_test % n_examples,
#                 np.squeeze(val_image),
#                 eval.get_top_k_pos(top_k),
#                 patch_size
#             )
#
#     avg_acc = avg_acc / n_tests
#     avg_acc[-1] = avg_acc[-1] / len(test_down_factors)
#     print(avg_acc)
#     plot.update_scalar([avg_acc])
#
#     print('-' * 80)
#     print('Training Epoch: {:d}'.format(epoch))
#     print("avg. loss: ", est.run_train(log_every=log_every))
#
#
# plot.hold()
