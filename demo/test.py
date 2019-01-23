import tensorflow as tf
import numpy as np

from utils.Param import Param
from utils.eval import Evaluator
from utils.viz import ScalarPlot
from network.model_fn import triplet_model_fn
from network.dataset.sem_data_sampler import train_input_fn, test_input_fn, load_full_dataset
from network.train import TripletEstimator


# set seed for reproduction
np.random.seed(2019)
tf.set_random_seed(2019)

# parameters
param = Param()


# load test dataset
tf.logging.info("Init test dataset ...")
test_filenames, test_xy, test_images = load_full_dataset(param.data_dir, param.test_datasets)

# data pipleline
tf.logging.info("Init train dataset and data pipeline ...")
with tf.device('/cpu:0'), tf.name_scope('input'):
    test_dataset, test_data_sampler = test_input_fn(
        patch_size=param.patch_size,
        n_channels=param.n_channels,
        batch_size=param.batch_size
    )

    data_iterator = tf.data.Iterator.from_structure(
        test_dataset.output_types,
        test_dataset.output_shapes
    )
    test_dataset_init = data_iterator.make_initializer(test_dataset)
    batch_data = data_iterator.get_next()

# build model
tf.logging.info("Creating the model ...")
anchors, positives, negatives = batch_data
spec = triplet_model_fn(
    anchors, positives, negatives, n_feats=param.n_features, name='triplet-net',
    mode='TRAIN', margin=param.margin,
    use_regularization_loss=param.use_regularization,
    learning_rate=param.learning_rate
)
estimator = TripletEstimator(spec, save_dir=param.log_dir,
                             model_path='../log/ms_softmargin_f128_lr')

# evaluator for image retrieval accuracy
evaluator = Evaluator(top_k=param.top_k, iou_threshold=param.iou_threshold)
n_eval_images = 5
eval_ind = np.arange(len(test_images))
np.random.shuffle(eval_ind)
eval_start_idx = 0



val_idx = 0

test_data_sampler.set_aug_vflip(False)
test_data_sampler.set_aug_hflip(False)

val_img = test_images[val_idx]
val_fn = test_filenames[val_idx]
val_xy = test_xy[val_idx]

# fixed query, varied db

test_data_sampler.set_image(val_img)

tf.logging.info("Preparing fixed-scale query samples.")
test_data_sampler.set_aug_multiscale(None)
test_data_sampler.set_sampling_method(False, xy=val_xy)
test_q = estimator.run(test_dataset_init, True)

tf.logging.info("Build multi-scale database & test ...")
test_results = dict()
for idx, down_factor in enumerate(param.down_factors):
    test_data_sampler.set_aug_multiscale([down_factor])
    test_data_sampler.set_sampling_method(True, stride=param.stride)
    test_db = estimator.run(test_dataset_init, False)

    acc, top_k_ind, top_k_ious = evaluator(test_db, test_q, param.patch_size)
    print("Accuracy @ {}: {}".format(down_factor, acc))

    results = {
        "db": test_db.to_dict(),
        "acc": acc,
        "down": down_factor,
        "top_k_ind": top_k_ind,
        "top_k_ious": top_k_ious
    }

    test_results["down-{:d}".format(down_factor)] = results

test_results["q"] = test_q.to_dict()
test_results["image"] = val_img
test_results["filename"] = val_fn
np.save("../test_ms_fixed.npy", test_results)








