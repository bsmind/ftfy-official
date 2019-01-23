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
param = Param(log_dir='../log/ms_softmargin_f128_test')


# load test dataset
tf.logging.info("Init test dataset ...")
test_filenames, test_xy, test_images = load_full_dataset(param.data_dir, param.test_datasets)

# data pipleline
tf.logging.info("Init train dataset and data pipeline ...")
with tf.device('/cpu:0'), tf.name_scope('input'):
    train_dataset = train_input_fn(
        data_dir=param.data_dir,
        dataset_path=param.train_datasets,
        batch_size=param.batch_size,
        n_channels=param.n_channels,
        patch_size=param.patch_size,
        n_img_per_iter=param.n_img_per_iter,
        n_crops_per_img=param.n_crops_per_img,
        n_iter=param.n_iter,
        aug_multiscale=param.down_factors
    )
    test_dataset, test_data_sampler = test_input_fn(
        patch_size=param.patch_size,
        n_channels=param.n_channels,
        batch_size=param.batch_size
    )

    data_iterator = tf.data.Iterator.from_structure(
        train_dataset.output_types,
        train_dataset.output_shapes
    )
    train_dataset_init = data_iterator.make_initializer(train_dataset)
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
estimator = TripletEstimator(spec, save_dir=param.log_dir)

# evaluator for image retrieval accuracy
evaluator = Evaluator(top_k=param.top_k, iou_threshold=param.iou_threshold)
n_eval_images = 5
eval_ind = np.arange(len(test_images))
np.random.shuffle(eval_ind)
eval_start_idx = 0

n_accuracy = 5
accuracy_update_idx = 0
accuracy = np.zeros((n_accuracy, len(param.down_factors)), dtype=np.float32)

n_lines = len(param.down_factors) + 1
legends = ["down-{:d}".format(factor) for factor in param.down_factors]
legends.append("mean")
acc_plot = ScalarPlot(n_lines, legends)
loss_plot = ScalarPlot(1, ['triplet-loss'])

test_data_sampler.set_aug_vflip(False)
test_data_sampler.set_aug_hflip(False)

for epoch in range(param.n_epoch):
    tf.logging.info('TRAIN {:d} start...'.format(epoch))
    loss = estimator.train(dataset_initializer=train_dataset_init, log_every=param.train_log_every)
    loss_plot.update(np.array([loss]))
    tf.logging.info('TRAIN {:d} end.'.format(epoch))

    tf.logging.info('TEST {:d} start...'.format(epoch))
    ind = eval_ind[eval_start_idx:eval_start_idx + n_eval_images]
    if len(ind) == 0:
        raise ValueError("There must be at least one image to test!")

    local_acc = np.zeros((len(param.down_factors), ), dtype=np.float32)
    for val_idx in ind:
        val_img = test_images[val_idx]
        val_fn = test_filenames[val_idx]
        val_xy = test_xy[val_idx]

        test_data_sampler.set_image(val_img)
        test_data_sampler.set_aug_multiscale(None)
        test_data_sampler.set_sampling_method(True, stride=param.stride)
        test_db = estimator.run(test_dataset_init, False)

        test_data_sampler.set_sampling_method(False, xy=val_xy)
        for idx, down_factor in enumerate(param.down_factors):
            test_data_sampler.set_aug_multiscale([down_factor])
            test_q = estimator.run(test_dataset_init, False)
            acc, _, _ = evaluator(test_db, test_q)
            local_acc[idx] += acc
    local_acc /= len(ind)

    accuracy[accuracy_update_idx] = local_acc
    accuracy_update_idx = (accuracy_update_idx+1) % n_accuracy

    if len(ind) < n_eval_images:
        np.random.shuffle(eval_ind)
        eval_start_idx = 0
        tf.logging.info('Shuffle test image array!')
    else:
        eval_start_idx += n_eval_images

    avg_accuracy = np.mean(accuracy, axis=0)
    avg_avg_accuracy = np.mean(avg_accuracy)
    acc_plot.update(np.append(avg_accuracy, avg_avg_accuracy))
    # print('accuracy: ', accuracy)
    # print('avg. accuracy: ', avg_accuracy)
    # print('avg. avg. accuracy: ', avg_avg_accuracy)
    tf.logging.info('Accuracy: {}, {}'.format(avg_accuracy, avg_avg_accuracy))
    tf.logging.info('TEST {:d} end.'.format(epoch))

    if epoch % param.save_every == 0:
        estimator.save(param.project_name, global_step=epoch)


acc_plot.hold()










