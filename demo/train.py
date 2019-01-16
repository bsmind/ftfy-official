import tensorflow as tf
import numpy as np

from utils.Param import Param
from utils.eval import Evaluator
from network.model_fn import triplet_model_fn
from network.dataset.sem_data_sampler import train_input_fn, test_input_fn, load_full_dataset
from network.train import TripletEstimator


# set seed for reproduction
np.random.seed(2019)
tf.set_random_seed(2019)

# parameters
param = Param(log_dir='../log')

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
estimator = TripletEstimator(spec)

# evaluator for image retrieval accuracy
evaluator = Evaluator(top_k=param.top_k, iou_threshold=param.iou_threshold)

for epoch in range(param.n_epoch):
    # tf.logging.info('TRAIN {:d} start...'.format(epoch))
    # estimator.train(dataset_initializer=train_dataset_init, log_every=param.train_log_every)
    # tf.logging.info('TRAIN {:d} end.'.format(epoch))

    val_idx = 0
    val_img = test_images[val_idx]
    val_fn = test_filenames[val_idx]
    val_xy = test_xy[val_idx]

    test_data_sampler.set_image(val_img)
    test_data_sampler.set_sampling_method(True, stride=param.stride)
    test_db = estimator.run(test_dataset_init, False)

    test_data_sampler.set_sampling_method(False, xy=val_xy)
    # todo: set multiscale parameters
    test_q = estimator.run(test_dataset_init, True)

    # todo: compute accuracy
    # accuracy, top_k_ind, top_k_iou = evaluator(test_db.features, test_q.features)
    evaluator(test_db.features, test_q.features)
    break





# if __name__ == '__main__':
#     tf.reset_default_graph()
#     tf.logging.set_verbosity(tf.logging.INFO)
#
#     # Load the parameters from json file
#     args = parser.parse_args()
#     json_path = os.path.join(args.model_dir, 'params.json')
#     assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
#     params = Params(json_path)
#
#     # Define the model
#     tf.logging.info("Creating the model...")
#     config = tf.estimator.RunConfig(tf_random_seed=230,
#                                     model_dir=args.model_dir,
#                                     save_summary_steps=params.save_summary_steps)
#     estimator = tf.estimator.Estimator(model_fn, params=params, config=config)
#
#     # Train the model
#     tf.logging.info("Starting training for {} epoch(s).".format(params.num_epochs))
#     estimator.train(lambda: train_input_fn(args.data_dir, params))
#
#     # Evaluate the model on the test set
#     tf.logging.info("Evaluation on test set.")
#     res = estimator.evaluate(lambda: test_input_fn(args.data_dir, params))
#     for key in res:
#         print("{}: {}".format(key, res[key]))









