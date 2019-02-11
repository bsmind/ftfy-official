import warnings
import os
import numpy as np

try:
    import tensorflow as tf
    from network.model_fn import triplet_model_fn
except ImportError:
    print("Fail to import DNN modules.")


from skimage.io import imread, imsave
from skimage.transform import resize
import matplotlib.pyplot as plt

import time

def load_data(data_dir, filename):
    return imread(os.path.join(data_dir, filename), as_gray=False)

def normalize_frames(frames):
    g_min, g_max = frames.min(), frames.max()
    for frame_idx in range(len(frames)):
        frame = frames[frame_idx]
        frame = (frame - frame.min()) / frame.ptp()
        frame = frame * (g_max - g_min) + g_min
        frames[frame_idx] = frame

    return frames

def normalize_image(im):
    return (im - im.min()) / im.ptp()

def show_frames(frames, interval=1.):
    fig, ax = plt.subplots(1, 1)
    h = ax.imshow(frames[0], vmin=frames.min(), vmax=frames.max())

    plt.ion()
    plt.show()
    time.sleep(interval)

    for frame in frames[1:]:
        h.set_data(frame)
        plt.draw()
        plt.pause(interval)
        time.sleep(interval)

    plt.ioff()
    plt.show()

def show_two_frames(frames_a, frames_b, interval=1., repeat=True):
    """Show two frames for the comparison
    TODO: handle runtime error when figure is closed by an user.
    """
    if len(frames_a) != len(frames_b):
        raise ValueError("Unmatched the number of frames!")

    n_frames = len(frames_a)
    frame_iter = 0

    fig, ax = plt.subplots(1, 2)
    h1 = ax[0].imshow(frames_a[frame_iter], vmin=frames_a.min(), vmax=frames_a.max())
    h2 = ax[1].imshow(frames_b[frame_iter], vmin=frames_b.min(), vmax=frames_b.max())

    plt.ion()
    plt.show()
    time.sleep(interval)

    try:
        while frame_iter < n_frames:
            frame_iter = frame_iter + 1
            if repeat:
                frame_iter = frame_iter % n_frames


            h1.set_data(frames_a[frame_iter])
            h2.set_data(frames_b[frame_iter])
            plt.draw()
            plt.pause(interval)
            time.sleep(interval)
    except:
        plt.close()
        print('figure is closed.')

def show_multi_frames(list_of_frames, interval=1., repeat=True):
    """Show two frames for the comparison
    TODO: handle runtime error when figure is closed by an user.
    """

    n_frames = len(list_of_frames)
    n_images = len(list_of_frames[0])
    image_iter = 0

    fig, ax = plt.subplots(1, n_frames)
    handlers = []
    for idx in range(n_frames):
        h = ax[idx].imshow(
            list_of_frames[idx][0],
            vmin=list_of_frames[idx].min(),
            vmax=list_of_frames[idx].max()
        )
        handlers.append(h)

    plt.ion()
    plt.show()
    time.sleep(interval)

    try:
        while image_iter < n_images:
            image_iter = image_iter + 1
            if repeat:
                image_iter = image_iter % n_images

            for idx in range(n_frames):
                h = handlers[idx]
                h.set_data(list_of_frames[idx][image_iter])
            plt.draw()
            plt.pause(interval)
            time.sleep(interval)
    except:
        plt.close()
        print('figure is closed.')

def my_resize(image, out_shape=(208, 208)):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        image = resize(image, out_shape)
    return image


class TripletNet(object):
    def __init__(self, model_path):
        self.anchors = tf.placeholder(tf.float32, (None, 208, 208, 1), "a")
        self.positives = tf.placeholder(tf.float32, (None, 208, 208, 1), "p")
        self.negatives = tf.placeholder(tf.float32, (None, 208, 208, 1), "n")

        self.spec = triplet_model_fn(
            self.anchors, self.positives, self.negatives,
            n_feats=128, name='triplet-net', mode='TEST'
        )

        config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            intra_op_parallelism_threads=8,
            inter_op_parallelism_threads=0
        )
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.saver = tf.train.Saver(max_to_keep=100)

        # initialize
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def get_feature(self, image_batches, batch_size=16):
        image_features = []

        N = len(image_batches)
        batch_iters = (N + batch_size) // batch_size
        for i in range(batch_iters):
            feed_dict = dict(self.spec.test_feed_dict)
            feed_dict[self.anchors] = image_batches[i*batch_size:(i+1)*batch_size]
            features = self.sess.run(
                self.spec.a_feat, feed_dict=feed_dict
            )
            image_features.append(features)

        image_features = np.concatenate(image_features, axis=0)
        return image_features

class AlignFrame(object):
    def __init__(self, x0, y0, w, h):
        self.x0 = x0
        self.y0 = y0
        self.w = w
        self.h = h

    def _get_patch(self, frame, x0=None, y0=None):
        if x0 is None: x0 = self.x0
        if y0 is None: y0 = self.y0
        return frame[y0: y0 + self.h, x0: x0 + self.w]

    def __call__(self, ref_frame, frame):
        ref_image = self._get_patch(ref_frame)
        width, height = frame.shape

        min_dist = np.inf
        x0, y0 = 0, 0
        for shift_y in range(height - self.h):
            for shift_x in range(width - self.w):
                tar_image = self._get_patch(frame, shift_x, shift_y)

                dist = np.sum(np.sqrt((ref_image - tar_image)**2))
                if dist < min_dist:
                    min_dist = dist
                    x0 = shift_x
                    y0 = shift_y

        print('distance: {:.3f}, shift x, y: ({}, {})'.format(
            min_dist, self.x0 - x0, self.y0 - y0
        ))

        aligned = np.roll(frame, (self.y0 - y0, self.x0 - x0), axis=(0, 1))
        return aligned

class AlignFrameNN(AlignFrame):
    def __init__(self, x0, y0, w, h, nn:TripletNet):
        super().__init__(x0, y0, w, h)
        self.nn = nn

    def __call__(self, ref_frame, frame):
        ref_image = self._get_patch(ref_frame)
        width, height = frame.shape

        ref_image = normalize_image(my_resize(ref_image, (208, 208)))
        ref_image = np.expand_dims(ref_image, 0)
        ref_image = np.expand_dims(ref_image, -1)
        ref_feat = self.nn.get_feature(ref_image)

        images = []
        x0, y0 = [], []
        for shift_y in range(height - self.h):
            for shift_x in range(width - self.w):
                tar_image = self._get_patch(frame, shift_x, shift_y)
                tar_image = normalize_image(my_resize(tar_image, (208, 208)))
                tar_image = np.expand_dims(tar_image, 0)
                tar_image = np.expand_dims(tar_image, -1)

                images.append(tar_image)
                x0.append(shift_x)
                y0.append(shift_y)

        images = np.concatenate(images, axis=0)
        x0, y0 = np.array(x0), np.array(y0)

        feats = self.nn.get_feature(images, batch_size=128)

        distances = np.sum(np.sqrt((ref_feat - feats)**2), axis=1)
        min_idx = np.argsort(distances)[0]
        min_dist = distances[min_idx]
        x0 = x0[min_idx]
        y0 = y0[min_idx]

        print('distance: {:.3f}, shift x, y: ({}, {})'.format(
            min_dist, self.x0 - x0, self.y0 - y0
        ))

        aligned = np.roll(frame, (self.y0 - y0, self.x0 - x0), axis=(0, 1))
        return aligned



interval = 0.05
data_dir = '/home/sungsooha/Desktop/Data/ftfy/align'
#filename = 'ni_xrf_raw_normalized.tif'
filename = 'tomo_wo3_raw.tif'
#model_path = './log/ms_softmargin_f128_lr'

frames = load_data(data_dir, filename)
print(frames.shape)

show_frames(frames, 0.1)

frames = normalize_frames(frames)
frames_before = frames.copy()
frames_l2 = frames.copy()
#frames_nn = frames.copy()

#net = TripletNet(model_path)


x0 = 25
y0 = 5
w =  100
h = 130
frame_mgr_l2 = AlignFrame(x0, y0, w, h)
#frame_mgr_nn = AlignFrameNN(x0, y0, w, h, net)
for frame_idx in range(len(frames)):
    if frame_idx == 0:
        continue
    print("Frame index: {:d}".format(frame_idx))
    frames_l2[frame_idx] = frame_mgr_l2(frames_l2[frame_idx-1], frames_l2[frame_idx])
    #frames_nn[frame_idx] = frame_mgr_nn(frames_nn[frame_idx-1], frames_nn[frame_idx])

# save result
# imsave(os.path.join(data_dir, 'frames_l2.tif'), frames_l2)
# imsave(os.path.join(data_dir, 'frames_nn.tif'), frames_nn)

show_multi_frames([frames_before, frames_l2], interval=interval)

