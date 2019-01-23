import numpy as np
import tensorflow as tf
from utils.utils import random_downsample, get_image_list, downsample

def inrange(ref_min, ref_max, v):
    return ref_min <= v < ref_max

def show_triplet(image, a_xy, p_xy, n_xy, rect_sz):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(1, 1)
    ax.imshow(image, vmin=0, vmax=1)
    ax.add_patch(
        Rectangle(
            xy=a_xy,
            width=rect_sz[1],
            height=rect_sz[0],
            facecolor='none', edgecolor='black', linewidth=1
        )
    )
    ax.add_patch(
        Rectangle(
            xy=p_xy,
            width=rect_sz[1],
            height=rect_sz[0],
            facecolor='none', edgecolor='blue', linewidth=1
        )
    )
    ax.add_patch(
        Rectangle(
            xy=n_xy,
            width=rect_sz[1],
            height=rect_sz[0],
            facecolor='none', edgecolor='red', linewidth=1
        )
    )
    plt.show()

class ImageSamplerIoU(object):
    def __init__(self, patch_size, step=1):
        self.patch_size = patch_size
        self.step = step

        self.max_dx = patch_size[1]//2 + 1
        self.max_dy = patch_size[0]//2 + 1

        x = np.arange(0, self.max_dx+1, step)
        y = np.arange(0, self.max_dy+1, step)
        self.iou_map = np.zeros((len(y), len(x)), dtype=np.float32)

        S2 = 2 * patch_size[0] * patch_size[1]
        ones = np.ones(self.patch_size, dtype=np.int32)
        for dy in x:
            for dx in y:
                ones[:, :] = 1
                ones[dy:, dx:] += 1
                intersect_area = np.sum(ones==2)
                self.iou_map[dy, dx] = intersect_area / (S2 - intersect_area)

    def __call__(self, low=0.0, high=1.0, n=1):
        lo_mask = low <= self.iou_map
        hi_mask = self.iou_map < high
        mask = np.multiply(lo_mask, hi_mask)
        dy, dx = np.where(mask)

        ind = np.random.choice(len(dy), n, replace=True)
        dy = dy[ind]
        dx = dx[ind]
        return dx, dy

class DataSamplerIoU(object):
    def __init__(
            self,
             images, patch_xy, n_channels,
             patch_size = (208, 208),
             n_img_per_iter = 10,
             n_crops_per_img = 10,
             aug_multiscale=None
    ):
        self.images = images # list of images
        self.patch_xy = patch_xy # list of list of xy, index must be matched with images
        self.n_channels = n_channels # # channels of the image data

        self.patch_size = patch_size # patch's (height, width)
        self.n_img_per_iter = n_img_per_iter
        self.n_crops_per_img = n_crops_per_img

        # augmentations
        self.aug_multiscale = aug_multiscale

        # helpers
        self.n_image = len(self.images)
        self.img_p = np.ones((self.n_image, ), dtype=np.float32)
        self.sampler = ImageSamplerIoU(self.patch_size)

    def __iter__(self):
        return self

    def _get_random_image_ind(self):
        if self.n_image <= self.n_img_per_iter:
            return np.random.choice(self.n_image, self.n_img_per_iter, replace=True)

        if np.sum(self.img_p) <= self.n_img_per_iter:
            self.img_p[:] = 1.

        img_p = self.img_p / self.img_p.sum()
        return np.random.choice(self.n_image, self.n_img_per_iter, replace=False, p=img_p)

    def _get_image(self, image, x0, y0):
        x1 = x0 + self.patch_size[1]
        y1 = y0 + self.patch_size[0]
        return image[y0:y1, x0:x1]

    def _aug_downsample(self, anchors, positives, negatives):
        if self.aug_multiscale is not None:
            n_samples = len(anchors)
            for i in range(n_samples):
                anchors[i] = random_downsample(anchors[i], self.aug_multiscale)
                positives[i] = random_downsample(positives[i], self.aug_multiscale)
                negatives[i] = random_downsample(negatives[i], self.aug_multiscale)

    # def _get_triplet_examples(self):
    #     # easy triplets
    #     # positive > 0.7, negative < 0.3
    #     p_dx, p_dy = self.sampler(low=0.7, high=1.0, n=1)
    #     n_dx, n_dy = self.sampler(low=0.0, high=0.3, n=1)
    #
    #     # hard triplets
    #     # positive > negative, both are either > 0.5 or < 0.5
    #     if np.random.random() < 0.5:
    #         iou_thresh = np.random.choice(np.arange(0.1, 0.5, 0.1), 2, replace=False)
    #     else:
    #         iou_thresh = np.random.choice(np.arange(0.5, 1.0, 0.1), 2, replace=False)
    #     iou_thresh = np.sort(iou_thresh)
    #     p_dx, p_dy = self.sampler(low=iou_thresh[1], high=1.0, n=1)
    #     n_dx, n_dy = self.sampler(low=iou_thresh[0], high=iou_thresh[1], n=1)
    #
    #     # semi-hard
    #     # p_dx, p_dy = self.sampler(low=0.7, high=1.0, n=1)
    #     # n_dx, n_dy = self.sampler(low=0.0, high=0.7, n=1)
    #
    #     _px, _py = p_dx[0], p_dy[0]
    #     _nx, _ny = n_dx[0], n_dy[0]
    #     return _px, _py, _nx, _ny


    def __next__(self):
        # select random images to generate random samples
        selected_img_ind = self._get_random_image_ind()

        # memory allocation to get the best performance (compared with using list)
        n_samples = self.n_img_per_iter * self.n_crops_per_img
        out_shape = (n_samples, ) + self.patch_size + (self.n_channels, )
        anchors = np.empty(out_shape, dtype=np.float32)
        positives = np.empty(out_shape, dtype=np.float32)
        negatives = np.empty(out_shape, dtype=np.float32)

        count = 0
        img_iter = 0
        while count < n_samples:
            img_ind = selected_img_ind[img_iter]
            self.img_p[img_ind] = 0
            cimg = self.images[img_ind]
            cxy = self.patch_xy[img_ind]

            imgh, imgw, _ = cimg.shape

            x = [x for x, _ in cxy]
            y = [y for _, y in cxy]
            if len(cxy) > self.n_crops_per_img:
                x = np.random.choice(x, self.n_crops_per_img, replace=False)
                y = np.random.choice(y, self.n_crops_per_img, replace=False)

            def _select_v(_min_v, _max_v, _v0, _v1, _dv):
                if inrange(_min_v, _max_v, _v0 - _dv) and inrange(_min_v, _max_v, _v1 + _dv):
                    _dir = 1 if np.random.random() < 0.5 else -1
                    return _v0 + _dir * _dv
                elif inrange(_min_v, _max_v, _v0 - _dv):
                    return _v0 - _dv
                else:
                    return _v0 + _dv

            for _x, _y in zip(x, y):
                if count >= n_samples: break

                iou_thresh = np.random.choice(np.arange(0.1, 1.0, 0.1), 2, replace=False)
                iou_thresh = np.sort(iou_thresh)
                p_dx, p_dy = self.sampler(low=iou_thresh[1], high=1.0, n=1)
                n_dx, n_dy = self.sampler(low=iou_thresh[0], high=iou_thresh[1], n=1)

                _px, _py = p_dx[0], p_dy[0]
                _nx, _ny = n_dx[0], n_dy[0]

                x0 = _x - self.patch_size[1]//2
                y0 = _y - self.patch_size[0]//2
                x1 = x0 + self.patch_size[1]
                y1 = y0 + self.patch_size[0]

                px0 = _select_v(0, imgw, x0, x1, _px)
                py0 = _select_v(0, imgh, y0, y1, _py)

                nx0 = _select_v(0, imgw, x0, x1, _nx)
                ny0 = _select_v(0, imgh, y0, y1, _ny)

                anchors[count] = self._get_image(cimg, x0, y0)
                positives[count] = self._get_image(cimg, px0, py0)
                negatives[count] = self._get_image(cimg, nx0, ny0)
                count += 1

            img_iter = (img_iter + 1) % len(selected_img_ind)

        self._aug_downsample(anchors, positives, negatives)

        return anchors, positives, negatives

class TestDataSamplerIoU(object):
    def __init__(self, patch_size=(208, 208)):
        self.image = None
        self.patch_size = patch_size

        self.is_grid = True
        # helper for grid sampling
        self.stride = (1, 1)
        self.row = 0 # (row, col) to extract a patch (upper-left corner)
        self.col = 0
        # helper for sampling from predifined xy position
        self.xy = None
        self.xy_idx = 0
        # helper for random sampling
        self.n_samplings = 100 # max. number of random samplings
        self.cnt = 0 # sampling counter

        # augmentation
        self.aug_multiscale = None
        self.aug_hflip = False
        self.aug_vflip = False

    def set_image(self, image):
        if len(image.shape) != 3:
            raise ValueError("Expect three dimension for image shape (h x w x c)")
        self.image = image

    def set_patch_size(self, patch_size):
        self.patch_size = patch_size

    def set_sampling_method(self, is_grid,
                            stride=None,
                            xy=None,
                            n_samplings=None):
        if is_grid and stride is None:
            raise ValueError("Grid sampling required to define stride and patch size.")

        if not is_grid and xy is None and n_samplings is None:
            raise ValueError("Non-grid sampling required to define either xy or n_samplings")

        self.is_grid = is_grid
        self.stride = stride
        self.xy = xy
        self.n_samplings = n_samplings

        self.row, self.col, self.xy_idx, self.cnt = 0, 0, 0, 0

    def set_aug_multiscale(self, multiscale):
        self.aug_multiscale = multiscale

    def set_aug_hflip(self, hflip):
        self.aug_hflip = hflip

    def set_aug_vflip(self, vflip):
        self.aug_vflip = vflip

    def __iter__(self):
        return self

    def _get_patch(self, x0, y0):
        patch = self.image[y0: y0 + self.patch_size[0], x0: x0 + self.patch_size[1]]
        return patch

    def _grid_sampling(self):
        if self.image is None or self.stride is None or self.patch_size is None:
            raise ValueError("Grid sampling required to define image, stride, patch_size.")

        if self.row > self.image.shape[0] - self.patch_size[0]:
            self.row, self.col = 0, 0
            raise StopIteration

        y0, x0 = self.row, self.col

        patch = self._get_patch(x0, y0)

        self.col = x0 + self.stride[1]
        if self.col > self.image.shape[1] - self.patch_size[1]:
            self.col = 0
            self.row = y0 + self.stride[0]

        return patch, x0, y0

    def _random_sampling(self):
        if self.image is None or self.n_samplings is None:
            raise ValueError("Random sampling required to define image and n_samplings.")

        if self.cnt >= self.n_samplings:
            self.cnt = 0
            raise StopIteration

        y0 = np.random.choice(self.image.shape[0] - self.patch_size[0])
        x0 = np.random.choice(self.image.shape[1] - self.patch_size[1])

        patch = self._get_patch(x0, y0)
        self.cnt += 1

        return patch, x0, y0

    def _xy_sampling(self):
        if self.image is None or self.xy is None:
            raise ValueError("XY sampling required to define image and xy.")

        if self.xy_idx >= len(self.xy):
            self.xy_idx = 0
            raise StopIteration

        cx, cy = self.xy[self.xy_idx]
        x0 = cx - self.patch_size[0]//2
        y0 = cy - self.patch_size[0]//2
        patch = self._get_patch(x0, y0)

        self.xy_idx += 1
        return patch, x0, y0

    def _aug_downsample(self, im):
        if self.aug_multiscale is not None:
            im = random_downsample(im, self.aug_multiscale)
        return im

    def _aug_hflip(self, im):
        if self.aug_hflip and np.random.random() < 0.5:
            im = np.fliplr(im)
        return im

    def _aug_vflip(self, im):
        if self.aug_vflip and np.random.random() < 0.5:
            im = np.flipud(im)
        return im

    def __next__(self):
        if self.is_grid:
            im, x0, y0 = self._grid_sampling()
        else:
            if self.xy is not None:
                im, x0, y0 = self._xy_sampling()
            else:
                im, x0, y0 = self._random_sampling()

        im = self._aug_downsample(im)
        im = self._aug_vflip(im)
        im = self._aug_hflip(im)
        im = (im - im.min()) / im.ptp()

        # to match output shape with DataSamplerIoU
        x0 = x0 * np.ones_like(im, dtype=np.float32)
        y0 = y0 * np.ones_like(im, dtype=np.float32)
        return im, x0, y0

def train_preprocessor(anchors, positives, negatives):

    # anchors = tf.image.random_flip_up_down(anchors)
    # anchors = tf.image.random_flip_left_right(anchors)
    # anchors = tf.image.random_contrast(anchors, lower=0.2, upper=1.8)
    # anchors = tf.image.random_brightness(anchors, max_delta=0.5)

    # anchors = tf.image.per_image_standardization(anchors)
    # positives = tf.image.per_image_standardization(positives)
    # negatives = tf.image.per_image_standardization(negatives)

    epsilon = tf.constant(1e-12, dtype=tf.float32)
    anchors = tf.div(
        tf.subtract(anchors, tf.reduce_min(anchors)),
        tf.maximum(tf.subtract(tf.reduce_max(anchors), tf.reduce_min(anchors)), epsilon)
    )

    positives = tf.div(
        tf.subtract(positives, tf.reduce_min(positives)),
        tf.maximum(tf.subtract(tf.reduce_max(positives), tf.reduce_min(positives)), epsilon)
    )

    negatives = tf.div(
        tf.subtract(negatives, tf.reduce_min(negatives)),
        tf.maximum(tf.subtract(tf.reduce_max(negatives), tf.reduce_min(negatives)), epsilon)
    )

    return anchors, positives, negatives

def _load_dataset(path):
    dataset = dict(np.load(path).item())
    return dataset['filenames'], dataset['XY']

def load_full_dataset(data_dir, path):
    filenames, xy = _load_dataset(path)
    images = get_image_list(data_dir, filenames)
    return filenames, xy, images

def train_input_fn(data_dir, dataset_path, batch_size=16,
             n_channels=1, patch_size=(208,208),
             n_img_per_iter=10, n_crops_per_img=10, n_iter=10,
             aug_multiscale=np.arange(2, 14, 2)):

    filenames, patch_xy = _load_dataset(dataset_path)
    images = get_image_list(data_dir, filenames)

    if len(images) == 0 or images[0].shape[-1] != n_channels:
        raise ValueError("No image data or unmatched channel dimension.")

    data_sampler = DataSamplerIoU(
        images, patch_xy,
        n_channels=n_channels,
        patch_size=patch_size,
        n_img_per_iter=n_img_per_iter,
        n_crops_per_img=n_crops_per_img,
        aug_multiscale=aug_multiscale,
    )
    output_shape = [n_img_per_iter*n_crops_per_img, *patch_size, n_channels]
    tf_dataset = (
        tf.data.Dataset
            .from_generator(generator=lambda: data_sampler,
                            output_types=(tf.float32, tf.float32, tf.float32),
                            output_shapes=(output_shape, output_shape, output_shape))
            .take(count=n_iter)
            .flat_map(map_func=lambda a, p, n: tf.data.Dataset.from_tensor_slices((a, p, n)))
            .map(train_preprocessor, 8)
            .shuffle(buffer_size=2*batch_size)
            .batch(batch_size=batch_size)
            .prefetch(1)
    )
    return tf_dataset

def test_input_fn(patch_size, n_channels, batch_size):
    data_sampler = TestDataSamplerIoU(patch_size=patch_size)
    output_shape = [*patch_size, n_channels]
    tf_dataset = (
        tf.data.Dataset()
            .from_generator(generator=lambda: data_sampler,
                            output_types=(tf.float32, tf.float32, tf.float32),
                            output_shapes=(output_shape, output_shape, output_shape))
            .batch(batch_size=batch_size)
            .prefetch(1)
    )
    return tf_dataset, data_sampler







