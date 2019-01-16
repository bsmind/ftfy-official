import numpy as np
from numpy.lib.stride_tricks import as_strided
from utils.utils import downsample, compute_local_std_map

def build_data_map(images, labels, n_class):
    """Build data hash map, key: class id -> value: images in the class id"""
    data_map = {}
    for i in range(n_class):
        data_map[i] = images[labels == i].copy()
    return data_map

def random_iou_offset(img_size, n_samples, low=0., high=1.):
    target_iou = np.random.uniform(
        low=low,
        high=high,
        size=n_samples)

    ratio = (1 - target_iou) / (1 + target_iou)
    offset_max_y = img_size[0] * ratio
    offset_max_y = offset_max_y.astype(np.int32)

    offset_y = np.array([
        np.random.randint(
            low=0,
            high=max_y if max_y else 1
        ) for max_y in offset_max_y
    ], dtype=np.int32)

    offset_x = (img_size[0] * ratio - offset_y) / (img_size[0] - offset_y)
    offset_x = img_size[1] * offset_x
    offset_x = offset_x.astype(np.int32)

    return offset_x, offset_y

def patchify(img, patch_shape):
    img = np.ascontiguousarray(img)  # won't make a copy if not needed
    H, W, C = img.shape
    h, w = patch_shape
    shape = ((H - h + 1), (W - w + 1), h, w, C)
    strides = img.itemsize * np.array([W, 1, W, 1, C])
    return as_strided(img, shape=shape, strides=strides)

class DataSampler(object):
    """Given data (images, labels), it randomly samples data for each class."""
    def __init__(self, data, n_class, n_class_per_iter, n_img_per_class):
        images, labels = data
        self.data_map = build_data_map(images, labels, n_class)
        self.n_class = n_class
        self.n_class_per_iter = n_class_per_iter
        self.n_img_per_class = n_img_per_class

    def __iter__(self):
        return self

    def __next__(self):
        choiced_classes = np.random.choice(
            self.n_class,
            self.n_class_per_iter,
            replace=False
        )

        images = []
        labels = []
        for c in choiced_classes:
            cimgs = self.data_map[c]
            choiced_imgs = np.random.choice(
                cimgs.shape[0],
                self.n_img_per_class,
                replace=False
            )
            images.append(cimgs[choiced_imgs])

            clabels = np.empty([self.n_img_per_class], dtype=np.int32)
            clabels[:] = c
            labels.append(clabels)

        images = np.concatenate(images, axis=0)
        labels = np.concatenate(labels, axis=0)
        return images, labels


class TripletDataSampler(DataSampler):
    def __init__(self, data, n_class, n_class_per_iter, n_img_per_class):
        super().__init__(data, n_class, n_class_per_iter, n_img_per_class)

        images, labels = data
        self.images = images
        self.labels = labels

        #np.where()

    def __next__(self):
        choiced_classes = np.random.choice(
            self.n_class,
            self.n_class_per_iter,
            replace=False
        )

        anchors = []
        positives = []
        negatives = []
        labels = [] # only anchors' labels
        for c in choiced_classes:
            cimgs = self.data_map[c]
            labels += [c] * self.n_img_per_class

            choiced_imgs_for_anchors = np.random.choice(
                cimgs.shape[0],
                self.n_img_per_class,
                replace=False
            )
            anchors.append(cimgs[choiced_imgs_for_anchors])

            # sampling positive examples
            p = np.ones(cimgs.shape[0], dtype=np.float32)
            p[choiced_imgs_for_anchors] = 0
            p /= p.sum()
            choiced_imgs_for_positive = np.random.choice(
                cimgs.shape[0],
                self.n_img_per_class,
                p=p,
                replace=False
            )
            positives.append(cimgs[choiced_imgs_for_positive])

            # sampling negative examples
            # To have high performance, first randomly choose indices of negative examples,
            # then retrieve images from the array.
            ind = np.where(self.labels != c)[0]
            choiced_imgs_for_negative = np.random.choice(
                ind,
                self.n_img_per_class,
                replace=False
            )
            nimgs = self.images[choiced_imgs_for_negative]
            negatives.append(nimgs)


        anchors = np.concatenate(anchors, axis=0)
        positives = np.concatenate(positives, axis=0)
        negatives = np.concatenate(negatives, axis=0)
        labels = np.array(labels, dtype=np.int32)
        return anchors, positives, negatives, labels


class DataSamplerIoU(object):
    """Given image data, it randomly samples (crop) data based on iou."""
    def __init__(self, data, n_channels,
                 patch_size,
                 iou_thresholds,
                 n_img_per_iter, n_crops_per_img,
                 aug_kwarg={}):
        self.data = data
        self.n_images = len(data)
        self.n_channels = n_channels
        self.patch_size = patch_size
        self.iou_thresholds = iou_thresholds
        self.n_img_per_iter = n_img_per_iter
        self.n_crops_per_img = n_crops_per_img
        self.img_p = np.ones((self.n_images,), dtype=np.float32)
        self.aug_kwarg = aug_kwarg

    def __iter__(self):
        return self

    def downsample(self, image, factors):
        if np.random.random() < 0.7:
            blk = np.random.choice(factors)
            if len(image.shape) == 3:
                factors = (blk, blk, 1)
            else:
                factors = (blk, blk)

            image = downsample(image, self.patch_size, factors)
        return image

    def aug_downsample(self, anchors, positives, negatives):
        down_factors = self.aug_kwarg.get('down_factors', None)
        if down_factors is not None:
            n_samples = len(anchors)
            for i in range(n_samples):
                anchors[i] = self.downsample(anchors[i], down_factors)
                positives[i] = self.downsample(positives[i], down_factors)
                negatives[i] = self.downsample(negatives[i], down_factors)
        #return anchors, positives, negatives

    def __next__(self):
        # selected random images to genrate random samples
        if self.n_images - np.sum(self.img_p) <= self.n_img_per_iter:
            self.img_p[:] = 1

        img_p = self.img_p / np.sum(self.img_p)
        selected_img_ind = np.random.choice(
            self.n_images,
            self.n_img_per_iter,
            replace=False,
            p=img_p
        )

        n_samples = self.n_img_per_iter * self.n_crops_per_img

        anchors = np.empty((n_samples, ) + self.patch_size + (self.n_channels,), dtype=np.float32)
        positives = np.empty((n_samples, ) + self.patch_size + (self.n_channels,), dtype=np.float32)
        negatives = np.empty((n_samples, ) + self.patch_size + (self.n_channels,), dtype=np.float32)
        # loop over the selected image
        for it, img_idx in enumerate(selected_img_ind):
            cimg = self.data[img_idx]
            height, width, channels = cimg.shape

            # no difference in performance with/without patchify (just for readability)
            cimg_patches = patchify(cimg, self.patch_size)

            # randomly select (x,y) position (left-upper corner)
            x = np.random.choice(width - 2*self.patch_size[1], self.n_crops_per_img, replace=False)
            y = np.random.choice(height - 2*self.patch_size[0], self.n_crops_per_img, replace=False)

            # positive samples
            p_offset_x, p_offset_y = random_iou_offset(
                img_size=self.patch_size,
                n_samples=self.n_crops_per_img,
                low=self.iou_thresholds[0],
                high=1.0
            )

            # negative samples
            n_offset_x, n_offset_y = random_iou_offset(
                img_size=self.patch_size,
                n_samples=self.n_crops_per_img,
                low=0.0,
                high=self.iou_thresholds[1]
            )

            # anchor samples
            it_offset = it * self.n_crops_per_img
            for i in range(self.n_crops_per_img):
                _x = x[i]
                _y = y[i]
                _px = p_offset_x[i]
                _py = p_offset_y[i]
                _nx = n_offset_x[i]
                _ny = n_offset_y[i]

                anchors[it_offset + i] = cimg_patches[_y, _x]
                positives[it_offset + i] = cimg_patches[_y + _py, _x + _px]
                negatives[it_offset + i] = cimg_patches[_y + _ny, _x + _nx]

            self.img_p[img_idx] = 0

        # multi-scale
        # down-sample and resize to the original size
        self.aug_downsample(anchors, positives, negatives)

        return anchors, positives, negatives


class TestDataSamplerIoU(object):
    def __init__(self, patch_size, stride=(1,1), image=None,
                 random_sampling=False, augmentor=None):
        self.nrows = None
        self.ncols = None
        self.image = None
        self.mask = None
        self.limit_r = None
        self.limit_c = None
        self.patch_size = None
        self.stride = stride
        self.r = 0
        self.c = 0
        self.anchor = None
        self.random_sampling = random_sampling
        self.augmentor = augmentor
        self.reset(image, stride, patch_size, random_sampling, augmentor)

    def reset(self, image, stride=None, patch_size=None,
              random_sampling=False, augmentor=None):
        if patch_size is not None:
            self.patch_size = patch_size

        if image is not None:
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=-1)

            self.image = image
            self.nrows = image.shape[0]
            self.ncols = image.shape[1]

            stdmap = compute_local_std_map(np.squeeze(image))
            stdmap = stdmap > 0.1
            #stdmap[:patch_size[0], :] = 0
            stdmap[-patch_size[0]:, :] = 0
            #stdmap[:, :patch_size[1]] = 0
            stdmap[:, -patch_size[1]:] = 0
            stdmap = stdmap.astype(np.bool)

            self.mask = stdmap

        if stride is not None:
            self.stride = stride

        if self.nrows is not None and self.ncols is not None:
            self.limit_r = self.nrows - self.patch_size[0]
            self.limit_c = self.ncols - self.patch_size[1]



        self.r = 0
        self.c = 0
        self.random_sampling = random_sampling
        self.augmentor = augmentor

    def __iter__(self):
        return self

    def __next__(self):
        if self.r > self.limit_r:
            raise StopIteration

        if self.random_sampling:
            rows, cols = np.where(self.mask)
            if len(rows):
                ind = np.random.choice(len(rows))
                lr = rows[ind]
                lc = cols[ind]
            else:
                lr = np.random.randint(0, self.limit_r)
                lc = np.random.randint(0, self.limit_c)
        else:
            lr = self.r
            lc = self.c

        ur = lr + self.patch_size[0]
        uc = lc + self.patch_size[1]
        patch = self.image[lr: ur, lc: uc]

        if self.augmentor is not None:
            patch = self.augmentor(patch)

        self.c = self.c + self.stride[1]
        if self.c > self.limit_c:
            self.c = 0
            self.r = self.r + self.stride[0]

        return patch, \
               lr * np.ones_like(patch), \
               lc * np.ones_like(patch)








