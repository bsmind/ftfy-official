import numpy as np
from network.dataset.data_sampler import DataSamplerIoU, random_iou_offset, patchify
from utils.utils import compute_local_std_map
from tqdm import tqdm

class DataSamplerIoUExt(DataSamplerIoU):
    def __init__(self,
                 data,
                 n_channels,
                 patch_size,
                 iou_thresholds,
                 n_img_per_iter,
                 n_crops_per_img,
                 aug_kwarg):
        super().__init__(data, n_channels, patch_size,
                         iou_thresholds, n_img_per_iter, n_crops_per_img, aug_kwarg)

        # todo: compute mask for each image (optimize)
        self.mask = []
        for img in tqdm(data, desc='image mask'):
            stdmap = compute_local_std_map(np.squeeze(img))
            stdmap = stdmap > 0.1
            #stdmap[:patch_size[0], :] = 0
            stdmap[-2*patch_size[0]:, :] = 0
            #stdmap[:, :patch_size[1]] = 0
            stdmap[:, -2*patch_size[1]:] = 0
            stdmap = stdmap.astype(np.bool)

            self.mask.append(stdmap)

    def __next__(self):
        # select random images to generate random samples
        if self.n_images - np.sum(self.img_p) <= self.n_img_per_iter:
            self.img_p[:] = 1

        img_p = self.img_p / np.sum(self.img_p)
        selected_img_ind = np.random.choice(
            self.n_images,
            self.n_img_per_iter,
            replace=False,
            p=img_p
        )

        # allocate memory to get the best performance
        # compared with appending to list approach
        n_samples = self.n_img_per_iter * self.n_crops_per_img
        out_shape = (n_samples, ) + self.patch_size + (self.n_channels,)
        anchors = np.empty(out_shape, dtype=np.float32)
        positives = np.empty(out_shape, dtype=np.float32)
        negatives = np.empty(out_shape, dtype=np.float32)

        count = 0 # the number of samples so far collected
        img_iter = 0
        # as one image can have less than `n_crops_per_img` valid location...
        while count < n_samples:
            img_ind = selected_img_ind[img_iter]
            cimg = self.data[img_ind]
            mask = self.mask[img_ind]

            # no difference in performance with/without patchify (just for readability)
            cimg_patches = patchify(cimg, self.patch_size)

            height, width, _ = cimg.shape

            # randomly select (x,y) position (left-upper corner)
            y, x = np.where(mask) # this is central positions
            if len(y) > self.n_crops_per_img:
                ind = np.random.choice(len(y), self.n_crops_per_img, replace=False)
                y = y[ind]
                x = x[ind]
            # shift the central positions to left-upper corner and crop them
            # to prevent from cropping out-of-region
            # y = np.maximum(y - self.patch_size[0], 0)
            # x = np.maximum(x - self.patch_size[1], 0)

            # positive samples
            p_offset_x, p_offset_y = random_iou_offset(
                img_size=self.patch_size,
                n_samples=len(y),
                low=self.iou_thresholds[0],
                high=1.0
            )

            # negative samples
            n_offset_x, n_offset_y = random_iou_offset(
                img_size=self.patch_size,
                n_samples=len(y),
                low=0.0,
                high=self.iou_thresholds[1]
            )

            for _x, _y, _px, _py, _nx, _ny in \
                zip(x, y, p_offset_x, p_offset_y, n_offset_x, n_offset_y):
                if count >= n_samples: break

                #print(count, _x, _y, _px, _py, _nx, _ny)

                anchors[count] = cimg_patches[_y, _x]
                positives[count] = cimg_patches[_y + _py, _x + _px]
                negatives[count] = cimg_patches[_y + _ny, _x + _nx]

                count += 1

            self.img_p[img_ind] = 0
            img_iter = (img_iter + 1) % self.n_img_per_iter

        self.aug_downsample(anchors, positives, negatives)
        return anchors, positives, negatives









