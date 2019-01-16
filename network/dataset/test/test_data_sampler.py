"""performance test for data sampler"""

from network.dataset.data_sampler import TripletDataSampler
#from network.dataset.data_sampler import DataSamplerIoU
from network.dataset.data_sampler_ext import DataSamplerIoUExt
from network.dataset.mnist_input_fn import get_mnist_train_data
import time
import numpy as np

def test_perf_triplet_data_sampler(n_samples=100):
    data = get_mnist_train_data("../../../data")
    data_sampler = TripletDataSampler(
        data=data,
        n_class=10,
        n_class_per_iter=10,
        n_img_per_class=10
    )

    start = time.time()
    for i, batch_data in zip(range(n_samples), data_sampler):
        pass
    end = time.time()

    print("Avg. sampling time: {:.3f} sec".format((end - start)/n_samples))

def test_perf_triplet_data_sampler_iou(n_samples=100):
    data = np.random.random((n_samples, 860, 1024, 1))
    data_sampler = DataSamplerIoUExt(
        data,
        n_channels = 1,
        patch_size = (416//2, 416//2),
        iou_thresholds=[0.7, 0.4],
        n_img_per_iter=10,
        n_crops_per_img=10,
        aug_kwarg={
            'down_factors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        }
    )

    start = time.time()
    for i, batch_data in zip(range(n_samples), data_sampler):
        #a, p, n = batch_data
        #print(a.shape)
        pass
    end = time.time()
    print("Avg. sampling time: {:.3f} sec".format((end - start)/n_samples))



if __name__ == "__main__":
    #test_perf_triplet_data_sampler(n_samples=100)
    test_perf_triplet_data_sampler_iou(10)


    # from numpy.lib.stride_tricks import as_strided
    # import matplotlib.pyplot as plt
    #
    # def patchify(img, patch_shape):
    #     img = np.ascontiguousarray(img) # won't make a copy if not needed
    #     H, W, C = img.shape
    #     h, w = patch_shape
    #     shape = ((H - h + 1), (W - w + 1), h, w, C)
    #     strides = img.itemsize * np.array([W, 1, W, 1, C])
    #     return as_strided(img, shape=shape, strides=strides)
    #
    # witdh = 6
    # height = 3
    # channels = 3
    # patch_size = 2
    # #img = np.arange(witdh * height, dtype=np.float32).reshape(witdh, height, 1)
    # img = np.arange(witdh*height*channels, dtype=np.float32).reshape(height, witdh, channels)
    #
    # patches = patchify(img, (patch_size, patch_size))
    # #patches = np.ascontiguousarray(patches)
    # #patches.shape = (-1, patch_size**2)
    # for img_patch in patches.reshape(-1, patch_size, patch_size, channels):
    #     print(img_patch)
    #     #print(img_patch.shape, img_patch.sum() / 2500)
