import numpy as np

class IoUSampler(object):
    '''
    IoUSampler class is to efficiently retrieve possible offset values (translation in x- and y-
    direction) with given IoU range. Note that it is assumed that IoU is computed between two
    rectange in the same size.
    '''
    def __init__(self, patch_size, step=1):
        self.patch_size = patch_size
        self.step = step

        max_dx = patch_size[1]//2 + 1
        max_dy = patch_size[0]//2 + 1

        x = np.arange(-max_dx, max_dx+1, step)
        y = np.arange(-max_dy, max_dy+1, step)
        self.iou_map = np.zeros((len(y), len(x)), dtype=np.float32)

        S2 = 2 * patch_size[0] * patch_size[1]
        for iy, dy in enumerate(y):
            for ix, dx in enumerate(x):
                w = np.maximum(0, patch_size[1] - abs(dx))
                h = np.maximum(0, patch_size[0] - abs(dy))
                inter = w * h
                self.iou_map[iy, ix] = np.clip(inter / (S2 - inter), 0., 1.)

        self.max_dx = max_dx
        self.max_dy = max_dy
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(1, 1)
        # cax = ax.imshow(self.iou_map)
        # ax.set_title('IOU map: {}'.format(patch_size))
        # cbar = fig.colorbar(cax, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
        # plt.show()

    def __call__(self, low=0.0, high=1.0, n=1):
        lo_mask = low <= self.iou_map
        hi_mask = self.iou_map < high
        mask = np.multiply(lo_mask, hi_mask)
        dy, dx = np.where(mask)

        # print('lo: {:.3f}, hi: {:.3f}, {:d}'.format(
        #     low, high, len(dy)
        # ))
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(1, 3)
        # ax[0].imshow(lo_mask)
        # ax[1].imshow(hi_mask)
        # ax[2].imshow(mask)
        # plt.show()

        replace = True if len(dy) >= n else False
        ind = np.random.choice(len(dy), n, replace=replace)
        dy = dy[ind] - self.max_dx
        dx = dx[ind] - self.max_dy
        return dx, dy

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    iou_sampler = IoUSampler((64, 64))

    dx, dy = iou_sampler(0.99, 1.1)
    print(dx, dy)

    plt.imshow(iou_sampler.iou_map)
    plt.show()