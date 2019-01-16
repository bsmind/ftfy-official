import numpy as np

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
        hi_mask = self.iou_map <= high
        mask = np.multiply(lo_mask, hi_mask)
        dy, dx = np.where(mask)

        ind = np.random.choice(len(dy), n, replace=True)
        dy = dy[ind]
        dx = dx[ind]
        return dx, dy



if __name__ == '__main__':

    width = 208
    height = 208

    sampler = ImageSamplerIoU((height, width))
    dx, dy = sampler(low=0.5, high=1.0, n=10)
    print(dx, dy)

    # min_x = -width//2 - 1
    # max_x = width//2 + 1
    #
    # min_y = -height//2 - 1
    # max_y = height//2 + 1
    #
    # step_x = 1
    # step_y = 1
    #
    # x = np.arange(min_x, max_x+1, step_x)
    # y = np.arange(min_y, max_y+1, step_y)
    #
    # iou_map = np.zeros((len(y), len(x)), dtype=np.float32)
    #
    # ox = 10
    # oy = 1000
    # ones = np.ones((height, width), dtype=np.int32)
    # ones[oy:, ox:] += 1
    # intersect_area = np.sum(ones==2)
    # iou = intersect_area / (2*height*width - intersect_area)
    # print(iou)
