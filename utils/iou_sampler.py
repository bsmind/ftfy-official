import numpy as np

class IoUSampler(object):
    '''
    IoUSampler class is to efficiently retrieve possible offset values (translation in x- and y-
    direction) with given IoU range. Note that it is assumed that IoU is computed between two
    rectange in the same size.
    '''
    def __init__(self, patch_size, step=1):
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
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

    def __call__(self, low=0.0, high=1.0, n=1):
        lo_mask = low <= self.iou_map
        hi_mask = self.iou_map < high
        mask = np.multiply(lo_mask, hi_mask)
        dy, dx = np.where(mask)

        replace = True if len(dy) >= n else False
        ind = np.random.choice(len(dy), n, replace=replace)
        dy = dy[ind] - self.max_dx
        dx = dx[ind] - self.max_dy
        return dx, dy



