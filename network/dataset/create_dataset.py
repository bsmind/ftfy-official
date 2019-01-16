'''
To manually create dataset
'''
import numpy as np
from matplotlib.patches import Rectangle
from utils.utils import get_fullpath, load_image, compute_local_std_map

class InteractiveImageHandler(object):
    def __init__(self, imh, image_size, patch_size):
        self.imh = imh
        self.image_size = image_size
        self.patch_size = patch_size

        self.idx_to_rect = dict()
        '''
        n: get new image
        o: output rect info
        r: add random 'n' rect
        d: dump rect info
        '''
        self.key_actions = {
            '<': lambda: print('next'),
            '>': lambda: print('next'),
            'o': lambda key, info: print(key, info),
            'r': lambda: self._add_random_rect(5),
            'd': lambda: self.dump_rect()
        }
        self.mask = None

        self.cidpress = None
        self.cidrelease = None
        self.cidkeypress = None
        self.cidkeyrelease = None

    def connect(self):
        self.cidpress = self.imh.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.imh.figure.canvas.mpl_connect(
            'button_release_event', self.on_release
        )
        self.cidkeypress = self.imh.figure.canvas.mpl_connect(
            'key_press_event', self.on_key_press
        )
        self.cidkeyrelease = self.imh.figure.canvas.mpl_connect(
            'key_release_event', self.on_key_release
        )

    def disconnect(self):
        self.imh.figure.canvas.mpl_disconnect(self.cidpress)
        self.imh.figure.canvas.mpl_disconnect(self.cidrelease)
        self.imh.figure.canvas.mpl_disconnect(self.cidkeypress)
        self.imh.figure.canvas.mpl_disconnect(self.cidkeyrelease)

    def register_key_actions(self, actions):
        for k, v in actions.items():
            self.key_actions[k] = v

    def _check_point(self, x, y):
        if x <= self.patch_size[1]//2 or y <= self.patch_size[0]//2:
            return False
        if x >= self.image_size[1] - self.patch_size[1]//2 or \
                y >= self.image_size[0] - self.patch_size[0]//2:
            return False
        return True

    def _xy_to_idx(self, x, y):
        return y * self.image_size[1] + x

    def _idx_to_xy(self, idx):
        y = idx // self.image_size[1]
        x = idx % self.image_size[1]
        return x, y

    def _get_rect(self, x, y):
        idx = self._xy_to_idx(x, y)
        rect = self.idx_to_rect.get(idx, None)
        is_new = False
        if rect is None:
            rect = Rectangle(
                xy=(x - self.patch_size[1]//2, y - self.patch_size[0]//2),
                width=self.patch_size[0],
                height=self.patch_size[1],
                facecolor='none', edgecolor='red', linewidth=1
            )
            self.idx_to_rect[idx] = rect
            is_new = True
        return rect, is_new

    def _add_rect(self, x, y):
        rect, is_new = self._get_rect(x, y)
        if is_new:
            self.imh._axes.add_patch(rect)

    def _del_rect(self, x, y):
        if len(self.idx_to_rect.keys()) == 0:
            return

        def _dist(_a):
            return np.sqrt((_a[0] - x)**2 + (_a[1] - y)**2)

        xy_list = []
        keys = []
        for key, rect in self.idx_to_rect.items():
            xy = rect.get_xy()
            check1 = xy[0] <= x <= xy[0] + self.patch_size[1]
            check2 = xy[1] <= y <= xy[1] + self.patch_size[1]
            if check1 and check2:
                xy_list.append(xy)
                keys.append(key)

        if len(xy_list) == 0:
            return

        xy_dist = np.array([_dist(xy) for xy in xy_list])
        idx = np.argmin(xy_dist)
        if isinstance(idx, np.ndarray):
            idx = idx[0]
        key_to_del = keys[idx]
        rect_to_del = self.idx_to_rect[key_to_del]
        rect_to_del.remove()
        del self.idx_to_rect[key_to_del]

    def _add_random_rect(self, n=5):
        xs, ys = [], []
        if self.mask is not None:
            ys, xs = np.where(self.mask)
            if len(ys) > n:
                ind = np.random.choice(len(ys), n, replace=False)
                ys = ys[ind]
                xs = xs[ind]

        if len(xs) == 0:
            min_y = self.patch_size[0]//2 + 1
            max_y = self.image_size[0] - self.patch_size[0]//2 - 1
            min_x = self.patch_size[1]//2 + 1
            max_x = self.image_size[1] - self.patch_size[1]//2 - 1
            xs = np.random.choice(
                np.arange(min_x, max_x, 1, dtype=np.int32),
                n, replace=False
            )
            ys = np.random.choice(
                np.arange(min_y, max_y, 1, dtype=np.int32),
                n, replace=False
            )

        for x, y in zip(xs, ys):
            self._add_rect(x, y)

        self.imh.figure.canvas.draw()

    def on_press(self, event):
        if event.inaxes is None:
            return

        x = int(event.xdata)
        y = int(event.ydata)
        if not self._check_point(x, y):
            return

        if event.button == 1: # add patch, left click
            self._add_rect(x, y)
        elif event.button == 3: # delete patch, right click
            self._del_rect(x, y)

    def on_release(self, event):
        self.imh.figure.canvas.draw()

    def on_key_press(self, event):
        if self.key_actions is None:
            return

        action = self.key_actions.get(event.key, None)
        if action is None:
            return

        if event.key == ',' or event.key == '.':
            self.set_image(*action())
        elif event.key == 'r':
            action()
        elif event.key == 'o':
            key = self.imh._axes.get_title()
            action(key, self.dump_rect())

    def on_key_release(self, event):
        self.imh.figure.canvas.draw()

    def clean_rect(self):
        for idx, rect in self.idx_to_rect.items():
            rect.remove()
        self.idx_to_rect = dict()

    def set_image(self, filename, image, mask=None, info=None):
        key = self.imh._axes.get_title()
        action = self.key_actions.get('o', None)
        if len(key) and action is not None:
            action(key, self.dump_rect())
        self.clean_rect()

        self.imh._axes.set_title(filename)
        if self.image_size == image.shape:
            self.imh.set_data(image)
            ph, pw = self.patch_size
            mask[:ph//2, :] = 0
            mask[-ph//2:, :] = 0
            mask[:, :pw//2] = 0
            mask[:,-pw//2:] = 0
            self.mask = mask
        else:
            self.imh.set_data(np.zeros(self.image_size, dtype=np.float32))
            self.mask = None

        if info is not None:
            for xy in info:
                x, y = xy
                #x = x + self.patch_size[1]//2
                #y = y + self.patch_size[0]//2
                self._add_rect(x, y)

    def dump_rect(self):
        xy = []
        for idx, rect in self.idx_to_rect.items():
            _xy = rect.get_xy()
            xy.append((_xy[0] + self.patch_size[1]//2, _xy[1] + self.patch_size[0]//2))
        #xy = [rect.get_xy() for idx, rect in self.idx_to_rect.items()]
        return xy

class ImageContainer(object):
    def __init__(self, data_dir, file_list):
        self.data_dir = data_dir
        self.file_list = file_list
        self.idx = -1

        # key: filename
        # info: list of xy
        self.file_to_info = dict()

    def reset(self):
        self.idx = 0

    def get_image(self, idx):
        '''Get an image'''
        filename = self.file_list[idx]
        info = self.file_to_info.get(filename, None)

        fullpath = get_fullpath(self.data_dir, filename)
        im = load_image(fullpath, True)
        mask = compute_local_std_map(im)
        mask = (mask > 0.1).astype(np.bool)

        return filename, im, mask, info

    def next(self):
        '''get a next image'''
        idx = self.idx + 1
        idx = np.maximum(0, idx % len(self.file_list))
        filename, im, mask, info = self.get_image(idx)
        self.idx = idx
        return filename, im, mask, info

    def prev(self):
        '''get a previous image'''
        idx = self.idx - 1
        if idx < 0:
            idx = len(self.file_list) - 1
        filename, im, mask, info = self.get_image(idx)
        self.idx = idx
        return filename, im, mask, info

    def save_info(self, key, info):
        if len(info):
            self.file_to_info[key] = info
            print("{:s}: {}".format(key, len(info)))

    def save_to_file(self, keys, output_path, metas=None):
        print('save to file: ', output_path)
        out_info = dict()

        n_files = 0
        n_rects = 0
        for key in keys:
            info = self.file_to_info.get(key, None)
            if info is not None:
                meta = None if metas is None else metas.get(key, None)
                print("{}: {} ({})".format(key, len(info), meta))
                out_info[key] = (info, meta)

                n_files += 1
                n_rects += len(info)

        print("Total # files: ", n_files)
        print("Total # rects: ", n_rects)

        np.save(output_path, out_info)
        print("save at ", output_path)



