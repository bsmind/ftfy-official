import os
import numpy as np
import tensorflow as tf
from skimage.io import imread
from tqdm import tqdm

tf.logging.set_verbosity(tf.logging.INFO)

def to_one_shot_labels(bboxes, imgsz=(256, 256), cellsz=(16,16)):
    """convert (cx, cy, w, h) to (response, _cx, _cy, _w, _h)
        cx, cy, w and h are in pixel coordinate, and
        _cx, _cy, _w, and _h are normalized one
    """
    if len(bboxes.shape) == 1:
        bboxes = np.expand_dims(bboxes, 0)

    n_bboxes = len(bboxes)
    n_cells = cellsz[0]*cellsz[1]

    cx = bboxes[:, 0] + .5
    cy = bboxes[:, 1] + .5
    w  = bboxes[:, 2]
    h  = bboxes[:, 3]

    cell_pos_x = np.floor(cx / imgsz[1] * cellsz[1])
    cell_pos_y = np.floor(cy / imgsz[0] * cellsz[0])
    cell_idx = cell_pos_y * cellsz[1] + cell_pos_x
    cell_offset = np.arange(n_bboxes) * n_cells
    cell_pos = cell_idx + cell_offset
    cell_pos = cell_pos.astype(np.uint)

    norm_bboxes = np.ones((n_bboxes, 5), dtype=np.float32)
    norm_bboxes[:,1] = cx / imgsz[1]
    norm_bboxes[:,2] = cy / imgsz[0]
    norm_bboxes[:,3] = w / imgsz[1]
    norm_bboxes[:,4] = h / imgsz[0]

    out_shape = (n_bboxes, n_cells, 5)
    out = np.zeros((n_bboxes*n_cells, 5), dtype=np.float32)
    out[cell_pos, :] = norm_bboxes
    out = out.reshape(out_shape)

    return out, norm_bboxes

def to_bboxes(labels, imgsz=(256,256)):
    if len(labels.shape) == 2:
        labels = np.expand_dims(labels, 0)

    idx1, idx2 = np.where(labels[...,0]==1)
    bboxes = labels[idx1, idx2, 1:]
    bboxes[..., 0] = bboxes[..., 0] * imgsz[1] - 0.5
    bboxes[..., 1] = bboxes[..., 1] * imgsz[0] - 0.5
    bboxes[..., 2] *= imgsz[1]
    bboxes[..., 3] *= imgsz[0]

    return bboxes

class FTFYPatchDataSampler(object):
    def __init__(self, base_dir, cellsz=(16,16)):
        self.base_dir = base_dir

        self.data = dict()
        self.sample_idx = 0
        self.n_samples = 0
        self.cellsz = cellsz


    def _load_image_fnames(self, dir_name, ext='bmp'):
        files = []
        dataset_dir = os.path.join(self.base_dir, dir_name)
        for f in os.listdir(dataset_dir):
            if f.endswith(ext):
                files.append(os.path.join(dataset_dir, f))
        return sorted(files)

    def _load_patches(
            self, dir_name, fnames, patch_size, n_channels,
            patches_per_row, patches_per_col, n_patches=None):
        if n_patches is None:
            patches_all = []
            n_patches = np.inf
        else:
            patches_all = np.zeros((n_patches, *patch_size, n_channels), dtype=np.float32)
        #print(self.PATCHES_PER_COL, self.PATCHES_PER_ROW)
        counter = 0
        done = False
        for f in tqdm(fnames, desc='Loading dataset %s' % dir_name):
            if done: break
            assert os.path.isfile(f), 'Not a file: %s' % f
            # todo: what if the maximum value is not 255?
            im = imread(f) / 255.
            patches_row = np.split(im, patches_per_row, axis=0)
            for row in patches_row:
                if done: break
                patches = np.split(row, patches_per_col, axis=1)
                for patch in patches:
                    if done: break
                    patch_tensor = patch.reshape(*patch_size, n_channels)
                    if isinstance(patches_all, list):
                        patches_all.append(patch_tensor)
                    else:
                        patches_all[counter] = patch_tensor
                    counter += 1
                    if counter >= n_patches:
                        done = True
        if isinstance(patches_all, list):
            patches_all = np.asarray(patches_all)
        return patches_all

    def load_dataset(
            self, data_dirs, src_dir, tar_dir='patches',
            src_ext='bmp', src_size=(256, 256), n_src_channels=1, src_per_col=10, src_per_row=10,
            tar_ext='bmp', tar_size=(128, 128), n_tar_channels=1, tar_per_col=10, tar_per_row=10,
            debug=True
    ):
        assert isinstance(data_dirs, list), 'Expect to get a list of data_dirs.'

        tar_offset = 0
        src_offset = 0

        datamap, bboxes = [], []
        tar_patches, src_patches = [], []
        for data_dir in data_dirs:
            info_path = os.path.join(self.base_dir, data_dir, src_dir, 'info.txt')
            with open(info_path, 'r') as file:
                n_targets = 0
                n_sources = -1
                for line in file:
                    tar_id, min_src_id, max_src_id, x0, y0, w, h = line.split()

                    tar_id = int(tar_id)
                    min_src_id, max_src_id = int(min_src_id), int(max_src_id)
                    x0, y0, w, h = float(x0), float(y0), float(w), float(h)
                    cx, cy = x0 + w/2., y0 + h/2.

                    datamap.append([
                        tar_id + tar_offset,
                        min_src_id + src_offset, max_src_id + src_offset
                    ])
                    bboxes.append([cx, cy, w, h])
                    n_targets += 1
                    n_sources = max(n_sources, max_src_id)

                # todo: load target and source images
                dir_name = os.path.join(data_dir, tar_dir)
                fnames = self._load_image_fnames(dir_name, tar_ext)
                tar_patches.append(
                    self._load_patches(dir_name, fnames, tar_size, n_tar_channels,
                                       tar_per_row, tar_per_col, n_targets)
                )

                dir_name = os.path.join(data_dir, src_dir)
                fnames = self._load_image_fnames(dir_name, src_ext)
                src_patches.append(
                    self._load_patches(dir_name, fnames, src_size, n_src_channels,
                                       src_per_row, src_per_col, n_sources)
                )

                tar_offset += n_targets
                src_offset += n_sources

        datamap = np.asarray(datamap, dtype=np.int32)
        bboxes = np.asarray(bboxes, dtype=np.float32)
        tar_patches = np.concatenate(tar_patches, axis=0)
        src_patches = np.concatenate(src_patches, axis=0)

        self.n_samples = len(datamap)
        self.sample_idx = 0
        ind = np.arange(self.n_samples)
        np.random.shuffle(ind)
        datamap = datamap[ind]
        bboxes = bboxes[ind]

        self.data['datamap'] = datamap
        self.data['bboxes'] = bboxes
        self.data['targets'] = tar_patches
        self.data['sources'] = src_patches

        if debug:
            output = tf.logging.info
            output('datamap : {}'.format(datamap.shape))
            output('bboxes  : {}'.format(bboxes.shape))
            output('targets : {}'.format(tar_patches.shape))
            output('sources : {}'.format(src_patches.shape))

    def generate_stats(self):
        mean = np.mean(self.data['targets'])
        std = np.std(self.data['targets'])
        return mean, std

    def normalize_data(self, mean, std):
        for i in tqdm(range(len(self.data['targets'])), desc='Normalizing targets'):
            self.data['targets'][i] = (self.data['targets'][i] - mean) / std

        for i in tqdm(range(len(self.data['sources'])), desc='Normalizing sources'):
            self.data['sources'][i] = (self.data['sources'][i] - mean) / std

    def __iter__(self):
        return self

    def __next__(self):
        if self.n_samples == 0 or self.sample_idx >= self.n_samples:
            self.sample_idx = 0
            ind = np.arange(self.n_samples)
            np.random.shuffle(ind)
            self.data['datamap'] = self.data['datamap'][ind]
            self.data['bboxes'] = self.data['bboxes'][ind]
            raise StopIteration

        tar_id, min_src_id, max_src_id = self.data['datamap'][self.sample_idx]
        src_id = np.random.randint(min_src_id, max_src_id, dtype=np.int32)
        bbox = self.data['bboxes'][self.sample_idx]

        tar = self.data['targets'][tar_id]
        src = self.data['sources'][src_id]
        labels, norm_bbox = to_one_shot_labels(bbox, src.shape[:2], self.cellsz)
        labels = labels[0]
        norm_bbox = norm_bbox[0]

        self.sample_idx += 1
        return src, tar, labels, norm_bbox

def input_fn(
    base_dir,
    cellsz=(16,16), n_parameters=5,
    src_size=(256, 256), tar_size=(128, 128), n_channels=1,
    batch_size=16
):
    data_sampler = FTFYPatchDataSampler(base_dir, cellsz)
    dataset = (
        tf.data.Dataset()
            .from_generator(generator=lambda: data_sampler,
                            output_types=(tf.float32, tf.float32, tf.float32, tf.float32),
                            output_shapes=(
                                [*src_size, n_channels],
                                [*tar_size, n_channels],
                                [cellsz[0]*cellsz[1], n_parameters],
                                [n_parameters]
                            ))
            .shuffle(buffer_size=2*batch_size)
            .batch(batch_size=batch_size)
            .prefetch(1)
    )
    return dataset, data_sampler

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from network.loss.ftfy import loss

    base_dir = '/home/sungsooha/Desktop/Data/ftfy/sem/train'
    project_dir = 'sources_shift' # ['sources', 'sources_square', 'sources_shift']
    batch_size = 4

    data_dirs = []
    for data_dir in os.listdir(base_dir):
        if os.path.isdir(os.path.join(base_dir, data_dir)):
            data_dirs.append(data_dir)
    data_dirs = sorted(data_dirs)

    with tf.device('/cpu:0'), tf.name_scope('input'):
        dataset, data_sampler = input_fn(base_dir, batch_size=batch_size)
        data_iterator = tf.data.Iterator.from_structure(
            dataset.output_types,
            dataset.output_shapes
        )
        dataset_init = data_iterator.make_initializer(dataset)
        batch_data = data_iterator.get_next()

    data_sampler.load_dataset(data_dirs, project_dir, debug=True)

    logits = tf.placeholder(tf.float32, (batch_size, 16, 16, 5*2))
    obj_loss, noobj_loss, coord_loss = loss(
        logits, batch_data[2], 2, 5
    )

    fig, ax = plt.subplots(2, batch_size)
    h_src, h_tar, rect = [], [], []
    for _ax in ax[0]:
        h = _ax.imshow(np.zeros((256, 256), dtype=np.float32), cmap='gray', vmin=0, vmax=1)
        h_src.append(h)
    for _ax in ax[1]:
        h = _ax.imshow(np.zeros((128, 128), dtype=np.float32), cmap='gray', vmin=0, vmax=1)
        h_tar.append(h)
    for _ax in ax[0]:
        _rect = Rectangle((0,0), 256, 256, linewidth=1, edgecolor='r', facecolor='none')
        _ax.add_patch(_rect)
        rect.append(_rect)

    plt.ion()
    plt.show()

    max_tests = 10
    i_test = 0

    with tf.Session() as sess:
        sess.run(dataset_init)
        try:
            while i_test < max_tests:
                sources, targets, labels, bboxes = sess.run(
                    [*batch_data],
                    feed_dict={
                        logits: np.zeros((batch_size, 16, 16, 10), dtype=np.float32)
                    }
                )
                i_test+=1

                for idx in range(len(sources)):
                    src = sources[idx]
                    tar = targets[idx]
                    label = labels[idx]
                    bbox = bboxes[idx]

                    h_src[idx].set_data(np.squeeze(src))
                    h_tar[idx].set_data(np.squeeze(tar))

                    #print(bbox)
                    _, cx, cy, w, h = bbox * 256
                    rect[idx].set_xy((cx - 0.5 - w/2, cy - 0.5 - h/2))
                    rect[idx].set_width(w)
                    rect[idx].set_height(h)

                    _bbox = to_bboxes(label)
                    print('bbox: ', [cx - 0.5, cy - 0.5, w, h])
                    print('from label: ', _bbox)



                plt.pause(5)


        except tf.errors.OutOfRangeError:
            pass
















