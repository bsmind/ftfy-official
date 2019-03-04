import os
import numpy as np

from skimage.io import imread, imsave

from utils.data import get_filenames
from utils.sem_data import PatchExtractor, PatchDataManager
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Rectangle
from matplotlib.image import AxesImage

from tqdm import tqdm

def plt_measure_px_size(base_dir, data_dir):
    h_im, h_bar = None, None
    px_size_path = os.path.join(base_dir, data_dir, 'pxsize.txt')
    file = open(px_size_path, 'w')
    for f in os.listdir(os.path.join(base_dir, data_dir)):
        if not f.endswith('tif'): continue

        fname = os.path.join(base_dir, data_dir, f)
        im = imread(fname, as_gray=True)
        im = im * 255
        im = im.astype(np.uint8)
        im_h, im_w = im.shape

        scale_bar = im.copy()
        scale_bar[:im_h-110, :] = 0
        scale_bar = scale_bar == 255

        _, cols = np.where(scale_bar)
        longest_seq = max(np.split(cols, np.where(np.diff(cols) != 1)[0]+1), key=len).tolist()
        scale_bar[:, :longest_seq[0]] = 0
        scale_bar[:, longest_seq[-1]+1:] = 0

        if h_im is None:
            h_im = plt.imshow(im, cmap='gray')
            h_bar = plt.imshow(scale_bar, alpha=0.5)
            plt.ion()
            plt.show()
        else:
            h_im.set_data(im)
            h_bar.set_data(scale_bar)
        plt.gca().set_title(f)
        plt.pause(0.001)


        bar_width = longest_seq[-1] - longest_seq[0] + 1
        bar_size = int(input('[{:s}] bar size: '.format(f)))

        print('File: ', f)
        print('Bar width  (px): ', bar_width)
        print('Bar size   (um): ', bar_size)
        print('Pixel size (um): ', bar_size / bar_width)
        file.write("{:s} {:d} {:d}\n".format(f, bar_width, bar_size))
    file.close()
    if h_im is not None:
        plt.ioff()
        plt.close()

def load_px_size(base_dir, data_dir):
    px_size_path = os.path.join(base_dir, data_dir, 'pxsize.txt')
    if not os.path.exists(px_size_path):
        plt_measure_px_size(base_dir, data_dir)

    fnames = []
    px_sizes = []
    file = open(px_size_path, 'r')
    for line in file:
        fname, bar_width, bar_size = line.split()
        px_size = float(bar_size) / float(bar_width)
        fnames.append(fname)
        px_sizes.append(px_size)

    fnames = np.array(fnames)
    px_sizes = np.array(px_sizes)
    ind = np.argsort(px_sizes)
    return fnames[ind], px_sizes[ind]

def generate_info(fname, pid, gid, iouid, cx, cy, side, down_factor):
    return "{:s} {:d} {:d} {:d} {:d} {:d} {:d} {:d}\n".format(
        fname, pid, gid, iouid, cx, cy, side, down_factor
    )



if __name__ == '__main__':
    base_dir = '/home/sungsooha/Desktop/Data/ftfy/sem'
    data_dir = 'train/set_065'
    output_dir = os.path.join(base_dir, data_dir, 'patches')
    do_measure_px_size = True
    has_keypoint = True

    down_factors = [1, 2, 4, 6, 8, 10]
    iou_ranges = [(0.7, 1.0), (0.5, 0.7), (0.3, 0.5)]
    n_iou_samples = 3

    patch_size = 128
    min_patch_size = 13

    # patch extractor
    patch_extractor = PatchExtractor(
        down_factors, iou_ranges, patch_size, min_patch_size)

    # patch data manager
    patchdata_mgr = PatchDataManager(
        output_dir, patches_per_col=10, patches_per_row=10,
        info_fname='info.txt'
    )

    patch_counter = 0
    group_counter = 0
    def make_info(fname, pid, gid, iouid, cx, cy, side, _down_factors):
        all_info = []
        for factor in _down_factors:
            info_str = generate_info(fname, pid, gid, iouid, cx, cy, side, factor)
            all_info.append(info_str)
            pid += 1
        return pid, all_info

    # ------------------------------------------------------------------------
    # load pixel size information
    # ------------------------------------------------------------------------
    filenames = []
    filenames += get_filenames(os.path.join(base_dir, data_dir), 'tif')
    n_filenames = len(filenames)
    assert n_filenames > 0, 'No files to process in the directory: {:s}.'.format(base_dir)
    filenames = sorted(filenames)
    # ------------------------------------------------------------------------
    # Load all image in the data_dir
    # ------------------------------------------------------------------------
    def load_image(fname):
        im = imread(fname, as_gray=True)
        im = im[:-110, :]
        return im
    # images = []
    # for fname in tqdm(filenames, desc='Load images'):
    #     im = imread(fname, as_gray=True)
    #     im = im[:-110, :] # to remove scale bar at the bottom of SEM image
    #     images.append(im)


    # ------------------------------------------------------------------------
    # Matplotlib GUI for patch extraction
    # ------------------------------------------------------------------------
    is_busy = False
    is_active = True
    picked_idx = 5
    image_idx = 0
    x1, y1, x2, y2 = None, None, None, None
    main_fig, main_ax = plt.subplots()
    sub_fig, sub_ax = plt.subplots(2, 3)
    sub_ax = sub_ax.ravel()

    # initialize
    image = load_image(filenames[image_idx])
    main_h = main_ax.imshow(image, cmap='gray', vmin=0, vmax=1)
    main_ax.set_title('{:d}/{:d}'.format(image_idx+1, len(filenames)))
    sub_h = []
    sub_marker = []
    for ax in sub_ax:
        h = ax.imshow(np.zeros((patch_size, patch_size), dtype=np.float32),
                      cmap='gray', vmin=0, vmax=1, picker=True)
        marker = Rectangle((1,1), 5, 5, facecolor='r', edgecolor='none')
        marker.set_visible(False)
        ax.add_patch(marker)
        sub_marker.append(marker)
        ax.axis('off')
        sub_h.append(h)
    sub_marker[picked_idx].set_visible(True)

    def on_extract_ms_patches():
        global x1, y1, x2, y2
        if None in [x1, y1, x2, y2]: return
        patch_extractor.extract_ms_patches(image, (x1, y1, x2, y2), n_iou_samples)
        patches, factors, box = patch_extractor.get_patches_with_info()
        for ax, h, patch, factor in zip(sub_ax, sub_h, patches, factors):
            h.set_data(patch)
            ax.set_title('{:.3f}'.format(factor))
        plt.pause(0.0001)

    def on_save_ms_patches():
        global group_counter, patch_counter
        patches, factors, box = patch_extractor.get_patches_with_info()
        patches_iou, _, boxes_iou = patch_extractor.get_patches_iou_with_info()

        factors = factors[:picked_idx+1]
        patches = patches[:picked_idx+1]

        ind = np.arange(len(patches_iou), dtype=np.int32) % len(down_factors)
        ind, = np.where(ind < (picked_idx+1))
        patches_iou = [patches_iou[i] for i in ind]

        # save key patches
        patch_counter, all_info = make_info(
            filenames[image_idx], patch_counter, group_counter, 0,
            box[0], box[1], box[2], factors
        )
        patchdata_mgr.add(patches[:picked_idx+1], all_info)


        rect = Rectangle(
            (box[0] - box[2]//2, box[1] - box[2]//2),
            box[2], box[2], edgecolor='red', facecolor='none'
        )
        main_ax.add_patch(rect)

        # save iou patches
        all_info_iou = []
        for i_iou in range(len(iou_ranges)):
            for i_sample in range(n_iou_samples):
                box = boxes_iou[i_iou*n_iou_samples + i_sample]
                patch_counter, all_info = make_info(
                    filenames[image_idx], patch_counter, group_counter, i_iou+1,
                    box[0], box[1], box[2], factors
                )
                all_info_iou += all_info
        patchdata_mgr.add(patches_iou, all_info_iou)

        group_counter += 1
        plt.pause(0.001)

    def on_next_image():
        global image_idx, image
        image_idx = image_idx + 1
        if image_idx == len(filenames):
            patchdata_mgr.dump()
            print('end of images, terminate!')
            exit()
        image = load_image(filenames[image_idx])
        main_h.set_data(image)
        main_ax.set_title('{:d}/{:d}'.format(image_idx + 1, len(filenames)))

        [p.remove() for p in reversed(main_ax.patches)]

        for h in sub_h:
            h.set_data(np.zeros((patch_size, patch_size), dtype=np.float32))

        patch_extractor.clear()
        plt.pause(0.001)

    def on_select(eclick, erelease):
        if is_busy:
            print('Busy for processing something...')
            return
        global x1, y1, x2, y2
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

    def on_key(event):
        global is_busy
        if is_busy:
            print('Busy for processing something...')
        key = event.key
        if key == '1':
            print('extract patches')
            is_busy = True
            on_extract_ms_patches()
            is_busy = False
        elif key == '2':
            is_busy = True
            print('save patches')
            on_save_ms_patches()
            is_busy = False
        elif key == '3':
            is_busy = True
            print('next images')
            on_next_image()
            is_busy = False
        elif key == 'a':
            print('toogle selector')
            on_key.rs.set_active(not on_key.rs.get_active())

    def on_pick(event):
        global picked_idx
        artist = event.artist
        if isinstance(artist, AxesImage):
            idx = sub_h.index(artist)
            sub_marker[picked_idx].set_visible(False)
            sub_marker[idx].set_visible(True)
            picked_idx = idx
            plt.pause(0.001)

    on_key.rs = RectangleSelector(main_ax, on_select, button=[3],
                           minspanx=min_patch_size, minspany=min_patch_size,
                           spancoords='pixels', interactive=True, useblit=True)

    main_fig.canvas.mpl_connect('key_press_event', on_key)
    sub_fig.canvas.mpl_connect('pick_event', on_pick)
    plt.show()

    patchdata_mgr.dump()

