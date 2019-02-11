import os
import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)

from skimage.io import imread
from sklearn.manifold import TSNE

from sklearn.neighbors import NearestNeighbors

from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D # for legend (trick)

from network.test import TripletNet


def load_fnames(base_dir, data_dir, ext='bmp'):
    files = []
    dataset_dir = os.path.join(base_dir, data_dir)
    for f in os.listdir(dataset_dir):
        if f.endswith(ext):
            files.append(os.path.join(dataset_dir, f))
    return sorted(files)

def load_patches(
        base_dir, data_dir,
        max_n_groups=10,
        patch_size=128, n_channels=1,
        patches_per_row=13,
        patches_per_col=6
):
    fnames = load_fnames(base_dir, data_dir)
    if len(fnames) > max_n_groups:
        fnames = np.random.choice(fnames, max_n_groups, replace=False)

    # 1st row: multi-scaled patches at a key point
    # (2nd - 5th)   row: 0.7 <= iou < 1.0
    # (6th - 9th)   row: 0.5 <= iou < 0.7
    # (10th - 13th) row: 0.3 <= iou < 0.5
    row_to_fecth = [0, 1, 5, 9]

    all_patches = []
    labels = []

    all_grouped_patches = []
    gid = 0

    for f in tqdm(fnames, desc='Loading patches %s' % data_dir):
        assert os.path.isfile(f), 'Not a file: %s' % f

        im = imread(f) / 255.
        patches_row = np.split(im, patches_per_row, axis=0)
        grouped_patches = []
        # loop over different iou range (see above)
        for i_row, row in enumerate(row_to_fecth):
            patches = np.split(patches_row[row], patches_per_col, axis=1)
            grouped_patches.append(patches_row[row])
            # loop over different scales (high to low)
            for i_patch, patch in enumerate(patches):
                patch_tensor = patch.reshape(patch_size, patch_size, n_channels)
                label = '{:d}_{:d}_{:d}'.format(gid, i_row, i_patch)
                all_patches.append(patch_tensor)
                labels.append(label)
        all_grouped_patches.append(np.vstack(grouped_patches))
        gid += 1

    return np.asarray(all_patches), np.array(labels), all_grouped_patches

def get_network(model_path):
    model_kwargs = dict(
        mean=0.36552,
        std=0.23587,
        patch_size=128,
        n_channels=1,
        n_feats=128,
        cnn_name='ftfy',
        shared_batch_layers=True,
        name='triplet-net'
    )
    return TripletNet(model_path, **model_kwargs)

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


if __name__ == '__main__':
    # for reproduction
    np.random.seed(2019)

    base_dir = '/home/sungsooha/Desktop/Data/ftfy/austin'
    data_dir = 'campus_patch'
    model_path = './log/campus/ckpt'
    max_n_groups = 10 #10

    net = get_network(model_path)

    patches, labels, grouped_patches = load_patches(base_dir, data_dir, max_n_groups=max_n_groups)
    features = net.get_feature(patches)

    print('patches: ', patches.shape)
    print('labels: ', len(labels))
    print('features: ', features.shape)

    # ------------------------------------------------------------------------
    #tSNE
    # ------------------------------------------------------------------------
    n_samples = len(features)
    n_components = 2
    preplexities = 50

    tsne = TSNE(
        n_components=n_components,
        perplexity=preplexities,
        init='random',
        random_state=0
    )
    t_features = tsne.fit_transform(features)

    # ------------------------------------------------------------------------
    # NN
    # ------------------------------------------------------------------------
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(t_features)

    # ------------------------------------------------------------------------
    # scatter plot
    # ------------------------------------------------------------------------
    shape_scheme = ["o", "^", "s", "X"] # [circle, triangle, square, x]

    base_size = 20
    size_scheme = [base_size*np.sqrt(2**n) for n in range(6)]
    size_scheme.reverse()
    #size_scheme = [base_size] * 6

    colors = []
    shapes = []
    sizes  = []
    for label in labels:
        gid, iou, scale = label.split('_')
        colors.append(int(gid))
        shapes.append(shape_scheme[int(iou)])
        sizes.append(size_scheme[int(scale)])
    colors = np.array(colors)
    shapes = np.array(shapes)
    sizes = np.array(sizes)

    n_groups = len(np.unique(colors))
    if n_groups > 10:
        cmap = discrete_cmap(n_groups, 'cubehelix')
        #cmap = discrete_cmap(n_groups, 'nipy_spectral')
    else:
        cmap = 'tab10'

    # start scatter plot
    fig, ax = plt.subplots(1, 1)
    for shape in shape_scheme:
        ind, = np.where(shapes == shape)
        h = ax.scatter(
            t_features[ind, 0], t_features[ind, 1],
            s=sizes[ind], c=colors[ind],
            marker=shape, cmap=cmap,
            #edgecolors='black'
        )
        h.set_clim(-0.5, n_groups - 0.5)

    # for highlighted groups
    highlighted_ind = colors == 0
    highlighted_handlers = []
    for shape in shape_scheme:
        ind = shapes == shape
        ind = highlighted_ind & ind
        ind, = np.where(ind)
        h = ax.scatter(
            t_features[ind, 0], t_features[ind, 1],
            s=sizes[ind], c=colors[ind],
            marker=shape, cmap=cmap,
            edgecolors='red'
        )
        h.set_clim(-0.5, n_groups - 0.5)
        highlighted_handlers.append(h)

    # for highlighted point
    # hh_idx = 0
    # hh = ax.scatter(
    #     [t_features[hh_idx, 0]], [t_features[hh_idx, 1]],
    #     s=[sizes[hh_idx]], c=[colors[hh_idx]],
    #     marker=shapes[hh_idx], cmap=cmap,
    #     edgecolors='blue'
    # )


    iou_legend = [
        Line2D([0], [0], linewidth=0, marker=shape) for shape in shape_scheme
    ]
    scale_legend = [
        Line2D([0], [0], linewidth=0, marker='o', markersize=sz/10) for sz in size_scheme
    ]

    cbar = fig.colorbar(h, ax=ax, ticks=range(n_groups))
    iou_legend = ax.legend(iou_legend, ['key', '>0.7', '>0.5', '>0.3'], loc=1)
    scale_legend = ax.legend(scale_legend, ['1', '2', '4', '6', '8', '10'], loc=4)
    ax.add_artist(iou_legend)
    fig.tight_layout()
    # end scatter plot

    # start grouped patch visualization
    picked = 0
    fig2, ax2 = plt.subplots(1, 1)
    imh = ax2.imshow(grouped_patches[picked], cmap='gray')

    tick_start = 64
    tick_step = 128
    ax2.set_xticks([tick_start + i_tick*tick_step for i_tick in range(6)])
    ax2.set_xticklabels(['1', '2', '4', '6', '8', '10'])
    ax2.set_yticks([tick_start + i_tick*tick_step for i_tick in range(4)])
    ax2.set_yticklabels(['key', '>0.7', '>0.5', '>0.3'])

    h, w = grouped_patches[0].shape
    for i_line in range(3):
        ypos = (i_line+1)*128 - 1
        ax2.plot([0, w-1], [ypos, ypos], linewidth=1, color='black')
    for i_line in range(5):
        xpos = (i_line+1)*128 - 1
        ax2.plot([xpos, xpos], [0, h-1], linewidth=1, color='black')
    fig2.tight_layout()
    # end

    def on_pick(event):
        global picked
        val = event.mouseevent.ydata
        if picked != int(np.round(val)):
            picked = int(np.round(val))
            imh.set_data(grouped_patches[picked])
            fig2.canvas.draw()

            # for highlighted points
            highlighted_ind = colors == picked
            for handler, shape in zip(highlighted_handlers, shape_scheme):
                ind = shapes == shape
                ind = highlighted_ind & ind
                ind, = np.where(ind)

                handler.set_offsets(t_features[ind,:])
                handler._sizes = sizes[ind]
                handler.set_array(colors[ind])
            fig.canvas.draw()
    #
    # def on_btn_press(event):
    #     if event.inaxes != ax:
    #         return
    #     dist, ind = nbrs.kneighbors([[event.xdata, event.ydata]])
    #     dist = dist[0][0]
    #     ind = ind[0][0]
    #     if dist < 0.1:
    #         global hh_idx
    #         hh_idx = ind
    #         hh.set_offsets([t_features[hh_idx, :]])
    #         hh._sizes = [sizes[hh_idx]]
    #         hh.set_array(np.array([colors[hh_idx]]))
    #         fig.canvas.draw()


    # def on_btn_release(event):
    #     pass

    cbar.ax.set_picker(5)
    fig.canvas.mpl_connect('pick_event', on_pick)
    # fig.canvas.mpl_connect('button_press_event', on_btn_press)
    # fig.canvas.mpl_connect('button_release_event', on_btn_release)

    plt.show()

