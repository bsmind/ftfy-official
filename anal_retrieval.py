import os
import pickle
import numpy as np
from network.test import TripletNet
from skimage.io import imsave, imread
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from tqdm import tqdm

import matplotlib.pyplot as plt

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

def load_retrieval_set(base_dir, dir_name, fname='retrieval.txt'):
    retrieval_fname = os.path.join(base_dir, dir_name, fname)
    assert os.path.isfile(retrieval_fname), 'Cannot find %s!' % fname

    patch_info = []
    labels = []


    with open(retrieval_fname, 'r') as f:
        for line in tqdm(f, 'Load retrieval set'):
            # patch_idx: patch index in the associated patch dataset
            # label: assigned labels for this patch
            # is_query: if 1, use this patch as a query example during retrieval test
            patch_idx, label, is_query = line.split()
            patch_idx = int(patch_idx)
            is_query = int(is_query)

            # from patch index, we can calculate more specific information about the patch
            # and this will be used for analysis.
            gid = patch_idx // 78 # each group contains 78 (13*6) patches
            irow = (patch_idx%78) // 6 # can calculate which IoU set it belongs to
            icol = (patch_idx%78) % 6  # can calculate down sampling factor

            if irow == 0:
                iou = 0 # key
            elif irow < 5:
                iou = 1 # <=0.7
            elif irow < 10:
                iou = 2 # <=0.5
            else:
                iou = 3 # <=0.3

            patch_info.append([patch_idx, gid, iou, icol, is_query])
            labels.append(label)

    return np.asarray(patch_info), np.array(labels)

def load_fnames(base_dir, data_dir, ext='bmp'):
    files = []
    dataset_dir = os.path.join(base_dir, data_dir)
    for f in os.listdir(dataset_dir):
        if f.endswith(ext):
            files.append(os.path.join(dataset_dir, f))
    return sorted(files)

def load_patches(
        base_dir, data_dir,
        patch_size=128, n_channels=1,
        patches_per_row=13,
        patches_per_col=6
):
    fnames = load_fnames(base_dir, data_dir)
    n_groups = len(fnames)

    n_patches = n_groups * patches_per_col * patches_per_row
    all_patches = np.zeros((n_patches, patch_size, patch_size, n_channels), dtype=np.float32)
    count = 0
    for f in tqdm(fnames, desc='Loading patches %s' % data_dir):
        assert os.path.isfile(f), 'Not a file: %s' % f

        im = imread(f) / 255.
        patches_row = np.split(im, patches_per_row, axis=0)
        for row in patches_row:
            patches = np.split(row, patches_per_col, axis=1)
            for patch in patches:
                patch_tensor = patch.reshape(patch_size, patch_size, n_channels)
                all_patches[count] = patch_tensor
                count += 1

                if count == n_patches: break
            if count == n_patches: break

    return all_patches

def save_patches(patches, base_dir, data_dir, patches_per_row=10, patches_per_col=10):

    output_dir = os.path.join(base_dir, data_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    def _save_patches(count, patchset_row):
        patchset = np.vstack(patchset_row)
        patchset *= 255.
        patchset = patchset.astype(np.uint8)

        # save
        fname = os.path.join(output_dir, 'patchset_{:06d}.bmp'.format(count))
        imsave(fname, patchset)

        return count + 1

    n_patches = len(patches)
    patchset_row = []
    count = 0

    for i in range(0, n_patches, patches_per_col):
        j = min(n_patches, i+patches_per_col)
        patch_row = np.squeeze(patches[i:j])
        patchset_row.append(np.hstack(patch_row))

        if len(patchset_row) == patches_per_row:
            count = _save_patches(count, patchset_row)
            patchset_row = []

def get_feature(model_path, patches):
    # todo: load mean, std with pickle
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
    net = TripletNet(model_path, **model_kwargs)
    return net.get_feature(patches, 128)

def compute_retrieval_precision(labels, knn_ind):
    n_queries = len(knn_ind)

    n_success = 0
    success = []
    for ind in knn_ind:
        q_label = labels[ind[0]]
        r_labels = labels[ind[1:]]

        is_success = q_label in r_labels
        n_success += int(is_success)
        success.append(int(is_success))
    return n_success / n_queries, np.array(success)

if __name__ == '__main__':
    INFO_IDX = 0
    INFO_GID = 1
    INFO_IOU = 2
    INFO_DOWN = 3
    INFO_Q = 4

    N_SCALES = 6

    base_dir = '/home/sungsooha/Desktop/Data/ftfy/austin'
    data_dir = 'campus_patch'
    model_path = './log/campus/ckpt'
    retrieval_data_dir = 'campus_retrieval'

    nbrs_name = 'campus_campus'
    n_neighbors = 5

    # ------------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------------
    patch_info, patch_labels = load_retrieval_set(base_dir, data_dir)

    if retrieval_data_dir is None:
        patches = load_patches(base_dir, data_dir)
        patches = patches[patch_info[:, 0]]
        save_patches(
            patches,
            base_dir,
            'campus_retrieval'
        )
    else:
        patches = load_patches(base_dir, retrieval_data_dir,
                               patches_per_row=10, patches_per_col=10)

    feature_path = os.path.join(base_dir, retrieval_data_dir, 'features.npy')
    if os.path.exists(feature_path):
        features = np.load(feature_path)
    else:
        features = get_feature(model_path, patches)
        np.save(feature_path, features)

    # ------------------------------------------------------------------------
    # NN
    # ------------------------------------------------------------------------
    knn_path = os.path.join(base_dir, retrieval_data_dir, 'knn_{:s}_{:d}.npy'.format(
        nbrs_name, n_neighbors
    ))
    if os.path.exists(knn_path):
        knn_ind = np.load(knn_path)
    else:
        nbrs = NearestNeighbors(
            n_neighbors=n_neighbors + 1,
            algorithm='ball_tree',
            metric='euclidean'
        ).fit(features)

        query_ind, = np.where(patch_info[:,INFO_Q])
        _, knn_ind = nbrs.kneighbors(features[query_ind, :])
        np.save(knn_path, knn_ind)

    # ------------------------------------------------------------------------
    # t-SNE
    # ------------------------------------------------------------------------
    tsne_path = os.path.join(base_dir, retrieval_data_dir, 'tsne.npy')
    if os.path.exists(tsne_path):
        t_features = np.load(tsne_path)
    else:
        n_samples = len(features)
        n_components = 2
        perplexities = 50
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexities,
            init='random',
            random_state=0,
            verbose=1
        )
        t_features = tsne.fit_transform(features)
        np.save(tsne_path, t_features)

    # ------------------------------------------------------------------------
    # Retrieval precision
    # ------------------------------------------------------------------------
    r_precision, is_passed = compute_retrieval_precision(patch_labels, knn_ind)
    print('Retrieval precision: {:.3f}'.format(r_precision))

    # ------------------------------------------------------------------------
    # Histogram analysis
    # - what is the distribution of multi-scale factor
    #   - in the dataset ?
    #   - for the query examples?
    #   - for the query example in success or fail?
    # ------------------------------------------------------------------------
    def _get_scale_histogram(down_factors):
        hist = np.zeros(N_SCALES, np.int32)
        for f in down_factors: hist[f] += 1
        return hist
    hist_overall = _get_scale_histogram(patch_info[:, INFO_DOWN])
    hist_query   = _get_scale_histogram(patch_info[patch_info[:, INFO_Q]==1, INFO_DOWN])

    ind, = np.where(is_passed==1)
    hist_passed  = _get_scale_histogram(patch_info[ind, INFO_DOWN])

    ind, = np.where(is_passed==0)
    hist_failed  = _get_scale_histogram(patch_info[ind, INFO_DOWN])

    print('hist overall: ', hist_overall)
    print('hist query  : ', hist_query)
    print('hist passed : ', hist_passed)
    print('hist failed : ', hist_failed)

    fig, ax = plt.subplots(2, 2)
    ax = ax.ravel()
    ax[0].bar(range(1, N_SCALES+1), hist_overall)
    ax[1].bar(range(1, N_SCALES+1), hist_query)
    ax[2].bar(range(1, N_SCALES+1), hist_passed)
    ax[3].bar(range(1, N_SCALES+1), hist_failed)
    plt.show()

    # ------------------------------------------------------------------------
    # Feature space analysis
    # - map to low dimension using t-SNE
    # - visualize points with interactive scatter plot
    # - interactions:
    #       - click on a point: show image
    # ------------------------------------------------------------------------
    shape_scheme = ["o", "^", "s", "X"]  # [circle, triangle, square, X]
    def _get_scatter_scheme(gIDs, iouIDs, scaleIDs):
        base_size = 20
        size_scheme = [base_size*np.sqrt(2**n) for n in range(N_SCALES)]
        size_scheme.reverse()

        counter = 0
        color_scheme = dict()
        for gid in np.unique(gIDs):
            color_scheme[gid] = counter
            counter += 1

        colors, shapes, sizes = [], [], []
        for gid, iou, sc in zip(gIDs, iouIDs, scaleIDs):
            colors.append(color_scheme[gid])
            shapes.append(shape_scheme[iou])
            sizes.append(size_scheme[sc])

        return np.array(colors), np.array(shapes), np.array(sizes)


    p_colors, p_shapes, p_sizes = _get_scatter_scheme(
        patch_info[:, INFO_GID], patch_info[:, INFO_IOU], patch_info[:, INFO_DOWN]
    )
    cmap = discrete_cmap(len(np.unique(p_colors)), 'cubehelix')

    # filter patches acturall included in knn

    fig, ax = plt.subplots()
    for shape in shape_scheme:
        ind, = np.where(p_shapes == shape)
        ax.scatter(
            t_features[ind, 0], t_features[ind, 1],
            s=p_sizes[ind], c=p_colors[ind],
            marker=shape, cmap=cmap
        )

    plt.show()

