import os
import logging
import warnings
import pickle
import numpy as np
from skimage.io import imread
from skimage.transform import rescale
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from network.test import FTFYNet
from test_triplet import TestData, calc_iou

if __name__ == '__main__':
    base_dir = '/home/sungsooha/Desktop/Data/ftfy/sem/train'
    result_dir = 'test_ftfy_100'
    load_dir = 'test_triplet'

    batch_size = 512
    top_K      = 5
    iou_threshold = 0.7

    # FTFY arguments
    TAR_PATCH_SIZE = 128
    SRC_PATCH_SIZE = 256
    log_dir = './log/sem_ftfy_full2'
    model_path = os.path.join(log_dir, 'ckpt')
    epoch = None
    mean, std = pickle.load(open(os.path.join(log_dir, 'stats_sem.pkl'), 'rb'))

    # logging
    logLevel = 'INFO'
    logFormat = '%(asctime)s:%(levelname)s:%(message)s'
    logFile = 'test.log'
    logging.basicConfig(level=logLevel, format=logFormat,
                        filename=os.path.join(base_dir, logFile))
    log = logging.getLogger('FTFY_TEST')

    # ------------------------------------------------------------------------
    # fetch available data directories
    # ------------------------------------------------------------------------
    data_dirs = []
    for f in os.listdir(base_dir):
        if os.path.isdir(os.path.join(base_dir,f)):
            data_dirs.append(f)
    data_dirs = sorted(data_dirs)
    assert len(data_dirs) > 0, 'No available data directories.'

    # ------------------------------------------------------------------------
    # Network
    # ------------------------------------------------------------------------
    net = FTFYNet(model_path, epoch,
                  mean=mean, std=std,
                  src_size=(SRC_PATCH_SIZE, SRC_PATCH_SIZE),
                  src_cell_size=(SRC_PATCH_SIZE//16, SRC_PATCH_SIZE//16))

    # ------------------------------------------------------------------------
    # Setup matplotlib for visualization
    # ------------------------------------------------------------------------
    bbox_fig, bbox_ax = plt.subplots(1, 1)
    h1 = bbox_ax.imshow(np.zeros((850, 1280), dtype=np.float32), cmap='gray', vmin=0, vmax=1)
    bbox_ax.axis('off')
    q_rect = Rectangle((0, 0), TAR_PATCH_SIZE, TAR_PATCH_SIZE,
                       linewidth=1, edgecolor='r', facecolor='none')
    k_rect = [
        Rectangle((0, 0), TAR_PATCH_SIZE, TAR_PATCH_SIZE,
                  linewidth=1, edgecolor='y', facecolor='none') for _ in range(top_K)
    ]
    [bbox_ax.add_patch(_rect) for _rect in [q_rect, *k_rect]]

    topk_fig, topk_ax = plt.subplots()
    h2 = topk_ax.imshow(np.zeros((TAR_PATCH_SIZE, 6*TAR_PATCH_SIZE), dtype=np.float32),
                   cmap='gray', vmin=0, vmax=1)
    topk_ax.axis('off')
    topk_q_rect = Rectangle((0, 0), TAR_PATCH_SIZE-1, TAR_PATCH_SIZE-2,
                       linewidth=1, edgecolor='r', facecolor='none')
    topk_1_rect = Rectangle((0, 0), TAR_PATCH_SIZE-1, TAR_PATCH_SIZE-2,
                       linewidth=1, edgecolor='y', facecolor='none')
    [topk_ax.add_patch(_rect) for _rect in [topk_q_rect, topk_1_rect]]

    bbox_fig.tight_layout()
    topk_fig.tight_layout()

    plt.ion()
    plt.show()

    # ------------------------------------------------------------------------
    # Test dataset
    # ------------------------------------------------------------------------
    n_total_test = 0
    n_success_k  = 0
    n_success_1  = 0

    for data_dir in tqdm(data_dirs, desc='TEST FTFY'):
        output_dir = os.path.join(base_dir, data_dir, result_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        fig_output_dir_f = os.path.join(base_dir, data_dir, result_dir, 'images/failed')
        if not os.path.exists(fig_output_dir_f):
            os.makedirs(fig_output_dir_f)

        fig_output_dir_s = os.path.join(base_dir, data_dir, result_dir, 'images/success')
        if not os.path.exists(fig_output_dir_s):
            os.makedirs(fig_output_dir_s)

        # we use same dataset used for triplet network
        dataset = TestData(base_dir, data_dir, os.path.join(base_dir, data_dir, load_dir))

        log.info('-' * 50)
        log.info('TEST with {:s} dataset'.format(data_dir))

        l_n_tests = 0
        l_n_success_k = 0
        l_n_success_1 = 0
        for fid, fname in enumerate(sorted(dataset.fnames)):
            img = imread(fname, as_gray=True)
            img = img[:-110, :]

            for q_idx in sorted(dataset.fileMap[fname]):
                cx, cy, side, down_factor = dataset.info[q_idx]
                x0 = cx - side//2
                y0 = cy - side//2

                # ----------------------------------------------------------------
                # extract query image
                q_im = img[y0:y0+side, x0:x0+side]
                q_bbox = [x0, y0, x0+side, y0+side]
                q_bbox = np.array([q_bbox], dtype=np.float32)

                # ----------------------------------------------------------------
                # sample db image coordinate
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    sImg = rescale(img, 1/down_factor)
                    #sImg = (sImg - sImg.min()) / sImg.ptp()

                # ----------------------------------------------------------------
                # predict bounding boxes
                im_tar = np.expand_dims(np.expand_dims(q_im, -1), 0)

                s_h, s_w = sImg.shape
                # In usages of the FTFY network, the source size is the matter
                src_size = int((side // down_factor) * 4.5) #SRC_PATCH_SIZE // down_factor
                sx0 = np.arange(0, s_w - src_size, src_size//2)
                sy0 = np.arange(0, s_h - src_size, src_size//2)

                if s_w - sx0[-1] > side / down_factor:
                    sx0 = np.append(sx0, s_w - src_size - 1)

                if s_h - sy0[-1] > side / down_factor:
                    sy0 = np.append(sy0, s_h - src_size - 1)

                sx0, sy0 = np.meshgrid(sx0, sy0)
                sx0 = sx0.flatten()
                sy0 = sy0.flatten()

                pred_confidence = []
                pred_bboxes = []
                for src_x0, src_y0 in zip(sx0, sy0):
                    im_src = sImg[src_y0:src_y0+src_size, src_x0:src_x0+src_size]
                    im_src = np.expand_dims(np.expand_dims(im_src, -1), 0)

                    confidence, bboxes = net.run(im_src, im_tar, top_K)

                    bboxes[..., 0] = bboxes[..., 0] * src_size - 0.5 + src_x0
                    bboxes[..., 1] = bboxes[..., 1] * src_size - 0.5 + src_y0
                    bboxes[..., 2] *= src_size
                    bboxes[..., 3] *= src_size

                    pred_confidence.append(np.squeeze(confidence))
                    pred_bboxes.append(np.squeeze(bboxes))

                pred_confidence = np.hstack(pred_confidence)
                pred_bboxes = np.vstack(pred_bboxes)

                ind = np.argsort(pred_confidence)[::-1]
                pred_confidence = pred_confidence[ind[:top_K]]
                pred_bboxes = pred_bboxes[ind[:top_K]]
                pred_bboxes = np.stack([
                    pred_bboxes[..., 0] - pred_bboxes[..., 2]/2,
                    pred_bboxes[..., 1] - pred_bboxes[..., 2]/2,
                    pred_bboxes[..., 0] + pred_bboxes[..., 3]/2,
                    pred_bboxes[..., 1] + pred_bboxes[..., 3]/2
                ], axis=-1) * down_factor

                iou = calc_iou(q_bbox, pred_bboxes)

                # ----------------------------------------------------------------
                # Collect output and visualize
                retrieve_test = np.array(iou > iou_threshold, dtype=np.int32)

                is_success = retrieve_test.sum() > 0
                l_n_tests += 1
                l_n_success_k += int(is_success)
                l_n_success_1 += retrieve_test[0]

                log.info("data set         : %s" % data_dir)
                log.info("fname            : %s" % fname)
                log.info("ID               : %s" % q_idx)
                log.info("is success       : {:s}".format('YES' if is_success else 'NO'))
                log.info("down factor      : %s", down_factor)
                log.info("expected bbox    : {}".format(q_bbox))
                log.info("retrieved bbox   :\n {}".format(pred_bboxes))
                log.info("retrieved iou    : {}".format(iou))
                log.info("retrieved conf   : {}".format(pred_confidence*100))
                log.info("retrieve test    : {}".format(retrieve_test))
                log.info("\n")

                #print(q_idx, side, src_size, down_factor, is_success)

                # start visualization --------------------------------------------
                img = (img - img.min()) / img.ptp()
                h1.set_data(img)
                for bbox in q_bbox:
                    q_rect.set_xy((bbox[0], bbox[1]))
                    q_rect.set_width(bbox[2] - bbox[0])
                    q_rect.set_height(bbox[3] - bbox[1])

                for it, bbox in enumerate(pred_bboxes):
                    k_rect[it].set_xy((bbox[0], bbox[1]))
                    k_rect[it].set_width(bbox[2] - bbox[0])
                    k_rect[it].set_height(bbox[3] - bbox[1])

                patches = [np.squeeze(net.get_input(im_tar, False))]
                for bbox in pred_bboxes:
                    bbox = np.round(bbox / down_factor).astype(np.int32)
                    x0, y0, x1, y1 = bbox
                    x0 = np.clip(x0, 0, s_w-1)
                    y0 = np.clip(y0, 0, s_h-1)
                    x1 = np.clip(x1, 0, s_w-1)
                    y1 = np.clip(y1, 0, s_h-1)
                    _im = np.expand_dims(np.expand_dims(sImg[y0:y1, x0:x1], -1), 0)
                    patches.append(np.squeeze(net.get_input(_im, False)))

                patches = [(_im - _im.min()) / _im.ptp() for _im in patches]
                patches = np.hstack(patches)
                h2.set_data(patches)
                idx = int(np.argmax(iou)) + 1
                topk_1_rect.set_xy((idx*TAR_PATCH_SIZE, 0))

                plt.pause(0.001)
                plt.draw()

                fig_output_dir = fig_output_dir_s if is_success else fig_output_dir_f
                bbox_fig.savefig(os.path.join(fig_output_dir, 'bbox_{:03d}_{:03d}_{:d}.png'.format(
                    fid, q_idx, int(is_success)
                )))
                topk_fig.savefig(os.path.join(fig_output_dir, 'topk_{:03d}_{:03d}_{:d}.png'.format(
                    fid, q_idx, int(is_success)
                )))

        #print(l_n_tests, l_n_success_k, l_n_success_1)
        log.info('-'*50)
        log.info('TEST Result     :')
        log.info('data set        : %s' % data_dir)
        log.info('# total         : %s' % l_n_tests)
        log.info('# success @ {:d}: {:d}'.format(top_K, l_n_success_k))
        log.info('# success @ {:d}: {:d}'.format(1, l_n_success_1))
        log.info('avg. retrieval @ {:d}: {:.3f}'.format(top_K, l_n_success_k / l_n_tests))
        log.info('avg. retrieval @ {:d}: {:.3f}'.format(1, l_n_success_1 / l_n_tests))

        n_total_test += l_n_tests
        n_success_k += l_n_success_k
        n_success_1 += l_n_success_1

    log.info('-'*50)
    log.info('Final result  :')
    log.info('# total       : %s' % n_total_test)
    log.info('# success @ {:d}: {:d}'.format(top_K, n_success_k))
    log.info('# success @ {:d}: {:d}'.format(1, n_success_1))
    log.info('avg. retrieval @ {:d}: {:.3f}'.format(top_K, n_success_k / n_total_test))
    log.info('avg. retrieval @ {:d}: {:.3f}'.format(1, n_success_1 / n_total_test))



















