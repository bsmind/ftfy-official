import os
import warnings
import pickle
import logging
import numpy as np
from skimage.io import imread
from skimage.transform import rescale

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from tqdm import tqdm

from network.test import TripletNet

def load_patch_info(
        base_dir, data_dir,
        patch_dir='patches', fname='info.txt',
        max_n_queries=50, max_side=100
):
    info_fname = os.path.join(base_dir, data_dir, patch_dir, fname)
    assert os.path.exists(info_fname), 'There is no file: %s' % info_fname

    all_fname, all_info = [], []
    with open(info_fname, 'r') as file:
        for line in file:
            tokens = line.split()
            if int(tokens[3]) > 0: continue

            info = np.array(tokens[4:]).astype(np.int32) # cx, cy, side, down factor
            if info[2] > max_side: continue

            all_fname.append(tokens[0])
            all_info.append(info)

    # randomly choose
    ind = np.arange(len(all_info))
    np.random.shuffle(ind)
    selected_fname = [all_fname[i] for i in ind[:max_n_queries]]
    selected_info = [all_info[i] for i in ind[:max_n_queries]]

    fileMap = dict()
    for i, fname in enumerate(selected_fname):
        if fileMap.get(fname, None) is None:
            fileMap[fname] = []
        fileMap[fname].append(i)

    return fileMap, selected_info

def calc_iou(bbox1, bbox2):
    lu = np.maximum(bbox1[..., :2], bbox2[..., :2])
    rb = np.minimum(bbox1[..., 2:], bbox2[..., 2:])

    inter = np.maximum(0, rb - lu)
    inter_area = inter[..., 0] * inter[..., 1]

    area1 = (bbox1[..., 2] - bbox1[..., 0]) * (bbox1[..., 3] - bbox1[..., 1])
    area2 = (bbox2[..., 2] - bbox2[..., 0]) * (bbox2[..., 3] - bbox2[..., 1])

    union_area = np.maximum(area1 + area2 - inter_area, 1e-10)
    iou = np.clip(inter_area / union_area, 0., 1.)

    return iou

class TestData(object):
    def __init__(self, base_dir:str, data_dir:str,
                 load_dir:str=None,
                 force_to_new:bool=False):
        loaded = False
        self.fileMap = None
        self.info = None
        self.fnames = None

        if load_dir is not None:
            has_filemap = os.path.exists(os.path.join(load_dir, 'filemap.txt'))
            has_info = os.path.exists(os.path.join(load_dir, 'info.txt'))
            if has_filemap and has_info:
                self.load(load_dir)
                loaded = True
                #print('loaded from %s' % load_dir)

        if not loaded or force_to_new:
            self.fileMap, self.info = load_patch_info(base_dir, data_dir)
            self.fnames = list(self.fileMap.keys())
            #print('New dataset is created.')

    def load(self, output_dir:str):
        self.fileMap = dict()
        n_examples = 0
        file = open(os.path.join(output_dir, 'filemap.txt'), 'r')
        for line in file:
            tokens = line.split()
            key = tokens[0]
            value = np.array(tokens[1:]).astype(np.int32)
            self.fileMap[key] = value.tolist()
            n_examples += len(self.fileMap.get(key, []))
        file.close()

        self.info = []
        file = open(os.path.join(output_dir, 'info.txt'), 'r')
        for line in file:
            tokens = line.split()
            info = np.array(tokens[1:]).astype(np.int32)
            self.info.append(info)
        file.close()

        self.fnames = list(self.fileMap.keys())
        assert n_examples == len(self.info), \
            'Mismatched number of items in filemap.txt and info.txt'


    def dump(self, output_dir:str):
        file = open(os.path.join(output_dir, 'filemap.txt'), 'w')
        for key, value in self.fileMap.items():
            value_str = ''
            for v in value:
                value_str += '{:d} '.format(v)
            file.writelines('{:s} {:s}\n'.format(key, value_str))
        file.close()

        file = open(os.path.join(output_dir, 'info.txt'), 'w')
        for i, (cx, cy, side, down_factor) in enumerate(self.info):
            file.writelines('{:d} {:d} {:d} {:d} {:d}\n'.format(
                i, cx, cy, side, down_factor
            ))
        file.close()

if __name__ == '__main__':
    base_dir = '/home/sungsooha/Desktop/Data/ftfy/sem/train'
    result_dir = 'test_triplet' # ['test_triplet', 'test_ftfy']

    batch_size = 512 # batch size for db construction
    top_K      = 5

    # TripletNet arguments
    PATCH_SIZE = 128
    log_dir = './log/sem2'
    model_path = os.path.join(log_dir, 'ckpt')
    epoch      = 20//5
    mean, std  = pickle.load(open(os.path.join(log_dir, 'stats_sem.pkl'), 'rb'))

    # logging
    logLevel = 'INFO'
    logFormat = '%(asctime)s:%(levelname)s:%(message)s'
    logFile = 'test.log'
    logging.basicConfig(level=logLevel, format=logFormat,
                        filename=os.path.join(base_dir, logFile))
    log = logging.getLogger('TRIPLET_TEST')

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
    net = TripletNet(model_path, epoch,
                     mean=mean, std=std, cnn_name='ftfy', trainable=False)

    # ------------------------------------------------------------------------
    # Setup matplotlib for visualization
    # ------------------------------------------------------------------------
    bbox_fig, bbox_ax = plt.subplots(1, 1)
    h1 = bbox_ax.imshow(np.zeros((850, 1280), dtype=np.float32), cmap='gray', vmin=0, vmax=1)
    bbox_ax.axis('off')
    q_rect = Rectangle((0, 0), PATCH_SIZE, PATCH_SIZE,
                       linewidth=1, edgecolor='r', facecolor='none')
    k_rect = [
        Rectangle((0, 0), PATCH_SIZE, PATCH_SIZE,
                  linewidth=1, edgecolor='y', facecolor='none') for _ in range(top_K)
    ]
    [bbox_ax.add_patch(_rect) for _rect in [q_rect, *k_rect]]

    topk_fig, topk_ax = plt.subplots()
    h2 = topk_ax.imshow(np.zeros((PATCH_SIZE, 6*PATCH_SIZE), dtype=np.float32),
                   cmap='gray', vmin=0, vmax=1)
    topk_ax.axis('off')
    topk_q_rect = Rectangle((0, 0), PATCH_SIZE-1, PATCH_SIZE-2,
                       linewidth=1, edgecolor='r', facecolor='none')
    topk_1_rect = Rectangle((0, 0), PATCH_SIZE-1, PATCH_SIZE-2,
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

    for data_dir in tqdm(data_dirs, desc='TEST TRIPLET'):
        output_dir = os.path.join(base_dir, data_dir, result_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        fig_output_dir_f = os.path.join(base_dir, data_dir, result_dir, 'images/failed')
        if not os.path.exists(fig_output_dir_f):
            os.makedirs(fig_output_dir_f)

        fig_output_dir_s = os.path.join(base_dir, data_dir, result_dir, 'images/success')
        if not os.path.exists(fig_output_dir_s):
            os.makedirs(fig_output_dir_s)

        dataset = TestData(base_dir, data_dir, output_dir)
        dataset.dump(output_dir)

        log.info('-' * 50)
        log.info('TEST with {:s} dataset'.format(data_dir))

        l_n_tests = 0
        l_n_success_k = 0
        l_n_success_1 = 0
        for fid, fname in enumerate(sorted(dataset.fnames)):

            img = imread(fname, as_gray=True)
            img = img[:-110, :]
            #img /= 255.
            #img = (img - img.min()) / img.ptp()
            #print('original image: ', img.shape)
            log.info('-'*50)
            log.info('TEST with {:s}'.format(fname))

            for q_idx in sorted(dataset.fileMap[fname]):
                cx, cy, side, down_factor = dataset.info[q_idx]
                x0 = cx - side//2
                y0 = cy - side//2

                # ----------------------------------------------------------------
                # extract query image
                q_im = img[y0:y0+side, x0:x0+side]
                q_bbox = [x0, y0, x0+side, y0+side]

                # ----------------------------------------------------------------
                # sample db image coordinate
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    sImg = rescale(img, 1/down_factor)
                    sImg = (sImg - sImg.min()) / sImg.ptp()

                s_h, s_w = sImg.shape
                s_side = side / down_factor
                s_side_int = int(np.round(s_side))

                sx0 = np.arange(0, s_w - s_side_int, s_side_int//4)
                sy0 = np.arange(0, s_h - s_side_int, s_side_int//4)

                sx0, sy0 = np.meshgrid(sx0, sy0)
                sx0 = sx0.flatten()
                sy0 = sy0.flatten()

                bboxes = np.zeros((len(sx0), 4), dtype=np.int32)
                bboxes[..., 0] = sx0
                bboxes[..., 1] = sy0
                bboxes[..., 2] = sx0 + s_side_int
                bboxes[..., 3] = sy0 + s_side_int

                # ----------------------------------------------------------------
                # compute iou to find the expected top-5
                q_bbox = np.array([q_bbox], dtype=np.float32) / down_factor
                q_iou = calc_iou(q_bbox, bboxes)
                q_iou_ind = np.argsort(q_iou)[::-1]
                q_iou_ind = q_iou_ind[:top_K]

                # ----------------------------------------------------------------
                # convert to q_img to feature vector
                q_im = np.expand_dims(np.expand_dims(q_im, -1), 0)
                q_feat = net.get_feature(q_im)
                q_im = net.get_input(q_im, batch_size)

                # ----------------------------------------------------------------
                # compute feature vectors for DB
                db_im = []
                db_dist = []
                for bbox_it, (_x0, _y0, _x1, _y1) in enumerate(bboxes):
                    db_im.append(np.expand_dims(sImg[_y0:_y1, _x0:_x1], -1))
                    if (bbox_it+1)%batch_size == 0 or (len(db_im) and bbox_it+1 == len(bboxes)):
                        db_feat = net.get_feature(np.asarray(db_im, dtype=np.float32), batch_size)
                        db_im = []
                        dist = np.sqrt(np.sum((db_feat - q_feat) ** 2, axis=1))
                        db_dist.append(dist)
                db_dist = np.concatenate(db_dist, axis=0)
                db_dist_ind = np.argsort(db_dist)
                db_dist_ind = db_dist_ind[:top_K]

                db_im = []
                for (_x0, _y0, _x1, _y1) in bboxes[db_dist_ind]:
                    db_im.append(np.expand_dims(sImg[_y0:_y1, _x0:_x1], -1))
                db_im = np.asarray(db_im, dtype=np.float32)
                db_im = net.get_input(db_im, batch_size)

                # ----------------------------------------------------------------
                # Collect output and visualize
                retrieve_test = np.zeros(top_K, dtype=np.int32)
                for iii, retrieved_idx in enumerate(db_dist_ind):
                    if retrieved_idx in q_iou_ind:
                        retrieve_test[iii] = 1

                is_success = retrieve_test.sum() > 0
                l_n_tests += 1
                l_n_success_k += int(is_success)
                l_n_success_1 += retrieve_test[0]

                log.info("data set         : %s" % data_dir)
                log.info("fname            : %s" % fname)
                log.info("ID               : %s" % q_idx)
                log.info("is success       : {:s}".format('YES' if is_success else 'NO'))
                log.info("down factor      : %s", down_factor)
                log.info("expected indices : {}".format(q_iou_ind))
                log.info("expected  iou    : {}".format(q_iou[q_iou_ind]))
                log.info("retrieved indices: {}".format(db_dist_ind))
                log.info("retrieved iou    : {}".format(q_iou[db_dist_ind]))
                log.info("retrieved dist   : {}".format(db_dist[db_dist_ind]))
                log.info("retrieve test    : {}".format(retrieve_test))
                log.info("\n")

                img = (img - img.min()) / img.ptp()
                h1.set_data(img)
                for bbox in q_bbox:
                    bbox *= down_factor
                    q_rect.set_xy((bbox[0], bbox[1]))
                    q_rect.set_width(bbox[2] - bbox[0])
                    q_rect.set_height(bbox[3] - bbox[1])

                for it, bbox in enumerate(bboxes[db_dist_ind]):
                    bbox *= down_factor
                    k_rect[it].set_xy((bbox[0], bbox[1]))
                    k_rect[it].set_width(bbox[2] - bbox[0])
                    k_rect[it].set_height(bbox[3] - bbox[1])

                patches = [
                    (_im - _im.min()) / _im.ptp()
                    for _im in np.squeeze(np.vstack([q_im,db_im]))
                ]
                patches = np.hstack(patches)
                h2.set_data(patches)
                idx = int(np.argmax(q_iou[db_dist_ind])) + 1
                topk_1_rect.set_xy((idx*PATCH_SIZE, 0))

                plt.pause(0.0001)
                plt.draw()
                fig_output_dir = fig_output_dir_s if is_success else fig_output_dir_f
                bbox_fig.savefig(os.path.join(fig_output_dir, 'bbox_{:03d}_{:03d}_{:d}.png'.format(
                    fid, q_idx, int(is_success)
                )))
                topk_fig.savefig(os.path.join(fig_output_dir, 'topk_{:03d}_{:03d}_{:d}.png'.format(
                    fid, q_idx, int(is_success)
                )))

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



