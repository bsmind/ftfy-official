import operator
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

def fpr(labels, scores, recall_rate = 0.95):
    """Error rate at 95% recall"""
    # sort label-score tuples by the score in descending order
    #indices = np.argsort(scores)[::-1]
    #sorted_scores = scores[ind]
    #sorted_labels = labels[ind]
    sorted_scores = sorted(zip(labels, scores), key=operator.itemgetter(1), reverse=False)
    # compute error rate
    n_match = sum(1 for x in sorted_scores if x[0] == 1)
    n_thresh = recall_rate * n_match

    tp = 0
    count = 0
    for label, score in sorted_scores:
        #print(score)
        count += 1
        if label == 1:
            tp += 1
        if tp >= n_thresh:
            break

    return float(count - tp) / count

def retrieval_recall_K(features, labels, is_query, K: list, collect_top_5=False):
    recall_rate = np.zeros(len(K), dtype=np.float32)
    query_ind, = np.nonzero(is_query)
    n_queries = len(query_ind)
    max_k = max(K)

    top_5_collection = []

    for q_idx in query_ind:
        q_feat = features[q_idx]
        q_label = labels[q_idx]

        dist = np.sqrt(np.sum((features - q_feat) ** 2, axis=1))
        sorted_ind = np.argsort(dist)
        # exclude queried features
        sorted_ind = sorted_ind[sorted_ind != q_idx]
        sorted_ind = sorted_ind[:max_k]

        r_labels = labels[sorted_ind]
        first = np.where(r_labels == q_label)[0]

        if collect_top_5:
            success = 0 if len(first) == 0 else int(first[0] < 5)
            top_5_collection.append(
                [q_idx] + sorted_ind[:5].tolist() + [success]
            )

        if len(first) == 0:
            continue

        first = first[0]
        for idx, k in enumerate(K):
            recall_rate[idx] += int(first < k)

    recall_rate /= n_queries
    top_5_collection = np.asarray(top_5_collection, dtype=np.int32)
    return recall_rate, top_5_collection

def calc_iou(bbox1, bbox2):
    bbox1_t = np.stack(
        [
            bbox1[..., 0] - bbox1[..., 2] / 2.,
            bbox1[..., 1] - bbox1[..., 3] / 2.,
            bbox1[..., 0] + bbox1[..., 2] / 2.,
            bbox1[..., 1] + bbox1[..., 3] / 2.,
        ],
        axis=-1
    )
    bbox2_t = np.stack(
        [
            bbox2[..., 0] - bbox2[..., 2] / 2.,
            bbox2[..., 1] - bbox2[..., 3] / 2.,
            bbox2[..., 0] + bbox2[..., 2] / 2.,
            bbox2[..., 1] + bbox2[..., 3] / 2.,
        ],
        axis=-1
    )

    lu = np.maximum(bbox1_t[..., :2], bbox2_t[..., :2])
    rb = np.minimum(bbox1_t[..., 2:], bbox2_t[..., 2:])

    inter = np.maximum(0, rb - lu)
    inter_area = inter[..., 0] * inter[..., 1]

    area1 = bbox1[..., 2] * bbox1[..., 3]
    area2 = bbox2[..., 2] * bbox2[..., 3]

    union_area = np.maximum(area1 + area2 - inter_area, 1e-10)
    iou = np.clip(inter_area / union_area, 0., 1.)

    return iou

def calc_iou_k(pred_bboxes_k, bboxes):
    all_iou_k = []
    for bboxes_k, bbox in zip(pred_bboxes_k, bboxes):
        iou_k = calc_iou(bboxes_k, bbox)
        all_iou_k.append(iou_k)

    return np.asarray(all_iou_k)

def ftfy_retrieval_accuracy(iou_k, top_k:list, iou_thrs:list):
    n_samples = iou_k.shape[0]
    accuracy = np.zeros((len(iou_thrs), len(top_k)), dtype=np.float32)

    for i, thrs in enumerate(iou_thrs):
        for j, k in enumerate(top_k):
            acc = np.sum(np.max(iou_k[:, :k], axis=-1) >= thrs)
            accuracy[i, j] = acc

    accuracy /= n_samples
    return accuracy