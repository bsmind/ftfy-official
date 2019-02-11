import operator
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

def calc_iou(ref_y, ref_x, other_ys, other_xs, rect_sz):
    '''Calculate iou'''
    ul_x = np.maximum(ref_x, other_xs)
    ul_y = np.maximum(ref_y, other_ys)

    br_x = np.minimum(ref_x + rect_sz[1], other_xs + rect_sz[1])
    br_y = np.minimum(ref_y + rect_sz[0], other_ys + rect_sz[0])

    #intersect_area = np.maximum(0., (br_x - ul_x) * (br_y - ul_y))
    intersect_area = np.maximum(0, br_x - ul_x) * np.maximum(0, br_y - ul_y)
    area = rect_sz[0] * rect_sz[1]
    return np.clip(intersect_area / (2*area - intersect_area), 0., 1.)

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



