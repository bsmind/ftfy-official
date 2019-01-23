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


class Evaluator_old(object):
    def __init__(self, data, patch_size):
        self.data = data
        self.n_data = len(data)
        self.patch_size = patch_size

        self.used_ind = []

        self.db_features = []
        self.db_rows = []
        self.db_cols = []

        self.q_img = []
        self.q_features = []
        self.q_rows = []
        self.q_cols = []

        self.distmap = None

    def reset(self):
        '''Reset all'''
        self.used_ind = []
        self.reset_db()
        self.reset_q()

    def reset_db(self):
        '''Reset DB and queries'''
        self.db_features = []
        self.db_rows = []
        self.db_cols = []
        self.distmap = None

    def reset_q(self):
        '''Reset queries'''
        self.q_img = []
        self.q_features = []
        self.q_rows = []
        self.q_cols = []
        self.distmap = None

    def get_image(self, idx=None):
        '''Get an image'''
        if idx is None:
            p = np.array([1.0] * self.n_data)
            p[self.used_ind] = 0.
            idx = np.random.choice(
                self.n_data,
                replace=False,
                p=p/p.sum()
            )
            self.used_ind.append(idx)

        idx = max(0, min(idx, self.n_data-1))
        return self.data[idx]

    def add_item(self, features, rows, cols, is_db=False, q_img=None):
        '''Add item'''
        if is_db:
            self.db_features.append(features)
            self.db_rows.append(rows)
            self.db_cols.append(cols)
        else:
            self.q_features.append(features)
            self.q_rows.append(rows)
            self.q_cols.append(cols)
            if q_img is not None:
                self.q_img.append(q_img)

    def fit(self, q_features=None):
        '''Compute distance map size of (# queries x # db items)'''
        if isinstance(self.db_features, list):
            self.db_features = np.concatenate(self.db_features, axis=0)
            self.db_rows = np.concatenate(self.db_rows, axis=0)
            self.db_cols = np.concatenate(self.db_cols, axis=0)

        if q_features is None:
            if isinstance(self.q_features, list):
                self.q_features = np.concatenate(self.q_features, axis=0)
                self.q_rows = np.concatenate(self.q_rows, axis=0)
                self.q_cols = np.concatenate(self.q_cols, axis=0)
                self.q_img = np.concatenate(self.q_img, axis=0)
            self.distmap = euclidean_distances(self.q_features, self.db_features)
            return self.distmap
        else:
            return euclidean_distances(q_features, self.db_features)

    def get_top_k(self, top_k, distmap=None):
        '''Return top K indices (# queries x top K)'''
        if distmap is None:
            distmap = self.distmap

        if distmap is None:
            raise ValueError("Invalid distance map (None)!")

        top_k_ind = np.argsort(distmap, axis=1)
        top_k_ind = top_k_ind[:, :top_k]
        return top_k_ind

    def get_top_k_pos(self, top_k):
        top_k_ind = self.get_top_k(top_k)
        N = top_k_ind.shape[0]
        all_pos = []
        for i in range(N):
            ind = top_k_ind[i]
            db_rows = self.db_rows[ind] # vector
            db_cols = self.db_cols[ind] # vector
            q_rows = self.q_rows[i] # scalar
            q_cols = self.q_cols[i] # scalar


            pos = list()
            pos.append((q_cols, q_rows))
            for x, y in zip(db_cols, db_rows):
                pos.append((x, y))
            all_pos.append(pos)

        return all_pos, self.q_img

    def get_accuracy(self, top_k=5, iou_threshold=0.7, distmap=None):
        '''Return averaged retrieval accuracy'''
        top_k_ind = self.get_top_k(top_k, distmap)
        N = top_k_ind.shape[0]
        accuracy = 0
        for i in range(N):
            ind = top_k_ind[i]
            db_rows = self.db_rows[ind] # vector
            db_cols = self.db_cols[ind] # vector
            q_rows = self.q_rows[i] # scalar
            q_cols = self.q_cols[i] # scalar
            ious = calc_iou(q_rows, q_cols, db_rows, db_cols, self.patch_size)
            accuracy += np.sum(ious > iou_threshold) > 0
        return accuracy / N

class Evaluator(object):
    def __init__(self, iou_threshold, top_k=None):
        self.top_k = top_k
        self.iou_threshold = iou_threshold

    def __call__(self, db, queries, rect_sz=(204,204)):
        '''distmap
        - # queries x # db items
        '''
        distmap = euclidean_distances(queries.features, db.features)
        n_queries = len(distmap)

        # compute top k (# queries x top_k)
        top_k_ind = np.argsort(distmap, axis=1)
        if self.top_k is not None:
            top_k_ind = top_k_ind[:, :self.top_k]

        # compute accuracy
        top_k_ious = []
        accuracy = 0
        for i in range(n_queries):
            ind = top_k_ind[i]
            db_x0 = db.x0[ind]
            db_y0 = db.y0[ind]
            q_x0 = queries.x0[i]
            q_y0 = queries.y0[i]
            ious = calc_iou(q_y0, q_x0, db_y0, db_x0, rect_sz=rect_sz)
            accuracy += np.sum(ious > self.iou_threshold) > 0
            top_k_ious.append(ious)
        accuracy /= n_queries
        top_k_ious = np.vstack(top_k_ious)

        return accuracy, top_k_ind, top_k_ious

