import tensorflow as tf
import numpy as np

def calc_iou(boxes1, boxes2, scope='iou'):
    with tf.variable_scope(scope):
        # transform (x_center, y_center, w, h) to (x1, y1, x2, y2)
        boxes1_t = tf.stack(
            [
                boxes1[..., 0] - boxes1[..., 2] / 2.,
                boxes1[..., 1] - boxes1[..., 3] / 2.,
                boxes1[..., 0] + boxes1[..., 2] / 2.,
                boxes1[..., 1] + boxes1[..., 3] / 2.,
            ],
            axis=-1
        )
        boxes2_t = tf.stack(
            [
                boxes2[..., 0] - boxes2[..., 2] / 2.,
                boxes2[..., 1] - boxes2[..., 3] / 2.,
                boxes2[..., 0] + boxes2[..., 2] / 2.,
                boxes2[..., 1] + boxes2[..., 3] / 2.,
            ],
            axis=-1
        )

        # calculate the left-upper & right-bottom corners
        lu = tf.maximum(boxes1_t[..., :2], boxes2_t[..., :2])
        rb = tf.minimum(boxes1_t[..., 2:], boxes2_t[..., 2:])

        # intersection
        inter = tf.maximum(0., rb - lu)
        inter_area = inter[..., 0] * inter[..., 1]

        # calculate the areas
        area1 = boxes1[..., 2] * boxes1[..., 3]
        area2 = boxes2[..., 2] * boxes2[..., 3]

        # calculate iou
        union_area = tf.maximum(area1 + area2 - inter_area, 1e-10)
        iou = tf.clip_by_value(inter_area / union_area, 0., 1.)

    return iou


def loss(logits, labels, n_bbox_estimators, n_parameters=5, csy=16, csx=16):
    """
    Compute loss for bbox prediction (from YOLO)

    Args:
        logits: (4-D tensor), [batch_size, csy, csx, N] where
            csy: cell size y
            csx: cell size x
            N: n_bbox_estimators * n_parameters (per bbox)

        labels: (4-D tensor), [batch_size, csy, csx, n_parameters]
            normalized one-shot bbox labels

        n_bbox_estimators: (int), the number of bbox estimators

            predicted bboxes
            0: x       - [0, 1] w.r.t a cell
            1: y       - [0, 1] w.r.t a cell
            2: sqrt(w) - [0, 1] w.r.t an image width
            3: sqrt(h) - [0, 1] w.r.t an image height
            todo: 4: angle   - [0, 1] w.r.t the center of the bbox


        n_parameters: (int), the number of parameters subject to be predicted by the network
            5: confidence, x_center, y_center, sqrt(w), sqrt(h)
            todo: 6: confidence, x_center, y_center, sqrt(w), sqrt(h), rotation angle

    Returns:

    """
    assert n_parameters==5, 'Currently n_parameters must be 5: {}'.format(n_parameters)
    logits_shape = tf.shape(logits)
    batch_size = logits_shape[0]
    #csy = logits_shape[1]
    #csx = logits_shape[2]

    # for training, csy = csx = 16
    __c_offset = np.transpose(np.reshape(
        [np.array(np.arange(csy))] * csx * n_bbox_estimators,
        [n_bbox_estimators, csy, csx]),
        [1, 2, 0]
    )

    with tf.variable_scope('ftfy-loss'):
        # parsing logits
        pred = tf.reshape(logits, [-1, csy, csx, n_bbox_estimators, n_parameters])
        pred_confidence = tf.reshape(pred[..., 0], [-1, csy, csx, n_bbox_estimators])
        pred_bboxes = tf.reshape(pred[..., 1:], [-1, csy, csx, n_bbox_estimators, n_parameters-1])

        # parsing labels
        response = tf.reshape(labels[..., 0], [-1, csy, csx, 1])
        bboxes = tf.tile(
            tf.reshape(labels[..., 1:], [-1, csy, csx, 1, n_parameters-1]),
            [1, 1, 1, n_bbox_estimators, 1]) # w.r.t an image size

        # transform predicted bbox to match with the labels
        offset_x = tf.tile(
            tf.reshape(tf.constant(__c_offset, dtype=tf.float32), [1, 16, 16, n_bbox_estimators]),
            [batch_size, 1, 1, 1]
        )
        offset_y = tf.transpose(offset_x, (0, 2, 1, 3))
        pred_bboxes_t = tf.stack([
            (pred_bboxes[..., 0] + offset_x) / tf.cast(csx, tf.float32),
            (pred_bboxes[..., 1] + offset_y) / tf.cast(csy, tf.float32),
            tf.square(pred_bboxes[..., 2]),
            tf.square(pred_bboxes[..., 3])
        ], axis=-1)
        bboxes_t = tf.stack([
            bboxes[..., 0] * tf.cast(csx, tf.float32) - offset_x,
            bboxes[..., 1] * tf.cast(csy, tf.float32) - offset_y,
            tf.sqrt(bboxes[..., 2]),
            tf.sqrt(bboxes[..., 3])
        ], axis=-1)

        # calculate iou
        pred_iou = calc_iou(pred_bboxes_t, bboxes)

        # create mask
        obj_mask = tf.cast(pred_iou >= tf.reduce_max(pred_iou, axis=-1, keepdims=True), tf.float32)
        obj_mask = obj_mask * response
        noobj_mask = tf.ones_like(obj_mask, dtype=tf.float32) - obj_mask
        coord_mask = tf.expand_dims(obj_mask, 4)

        # calculate object loss
        obj_delta = obj_mask * (pred_confidence - pred_iou)
        obj_loss = tf.reduce_mean(tf.reduce_sum(tf.square(obj_delta), axis=[1, 2, 3]))

        # calculate no-object loss
        noobj_delta = noobj_mask * pred_confidence
        noobj_loss = tf.reduce_mean(tf.reduce_sum(tf.square(noobj_delta), axis=[1, 2, 3]))

        # calculate coordinate loss
        coord_delta = coord_mask * (pred_bboxes - bboxes_t)
        coord_loss = tf.reduce_mean(tf.reduce_sum(tf.square(coord_delta), axis=[1, 2, 3, 4]))

    return obj_loss, noobj_loss, coord_loss

def inference(logits, n_bbox_estimators, n_parameters=5, csy=16, csx=16):
    assert n_parameters==5, 'Currently n_parameters must be 5: {}'.format(n_parameters)

    logits_shape = tf.shape(logits)
    batch_size = logits_shape[0]
    #csy = logits_shape[1]
    #csx = logits_shape[2]

    __c_offset = np.transpose(np.reshape(
        [np.array(np.arange(csy))] * csx * n_bbox_estimators,
        [n_bbox_estimators, csy, csx]
    ), [1, 2, 0])

    with tf.variable_scope('inference'):
        pred = tf.reshape(logits, [-1, csy, csx, n_bbox_estimators, n_parameters])
        pred_confidence = tf.reshape(pred[..., 0], [-1, csy, csx, n_bbox_estimators])
        pred_bboxes = tf.reshape(pred[..., 1:], [-1, csy, csx, n_bbox_estimators, n_parameters-1])

        offset_x = tf.tile(
            tf.reshape(tf.constant(__c_offset, dtype=tf.float32), [1, csy, csx, n_bbox_estimators]),
            [batch_size, 1, 1, 1]
        )
        offset_y = tf.transpose(offset_x, (0, 2, 1, 3))

        pred_bboxes_t = tf.stack([
            (pred_bboxes[..., 0] + offset_x) / tf.cast(csx, tf.float32),
            (pred_bboxes[..., 1] + offset_y) / tf.cast(csy, tf.float32),
            tf.square(pred_bboxes[..., 2]),
            tf.square(pred_bboxes[..., 3])
        ], axis=-1)

    pred_confidence = tf.reshape(pred_confidence, [-1, csy*csx*n_bbox_estimators])
    pred_bboxes_t = tf.reshape(pred_bboxes_t, [-1, csy*csx*n_bbox_estimators, n_parameters-1])

    return pred_confidence, pred_bboxes_t