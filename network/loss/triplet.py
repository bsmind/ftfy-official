"""Define functions to create the triplet loss with offline and online triplet mining."""

import tensorflow as tf

def batch_offline_triplet_loss(anchor, positive, negative, margin=None):
    """ Build offline triplet loss over a batch of embeddings.

    triplet_loss = max(margin + d_pos - d_neg, 0) where
        d_pos = || anchor - positive ||_2
        d_neg = || anchor - negative ||_2

    Args:
        anchor: tensor of anchor embeddings (batch_size, embed_dim)
        positive: tensor of positive embeddings (batch_size, embed_dim)
        negative: tensor of negative embeddings (batch_size, embed_dim)
        margin: margin for triplet loss, if None, use soft-margin triplet loss

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # d_pos = tf.sqrt(tf.reduce_sum(tf.square(anchor - positive), axis=1))
    # d_neg = tf.sqrt(tf.reduce_sum(tf.square(anchor - negative), axis=1))
    d_pos = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    d_neg = tf.reduce_sum(tf.square(anchor - negative), axis=1)

    if margin is not None:
        loss = tf.maximum(0., margin + d_pos - d_neg)
    else:
        loss = tf.log1p(tf.exp(d_pos - d_neg))

    loss = tf.reduce_mean(loss)
    return loss

def spreadout_triplet_loss(anchor, positive, negative):
    margin_0 = 1.41
    margin_1 = 0.5
    margin_2 = 0.7
    alpha = 1.0
    beta = 1.0
    descriptor_dim = 128

    d_over_1 = tf.constant(beta/descriptor_dim)
    alpha_tf = tf.constant(alpha)
    positive_margin = tf.constant(margin_1)

    eucd_p = tf.pow(tf.subtract(anchor, positive), 2)
    eucd_p = tf.reduce_sum(eucd_p, 1)
    eucd_p = tf.sqrt(eucd_p + 1e-6)

    eucd_n1 = tf.pow(tf.subtract(anchor, negative), 2)
    eucd_n1 = tf.reduce_sum(eucd_n1, 1)
    eucd_n1 = tf.sqrt(eucd_n1 + 1e-6)

    eucd_n2 = tf.pow(tf.subtract(positive, negative), 2)
    eucd_n2 = tf.reduce_sum(eucd_n2, 1)
    eucd_n2 = tf.sqrt(eucd_n2 + 1e-6)

    secMoment_n1 = tf.pow(tf.reduce_sum(tf.multiply(anchor, negative), 1), 2)

    mean = tf.pow(tf.reduce_mean(tf.reduce_sum(tf.multiply(anchor, negative), 1)), 2)

    # invertable loss for standard patches
    rand_neg = tf.reduce_mean(secMoment_n1)
    # covariance loss for transformed patches
    pos = tf.maximum(tf.subtract(positive_margin, tf.subtract(eucd_n1, eucd_p)), 0)
    # total loss
    loss = tf.reduce_mean(pos) + \
        tf.multiply(alpha_tf, mean + tf.maximum(tf.subtract(rand_neg, d_over_1), 0))

    return loss

def batch_offline_lossless_triplet_loss(anchor, positive, negative, N, beta, epsilon=1e-8):
    d_pos = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    d_neg = tf.reduce_sum(tf.square(anchor - negative), axis=1)

    # -ln(-x/N + 1)
    d_pos = -tf.log(
        tf.maximum(-tf.divide(d_pos, beta) + 1, epsilon)
    )
    d_neg = -tf.log(
        tf.maximum(-tf.divide((N - d_neg), beta) + 1, epsilon)
    )

    loss = d_neg + d_pos
    loss = tf.reduce_mean(loss)
    return loss










