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

    d_pos = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    d_neg = tf.reduce_sum(tf.square(anchor - negative), axis=1)

    if margin is not None:
        loss = tf.maximum(0., margin + d_pos - d_neg)
    else:
        loss = tf.log1p(tf.exp(d_pos - d_neg))

    loss = tf.reduce_mean(loss)
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










