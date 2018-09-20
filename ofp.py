from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import ofp_model


def inference(image_a, image_b, is_training=False):
    feature_a = ofp_model.vgg_conv(image_a, is_training=is_training, reuse=False)
    feature_b = ofp_model.vgg_conv(image_b, is_training=is_training, reuse=True)

    logits_ida = ofp_model.ofp_id(feature_a, is_training=is_training, reuse=False)
    logits_idb = ofp_model.ofp_id(feature_b, is_training=is_training, reuse=True)

    feature_fuse = tf.concat([feature_a, feature_b], axis=-1)
    logits = ofp_model.ofp_top(feature_fuse, is_training=is_training)

    return logits, logits_ida, logits_idb


def loss(logits, logits_ida, logits_idb, labels, image_ida, image_idb):
    logits = tf.reshape(logits, shape=[-1, 2])
    onehot_labels = tf.one_hot(indices=labels, depth=2, on_value=1, off_value=0)
    weights = tf.where(tf.equal(labels, 0), tf.to_float(tf.equal(labels, 0)), 
                       tf.to_float(tf.equal(labels, 1))*3)
    cross_entropy_loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels,
                                                         logits=logits, weights=weights, 
                                                         scope='cls_loss')
    
    logits_ida = tf.reshape(logits_ida, shape=[-1, 110])
    logits_idb = tf.reshape(logits_idb, shape=[-1, 110])
    onehot_ida = tf.one_hot(indices=image_ida, depth=110, on_value=1, off_value=0)
    onehot_idb = tf.one_hot(indices=image_idb, depth=110, on_value=1, off_value=0)

    cross_entropy_loss_ida = tf.losses.softmax_cross_entropy(onehot_labels=onehot_ida,
                                                             logits=logits_ida, scope='ida_loss')
    cross_entropy_loss_idb = tf.losses.softmax_cross_entropy(onehot_labels=onehot_idb,
                                                             logits=logits_idb, scope='idb_loss')

    # cross_entropy_loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels,
    #                                                      logits=logits, scope='loss')
    tf.summary.scalar('cls_loss', cross_entropy_loss)
    tf.summary.scalar('ida_loss', cross_entropy_loss_ida)
    tf.summary.scalar('idb_loss', cross_entropy_loss_idb)

    total_loss = 0.6 * cross_entropy_loss + 0.2 * cross_entropy_loss_ida + 0.2 * cross_entropy_loss_idb

    tf.losses.add_loss(total_loss)
    
    total_loss = tf.losses.get_total_loss()
    
    return total_loss
