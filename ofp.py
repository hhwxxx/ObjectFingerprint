from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import ofp_model


def inference(image_a, image_b, is_training=False):
    logits = ofp_model.model(image_a, image_b, is_training=is_training)

    return logits


def loss(logits, labels):
    logits = tf.reshape(logits, shape=[-1])
    cross_entropy_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels,
                                           logits=logits, scope='loss')
    tf.summary.scalar('corss_entropy_loss', cross_entropy_loss)
    tf.losses.add_loss(cross_entropy_loss)
    
    total_loss = tf.losses.get_total_loss()
    
    return total_loss