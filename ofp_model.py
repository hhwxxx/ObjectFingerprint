from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 

slim = tf.contrib.slim

WEIGHT_DECAY = 0.0005
DROPOUT_KEEP_PROB = 0.5

def vgg_conv(images, is_training=False, reuse=False):
    with tf.variable_scope('vgg_16'):
        with slim.arg_scope([slim.conv2d], 
                            padding='SAME',
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(WEIGHT_DECAY),
                            reuse=reuse) as scope:
            # block 1
            net = slim.repeat(images, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], stride=2, padding='VALID', scope='pool1')
            # block 2
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], stride=2, padding='VALID', scope='pool2')
            # block 3
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], stride=2, padding='VALID', scope='pool3')
            # block 4
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], stride=2, padding='VALID', scope='pool4')
            # block 5
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], stride=2, padding='VALID', scope='pool5')

            net = tf.reduce_mean(net, axis=[1, 2], keepdims=False)

    return net


def ofp_id(feature, is_training=False, reuse=False):
    with tf.variable_scope('ofp_id'):
        with slim.arg_scope([slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(WEIGHT_DECAY),
                            reuse=reuse):
            feature = slim.fully_connected(feature, 256, scope='fc1')
            feature = slim.dropout(feature, DROPOUT_KEEP_PROB,
                                   is_training=is_training, scope='dropout1')
            logits = slim.fully_connected(feature, 110, scope='fc2', activation_fn=None)
    
    return logits


def ofp_top(feature, is_training=False):
    with tf.variable_scope('ofp_top'):
        with slim.arg_scope([slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(WEIGHT_DECAY)):
            feature = slim.fully_connected(feature, 512, scope='fc1')
            feature = slim.dropout(feature, DROPOUT_KEEP_PROB,
                                        is_training=is_training, scope='dropout1')
            feature = slim.fully_connected(feature, 256, scope='fc2')
            feature = slim.dropout(feature, DROPOUT_KEEP_PROB,
                                        is_training=is_training, scope='dropout2')
            logits = slim.fully_connected(feature, 2, activation_fn=None, scope='fc3')
    
    return logits
