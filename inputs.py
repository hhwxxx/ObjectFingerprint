from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
import functools
import os

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('tfrecord_folder', './tfrecords', 'Folder containing tfrecords.')

IMAGE_SHAPE = (512, 512, 3)

NUMBER_TRAIN_PAIRS = 11424
NUMBER_TRAIN_MATCHED_PAIRS = 2856
NUMBER_TRAIN_UNMATCHED_PAIRS = 8568
NUMBER_VAL_PAIRS = 1776
NUMBER_VAL_MATCHED_PAIRS = 444
NUMBER_VAL_UNMATCHED_PARIS = 1332


def decode(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_a_data': tf.FixedLenFeature([], tf.string, default_value=''),
            'image_a_height': tf.FixedLenFeature([], tf.int64, default_value=0),
            'image_a_width': tf.FixedLenFeature([], tf.int64, default_value=0),
            'image_a_channel': tf.FixedLenFeature([], tf.int64, default_value=3),
            'image_a_index': tf.FixedLenFeature([], tf.int64, default_value=-1),
            'image_a_name': tf.FixedLenFeature([], tf.string, default_value=''),
            'image_a_format': tf.FixedLenFeature([], tf.string, default_value=''),
            'image_b_data': tf.FixedLenFeature([], tf.string, default_value=''),
            'image_b_height': tf.FixedLenFeature([], tf.int64, default_value=0),
            'image_b_width': tf.FixedLenFeature([], tf.int64, default_value=0),
            'image_b_channel': tf.FixedLenFeature([], tf.int64, default_value=3),
            'image_b_index': tf.FixedLenFeature([], tf.int64, default_value=-1),
            'image_b_name': tf.FixedLenFeature([], tf.string, default_value=''),
            'image_b_format': tf.FixedLenFeature([], tf.string, default_value=''),
            'label': tf.FixedLenFeature([], tf.int64, default_value=-1),
        }
    )

    image_a = tf.image.decode_jpeg(features['image_a_data'], channels=3)
    image_b = tf.image.decode_jpeg(features['image_b_data'], channels=3)
    image_ida = features['image_a_index']
    image_idb = features['image_b_index']
    label = tf.cast(features['label'], tf.int32)

    return image_a, image_b, label, image_ida, image_idb


def shift_image(image_a, image_b, width_shift_range, height_shift_range):
    """This fn will perform the horizontal or vertical shift"""
    if width_shift_range or height_shift_range:
        if width_shift_range:
            width_shift_range = tf.random_uniform([], 
                -width_shift_range * IMAGE_SHAPE[1], 
                width_shift_range * IMAGE_SHAPE[1])
        if height_shift_range:
            height_shift_range = tf.random_uniform([], 
                -height_shift_range * IMAGE_SHAPE[0], 
                height_shift_range * IMAGE_SHAPE[0])
        # Translate both 
        image_a = tf.contrib.image.translate(
            image_a, [width_shift_range, height_shift_range])
        image_b = tf.contrib.image.translate(
            image_b, [width_shift_range, height_shift_range])

    return image_a, image_b


def flip_image(horizontal_flip, image_a, image_b):
    if horizontal_flip:
        flip_prob = tf.random_uniform([], 0.0, 1.0)
        image_a, image_b = tf.cond( tf.less(flip_prob, 0.5), 
            lambda: (tf.image.flip_left_right(image_a), 
                     tf.image.flip_left_right(image_b)), 
            lambda: (image_a, image_b))

    return image_a, image_b


def normalize(image_a, image_b):
    # normalize image to [-1, 1]
    image_a = (2.0 / 255.0) * tf.to_float(image_a) - 1.0
    image_b = (2.0 / 255.0) * tf.to_float(image_b) - 1.0

    return image_a, image_b


def augment(image_a,
            image_b,
            label,
            image_ida,
            image_idb,
            resize=None,  # Resize the image to some size e.g. [512, 512]
            hue_delta=0,  # Adjust the hue of an RGB image by random factor
            horizontal_flip=False,  # Random left right flip,
            width_shift_range=0,  # Randomly translate the image horizontally
            height_shift_range=0):  # Randomly translate the image vertically 
    if resize is not None:
        # Resize both images
        image_a = tf.image.resize_images(
            image_a, resize, align_corners=True, 
            method=tf.image.ResizeMethod.BILINEAR)
        image_b = tf.image.resize_images(
            image_b, resize, align_corners=True, 
            method=tf.image.ResizeMethod.BILINEAR)
    
    if hue_delta:
        image_a = tf.image.random_hue(image_a, hue_delta)
        image_b = tf.image.random_hue(image_b, hue_delta)
    
    image_a, image_b = flip_image(horizontal_flip, image_a, image_b)
    image_a, image_b = shift_image(
        image_a, image_b, width_shift_range, height_shift_range)
    image_a, image_b = normalize(image_a, image_b)
    
    return image_a, image_b, label, image_ida, image_idb


train_config = {
    'resize': [IMAGE_SHAPE[0], IMAGE_SHAPE[1]],
    'hue_delta': 0.1,
    'horizontal_flip': True,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1
}
train_preprocessing_fn = functools.partial(augment, **train_config)

val_config = {
    'resize': [IMAGE_SHAPE[0], IMAGE_SHAPE[1]],
}
val_preprocessing_fn = functools.partial(augment, **val_config)


def inputs(dataset_split, is_training, batch_size, num_epochs=None):
    filename = os.path.join(FLAGS.tfrecord_folder, dataset_split + '.tfrecord')

    with tf.name_scope('input'):
        dataset = tf.data.TFRecordDataset(filename)

        dataset = dataset.map(decode)

        if is_training:
            dataset = dataset.map(train_preprocessing_fn)
        else:
            dataset = dataset.map(val_preprocessing_fn)

        min_queue_examples = int(NUMBER_VAL_PAIRS * 0.1)
        if is_training:
            min_queue_examples = int(NUMBER_TRAIN_PAIRS * 0.1)
            dataset = dataset.shuffle(buffer_size=min_queue_examples + 3 * batch_size)
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=min_queue_examples)
    
    iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()