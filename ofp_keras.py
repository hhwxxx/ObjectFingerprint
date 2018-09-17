from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf 
from tensorflow import keras 
import functools
import os

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('init_model', './init_models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                    'Initial model.')
flags.DEFINE_string('tfrecord_folder', '../tfrecords', 'Folder containing tfrecords.')
flags.DEFINE_string('train', 'train_mini', 'Train dataset.')
flags.DEFINE_string('val', 'val_mini', 'Val dataset.')
flags.DEFINE_integer('epochs', 100, 'Training epochs.')
flags.DEFINE_integer('train_batch_size', 4, 'Train batch size.')
flags.DEFINE_integer('val_batch_size', 2, 'Val batch size.')
flags.DEFINE_string('train_dir', './train', 'Train directory.')
flags.DEFINE_string('log_dir', './train/log', 'Log directory.')

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
    image_a_index = features['image_a_index']
    image_b_index = features['image_b_index']
    label = tf.cast(features['label'], tf.int32)

    return image_a, image_b, label


def shift_image(image, label, width_shift_range, height_shift_range):
    """This fn will perform the horizontal or vertical shift"""
    if width_shift_range or height_shift_range:
        if width_shift_range:
            width_shift_range = tf.random_uniform([], 
                                                  -width_shift_range * INPUT_SIZE[1],
                                                  width_shift_range * INPUT_SIZE[1])
        if height_shift_range:
            height_shift_range = tf.random_uniform([],
                                                   -height_shift_range * INPUT_SIZE[0],
                                                   height_shift_range * INPUT_SIZE[0])
        # Translate both 
        image = tf.contrib.image.translate(image, [width_shift_range, height_shift_range])
        label = tf.contrib.image.translate(label, [width_shift_range, height_shift_range])

    return image, label


def flip_image(horizontal_flip, image_a, image_b):
    if horizontal_flip:
        flip_prob = tf.random_uniform([], 0.0, 1.0)
        image_a, image_b = tf.cond(tf.less(flip_prob, 0.5), 
                                          lambda: (tf.image.flip_left_right(image_a), 
                                                   tf.image.flip_left_right(image_b)),
                                          lambda: (image_a, image_b))

    return image_a, image_b


def normalize(image_a, image_b, label):
    # normalize image to [-1, 1]
    image_a = (2.0 / 255.0) * tf.to_float(image_a) - 1.0
    image_b = (2.0 / 255.0) * tf.to_float(image_b) - 1.0

    return image_a, image_b, label


def augment(image_a,
            image_b,
            label,
            resize=None,  # Resize the image to some size e.g. [512, 512]
            hue_delta=0,  # Adjust the hue of an RGB image by random factor
            horizontal_flip=False,  # Random left right flip,
            width_shift_range=0,  # Randomly translate the image horizontally
            height_shift_range=0):  # Randomly translate the image vertically 
    if resize is not None:
        # Resize both images
        image_a = tf.image.resize_images(image_a, resize, align_corners=True, method=tf.image.ResizeMethod.BILINEAR)
        image_b = tf.image.resize_images(image_b, resize, align_corners=True, method=tf.image.ResizeMethod.BILINEAR)
    
    if hue_delta:
        image_a = tf.image.random_hue(image_a, hue_delta)
        image_b = tf.image.random_hue(image_b, hue_delta)
    
    image_a, image_b = flip_image(horizontal_flip, image_a, image_b)
    #image, label = shift_image(image, label, width_shift_range, height_shift_range)
    
    return image_a, image_b, label


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
        dataset = dataset.map(normalize)

        min_queue_examples = int(NUMBER_VAL_PAIRS * 0.4)
        if is_training:
            min_queue_examples = int(NUMBER_TRAIN_PAIRS * 0.4)
            dataset = dataset.shuffle(buffer_size=min_queue_examples + 3 * batch_size)
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=min_queue_examples)

    return dataset


def vgg_based():
    # Build VGG based model.
    img_input = keras.Input(shape=IMAGE_SHAPE)
    # Block 1
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    vgg_conv = keras.Model(inputs=img_input, outputs=x)
    vgg_conv.load_weights(filepath=FLAGS.init_model, by_name=False)

    x = keras.layers.Flatten(name='flatten')(x)
    x = keras.layers.Dense(1024, activation='relu', name='fc1')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(512, activation='relu', name='fc2')(x)
    x = keras.layers.Dropout(0.5)(x)

    model = keras.Model(inputs=img_input, outputs=x, name='vgg_based')

    return model


def siamese_model():
    image_a = keras.Input(shape=IMAGE_SHAPE)
    image_b = keras.Input(shape=IMAGE_SHAPE)

    vgg_based_model = vgg_based()

    feature_vector_a = vgg_based_model(image_a)
    feature_vector_b = vgg_based_model(image_b)
    feature_vector_concatenated = keras.layers.concatenate([feature_vector_a, feature_vector_b])
    feature_vector = keras.layers.Dense(512, activation='relu', name='fc3')(feature_vector_concatenated)
    feature_vector = keras.layers.Dropout(0.5)(feature_vector)

    output = keras.layers.Dense(1, activation='sigmoid')(feature_vector)

    vgg_siamese_model = keras.Model(inputs=[image_a, image_b], outputs=output)

    return vgg_siamese_model


def main(unused_argv):
    train_dataset = inputs(dataset_split=FLAGS.train, is_training=True, batch_size=FLAGS.train_batch_size, num_epochs=None)
    val_dataset = inputs(dataset_split=FLAGS.val, is_training=False, batch_size=FLAGS.val_batch_size, num_epochs=1)

    model = siamese_model()
    model.summary()

    model.compile(loss='binary_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    callbacks=[
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=FLAGS.train_dir, monitor='val_loss',
                                           verbose=1, save_best_only=True, save_weights_only=True,
                                           period=5),
        tf.keras.callbacks.TensorBoard(log_dir=FLAGS.log_dir, histogram_freq=10, write_graph=True, 
                                       write_grads=True, write_images=True, batch_size=FLAGS.train_batch_size),
        #tf.keras.callbacks.LearningRateScheduler(schedule=, verbose=1),
    ]
    #model.fit(train_dataset, epochs=FLAGS.epochs, 
    #          steps_per_epoch=int(np.ceil(NUMBER_TRAIN_PAIRS / FLAGS.train_batch_size)), 
    #          callbacks=callbacks, validation_data=val_dataset, 
    #          validation_steps=int(np.ceil(NUMBER_VAL_PAIRS / FLAGS.val_batch_size)))

    model.fit(train_dataset, epochs=FLAGS.epochs, steps_per_epoch=5, callbacks=callbacks, 
              validation_data=val_dataset, validation_steps=10)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
