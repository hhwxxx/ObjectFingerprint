from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import pandas as pd
from itertools import combinations
from sklearn.model_selection import train_test_split
import random
import os
import glob

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('image_folder', '/home/hhw/work/OFP/OFP-project/data/ring', 
                    'Folder containing images.')
flags.DEFINE_string('filelist', '/home/hhw/work/OFP/OFP-project/data/ring/index.txt', 
                    'Folder containing lists for training and validation.')
flags.DEFINE_string('output_dir', './tfrecords', 'Directory to save tfrecord.')
flags.DEFINE_string('image_format', 'JPG', 'Image format')

NUMBER_INDEX = 110
NUMBER_VAL_INDEX = 20


class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self, image_format='jpeg', channels=3):
        """Class constructor.
        Args:
          image_format: Image format. Only 'jpeg', 'jpg', or 'png' are supported.
          channels: Image channels.
        """
        with tf.Graph().as_default():
          self._decode_data = tf.placeholder(dtype=tf.string)
          self._image_format = image_format
          self._session = tf.Session()
          if self._image_format in ('jpeg', 'jpg', 'JPG'):
            self._decode = tf.image.decode_jpeg(self._decode_data,
                                                channels=channels)
          elif self._image_format == 'png':
            self._decode = tf.image.decode_png(self._decode_data,
                                              channels=channels)


    def read_image_dims(self, image_data):
        """Reads the image dimensions.
        Args:
          image_data: string of image data.
        Returns:
          image_height and image_width.
        """
        image = self.decode_image(image_data)

        return image.shape[:2]


    def decode_image(self, image_data):
        """Decodes the image data string.
        Args:
          image_data: string of image data.
        Returns:
          Decoded image data.
        Raises:
          ValueError: Value of image channels not supported.
        """
        image = self._session.run(self._decode,
                                  feed_dict={self._decode_data: image_data})
        if len(image.shape) != 3 or image.shape[2] not in (1, 3):
          raise ValueError('The image channels not supported.')

        return image


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def build_tfrecord(dataset_split, index):
    print('Processing {} data.'.format(dataset_split))

    data_list = pd.read_csv(FLAGS.filelist, sep=' ', header=None)
    data_list.columns = ['name', 'info', 'index']

    # data_list of dataset_split
    data_list = data_list[data_list['index'].isin(index)]
    combination = list(combinations(range(0, len(data_list)), 2))

    matched_pairs = [c for c in combination if data_list.iloc[c[0]][-1 ] == data_list.iloc[c[1]][-1]]
    unmatched_pairs = [c for c in combination if data_list.iloc[c[0]][-1] != data_list.iloc[c[1]][-1]]
    unmatched_pairs = random.sample(unmatched_pairs, len(matched_pairs) * 3)
    print('Number of matched pairs: {}\nNumber of unmatched pairs: {}'.format(
          len(matched_pairs), len(unmatched_pairs)))
    pairs = matched_pairs + unmatched_pairs
    random.shuffle(pairs)
    num_pairs = len(pairs)
    print('Total number of pairs: {}'.format(num_pairs))

    image_reader = ImageReader(image_format=FLAGS.image_format, channels=3)

    output_filename = os.path.join(FLAGS.output_dir, dataset_split + '_mini.tfrecord')
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
        for i in range(20):
            a, b = pairs[i]
            image_a_name = data_list.iloc[a]['name']
            image_a_index = data_list.iloc[a]['index']
            image_a_filename = os.path.join(FLAGS.image_folder, image_a_name)
            image_a_data = tf.gfile.FastGFile(image_a_filename, 'rb').read()
            image_a_height, image_a_width = image_reader.read_image_dims(image_a_data)

            image_b_name = data_list.iloc[b]['name']
            image_b_index = data_list.iloc[b]['index']
            image_b_filename = os.path.join(FLAGS.image_folder, image_b_name)
            image_b_data = tf.gfile.FastGFile(image_b_filename, 'rb').read()
            image_b_height, image_b_width = image_reader.read_image_dims(image_b_data)

            label = 1 if (image_a_index == image_b_index) else 0
            
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'image_a_data': _bytes_feature(image_a_data),
                        'image_a_height': _int64_feature(image_a_height),
                        'image_a_width': _int64_feature(image_a_width),
                        'image_a_channel': _int64_feature(3),
                        'image_a_index': _int64_feature(image_a_index),
                        'image_a_name': _bytes_feature(image_a_name),
                        'image_a_format': _bytes_feature(FLAGS.image_format),
                        'image_b_data': _bytes_feature(image_b_data),
                        'image_b_height': _int64_feature(image_b_height),
                        'image_b_width': _int64_feature(image_b_width),
                        'image_b_channel': _int64_feature(3),
                        'image_b_index': _int64_feature(image_b_index),
                        'image_b_name': _bytes_feature(image_b_name),
                        'image_b_format': _bytes_feature(FLAGS.image_format),
                        'label': _int64_feature(label),
                    }
                )
            )
            tfrecord_writer.write(example.SerializeToString())
    print('Finished processing {} data.'.format(dataset_split))
  


def main(unused_argv):
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
    
    train_index, val_index = train_test_split(range(0, NUMBER_INDEX), test_size=NUMBER_VAL_INDEX, shuffle=True)
    build_tfrecord(dataset_split='train', index=train_index)
    build_tfrecord(dataset_split='val', index=val_index)
    
    print('Finished.')


if __name__ == '__main__':
    tf.app.run()
