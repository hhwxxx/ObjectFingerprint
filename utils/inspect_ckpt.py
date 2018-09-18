from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as chkp

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('ckpt_dir', '/home/hhw/test/fcn/train', 'Checkpoint directory.')
flags.DEFINE_string('checkpoint', '/home/hhw/work/OFP/OFP_TF/init_models/vgg_16.ckpt', 'Checkpoint directory.')


def inspect_ckpt(ckpt_dir):
    # latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        # 'all_tensors' indicates whether to print all tensors (include tensor name and value).
        # 'all_tensor_names' indicates whether to print all tensor names.
        # If both arguments are False, then print the tensor names and shapes in the checkpoint file.
        chkp.print_tensors_in_checkpoint_file(ckpt.model_checkpoint_path,
                                              tensor_name='',
                                              all_tensors=False,
                                              all_tensor_names=True)
    else:
        print('No checkpoint file found.')


def inspect_ckpt_v2(checkpoint):
    # 'all_tensors' indicates whether to print all tensors (include tensor name and value).
    # 'all_tensor_names' indicates whether to print all tensor names.
    # If both arguments are False, then print the tensor names and shapes in the checkpoint file.
    chkp.print_tensors_in_checkpoint_file(checkpoint,
                                          tensor_name='',
                                          all_tensors=False,
                                          all_tensor_names=True)

def main(unused_argv):
    #inspect_ckpt(FLAGS.ckpt_dir)
    inspect_ckpt_v2(FLAGS.checkpoint)



if __name__ == '__main__':
    tf.app.run()
