from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import inputs
import ofp
import time

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('checkpoint_dir', '/home/hhw/work/OFP/OFP_TF/train/train_id',
                    'Directory containing trained checkpoints.')
flags.DEFINE_string('dataset_split', 'val', 'Dataset split used to evaluate.')
flags.DEFINE_integer('batch_size', 16, 'Batch size.')
flags.DEFINE_boolean('is_training', False, 'Is training?')
flags.DEFINE_integer('eval_interval_secs', 300, 'Evaluation interval seconds.')


def eval():
    with tf.Graph().as_default() as g:
        with tf.device('/cpu:0'):
            image_a, image_b, labels, image_ida, image_idb = inputs.inputs(dataset_split=FLAGS.dataset_split,
                                                     is_training=FLAGS.is_training,
                                                     batch_size=FLAGS.batch_size,
                                                     num_epochs=1)
        logits, logits_ida, logits_idb = ofp.inference(image_a, image_b, FLAGS.is_training)
        # sigmoid cross entropy
        # logits = tf.squeeze(logits, axis=[-1])
        # logits = tf.sigmoid(logits)
        # predictions = tf.round(logits, name='prediction')

        # softmax cross entropy
        logits = tf.squeeze(logits)
        predictions = tf.argmax(logits, axis=-1)

        accuracy, update_op = tf.metrics.accuracy(labels=labels, predictions=predictions,
                                                  name='accuracy')

        saver = tf.train.Saver(tf.model_variables())
        local_init = tf.local_variables_initializer()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(local_init)
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, save_path=ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                print('Restore succeed.')
            else:
                print('No checkpoint file found.')

            print('Evaluating {} dataset.'.format(FLAGS.dataset_split))
            while True:
                try:
                    sess.run(update_op)
                except:
                    #raise
                    break

            print('accuracy', sess.run(accuracy))


def main(unused_argv):
    while True:
        eval()
        time.sleep(FLAGS.eval_interval_secs)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
