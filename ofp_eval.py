from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import inputs
import ofp
import time
import math
import os

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('checkpoint_dir', './exp/train_id/train',
                    'Directory containing trained checkpoints.')
flags.DEFINE_string('eval_dir', './exp/train_id/eval',
                    'Directory containing trained checkpoints.')
flags.DEFINE_string('dataset_split', 'val', 'Dataset split used to evaluate.')
flags.DEFINE_integer('batch_size', 16, 'Batch size.')
flags.DEFINE_boolean('is_training', False, 'Is training?')
flags.DEFINE_integer('eval_interval_secs', 300, 'Evaluation interval seconds.')


def eval():
    with tf.Graph().as_default() as g:
        with tf.device('/cpu:0'):
            image_a, image_b, labels, image_ida, image_idb = inputs.inputs(
                dataset_split=FLAGS.dataset_split, is_training=FLAGS.is_training, 
                batch_size=FLAGS.batch_size, num_epochs=1)
        logits, logits_ida, logits_idb = ofp.inference(
            image_a, image_b, FLAGS.is_training)

        logits = tf.squeeze(logits)
        predictions = tf.argmax(logits, axis=-1)

        accuracy, update_op = tf.metrics.accuracy(
            labels=labels, predictions=predictions, name='acc')
        summary_op = tf.summary.scalar('accuracy', accuracy)
        
        num_batches = int(math.ceil(inputs.NUMBER_VAL_PAIRS / float(FLAGS.batch_size)))

        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            print('Get global_step from checkpoint name.')
        else:
            global_step = tf.train.get_or_create_global_step()
            print('Create global_step.')

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.train.MonitoredSession(
            session_creator=tf.train.ChiefSessionCreator(
                checkpoint_dir=FLAGS.checkpoint_dir,
                config=config
            )
        ) as mon_sess:
            print('Evaluating {} dataset.'.format(FLAGS.dataset_split))
            for _ in range(num_batches):
                mon_sess.run(update_op)
            
            summary = mon_sess.run(summary_op)
            summary_writer = tf.summary.FileWriter(
                logdir=FLAGS.eval_dir, graph=mon_sess.graph)
            summary_writer.add_summary(summary, global_step=global_step)
            print('accuracy', mon_sess.run(accuracy))


def main(unused_argv):
    if not os.path.exists(FLAGS.eval_dir):
        os.makedirs(FLAGS.eval_dir)
    while True:
        eval()
        time.sleep(FLAGS.eval_interval_secs)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
