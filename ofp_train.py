from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
import ofp
import ofp_model
import inputs
import os
from datetime import datetime
import time
import shutil

slim = tf.contrib.slim
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('log_frequency', 10, 'Log frequency.')
flags.DEFINE_string('train_dir', './train/train', 'Training directory.')
flags.DEFINE_string('dataset_split', 'train', 'Using which dataset split to train the network.')
flags.DEFINE_integer('batch_size', 2,'Batch size used for train.')
flags.DEFINE_boolean('is_training', True, 'Is training?')
flags.DEFINE_float('initial_learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('decay_steps', 1000, 'Decay steps in exponential learning rate decay policy.')
flags.DEFINE_boolean('staircase', False, 'The parameter of learning rate decay policy.')
flags.DEFINE_float('decay_rate', 0.9, 'Decay rate in exponential learning rate decay policy.')
flags.DEFINE_integer('max_steps', 10000, 'Max training step.')
flags.DEFINE_integer('save_checkpoint_steps', 500, 'Save checkpoint steps.')
flags.DEFINE_string('restore_ckpt_path', '/home/hhw/work/OFP/OFP_TF/init_models/vgg_16.ckpt', 
                    'Path to checkpoint.')


def train(dataset_split):
    with tf.Graph().as_default() as g:
        global_step = tf.train.get_or_create_global_step()

        with tf.device('/cpu:0'):
            image_a, image_b, labels = inputs.inputs(dataset_split=dataset_split, 
                                                     is_training=FLAGS.is_training, 
                                                     batch_size=FLAGS.batch_size, 
                                                     num_epochs=None)
        tf.summary.image('image_a', image_a)
        tf.summary.image('image_b', image_b)

        logits = ofp.inference(image_a, image_b, FLAGS.is_training)
        total_loss = ofp.loss(logits, labels)
        tf.summary.histogram('logits', logits)
        tf.summary.scalar('total_loss', total_loss)

        learning_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                                   global_step, FLAGS.decay_steps,
                                                   FLAGS.decay_rate, staircase=FLAGS.staircase)
        tf.summary.scalar('learning_rate', learning_rate)

        for var in tf.model_variables():
            tf.summary.histogram(var.op.name, var)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(total_loss, global_step)

        # with tf.variable_scope('adam_vars'): 
        #     optimizer = tf.train.AdamOptimizer(learning_rate)
        #     update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        #     with tf.control_dependencies(update_ops):
        #         train_op = optimizer.minimize(total_loss, global_step)
        # adam_vars = optimizer.variables()

        def name_in_checkpoint(var):
            return var.op.name.replace(FLAGS.model_variant, 'vgg_16')
        #variables_to_restore = slim.get_variables_to_restore(exclude=models.EXCLUDE_LIST_MAP[FLAGS.model_variant]+['adam_vars'])
        #variables_to_restore = {name_in_checkpoint(var):var for var in variables_to_restore if not 'BatchNorm' in var.op.name}
        #variables_to_restore = {name_in_checkpoint(var):var for var in variables_to_restore}

        variables_to_restore = slim.get_variables_to_restore(exclude=['ofp', 'dense'])
        restorer = tf.train.Saver(variables_to_restore)

        def init_fn(scaffold, sess):
            restorer.restore(sess, FLAGS.restore_ckpt_path)

        # Define early stopping hook
        class EarlyStoppingHook(tf.train.SessionRunHook):
            def __init__(self, tolerance=0.00075):
                self.tolerance = tolerance

            # Initialize global and internal steps
            def begin(self):
                self._global_step_tensor = training_util._get_or_create_global_step_read()
                self._prev_step = -1
                self._step = 0

            # Evaluate early stopping loss every 1000 steps
            # (avoiding repetition when multiple run calls are made each step)
            def before_run(self, run_context):
                if (self._step % 1000 == 0) and (not self._step == self._prev_step):
                    graph = run_context.session.graph
                    loss_name = "stopping_loss:0"
                    loss_tensor = graph.get_tensor_by_name(loss_name)
                    return tf.train.SessionRunArgs({'step': self._global_step_tensor,
                                                            'loss': loss_tensor})
                else:
                    return tf.train.SessionRunArgs({'step': self._global_step_tensor})
                                                            
            # Check if current loss is below tolerance for early stopping
            def after_run(self, run_context, run_values):
                if (self._step % 1000 == 0) and (not self._step == self._prev_step):
                    global_step = run_values.results['step']
                    current_loss = run_values.results['loss']
                    if current_loss < self.tolerance:
                        print("[Early Stopping Criterion Satisfied]")
                        run_context.request_stop()
                    self._prev_step = global_step
                else:
                    global_step = run_values.results['step']
                    self._step = global_step


        class _LoggerHook(tf.train.SessionRunHook):
            def begin(self):
                # self._step = tf.train.get_or_create_global_step()
                self._step = -1
                self._start_time = time.time()

            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs(total_loss)  # Asks for loss value.

            def after_run(self, run_context, run_values):
                if self._step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value = run_values.results
                    examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                    sec_per_batch = float(duration / FLAGS.log_frequency)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                'sec/batch)')
                    print (format_str % (datetime.now(), self._step, loss_value,
                                    examples_per_sec, sec_per_batch))

        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True
        with tf.train.MonitoredTrainingSession(
            checkpoint_dir=FLAGS.train_dir,
            scaffold=tf.train.Scaffold(init_fn=init_fn),
            hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                   tf.train.NanTensorHook(total_loss),
                   _LoggerHook()],
            config=config,
            save_checkpoint_steps=FLAGS.save_checkpoint_steps) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)


def main(unused_argv):
    if os.path.exists(FLAGS.train_dir):
        shutil.rmtree(FLAGS.train_dir)
    if not os.path.exists(FLAGS.train_dir): 
        os.makedirs(FLAGS.train_dir)
    train(FLAGS.dataset_split)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
