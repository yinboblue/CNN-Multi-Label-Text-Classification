# -*- coding:utf-8 -*-

import os
import time
import datetime
import logging
import tensorflow as tf
import data_helpers
from text_cnn import TextCNN

logging.getLogger().setLevel(logging.INFO)

# Parameters
# ==================================================

FLAGS = tf.flags.FLAGS
BASE_DIR = os.getcwd()

TRAININGSET_DIR = BASE_DIR + '/Train_content.json'
VALIDATIONSET_DIR = BASE_DIR + '/Test_content.json'

# Data loading params
tf.flags.DEFINE_string("training_data_file", TRAININGSET_DIR, "Data source for the training data.")
tf.flags.DEFINE_string("validation_data_file", VALIDATIONSET_DIR, "Data source for the validation data.")

# Model Hyperparameterss
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("gpu_options_allow_growth", True, "Allow gpu options growth")


def train_cnn():
    """Training CNN model."""

    # Load sentences, labels, and training parameters
    logging.info('✔︎ Loading data...')

    logging.info('✔︎ Training data processing...')
    train_data, train_data_max_seq_len = \
        data_helpers.load_data_and_labels(FLAGS.training_data_file, FLAGS.embedding_dim)

    logging.info('✔︎ Validation data processing...')
    validation_data, validation_data_max_seq_len = \
        data_helpers.load_data_and_labels(FLAGS.validation_data_file, FLAGS.embedding_dim)

    MAX_SEQUENCE_LENGTH = max(train_data_max_seq_len, validation_data_max_seq_len)
    logging.info('Max sequence length is: {}'.format(MAX_SEQUENCE_LENGTH))

    logging.info('✔︎ Training data padding...')
    x_train_front, x_train_behind, y_train = \
        data_helpers.pad_data(train_data, MAX_SEQUENCE_LENGTH)

    logging.info('✔︎ Validation data padding...')
    x_validation_front, x_validation_behind, y_validation = \
        data_helpers.pad_data(validation_data, MAX_SEQUENCE_LENGTH)

    # Build vocabulary
    VOCAB_SIZE = data_helpers.load_vocab_size(FLAGS.embedding_dim)
    pretrained_word2vec_matrix = data_helpers.load_word2vec_matrix(VOCAB_SIZE, FLAGS.embedding_dim)

    # Build a graph and cnn object
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = FLAGS.gpu_options_allow_growth
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=MAX_SEQUENCE_LENGTH,
                num_classes=y_train.shape[1],
                vocab_size=VOCAB_SIZE,
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                pretrained_embedding=pretrained_word2vec_matrix)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step, name="train_op")

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            logging.info("✔︎ Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Validation summaries
            validation_summary_op = tf.summary.merge([loss_summary, acc_summary])
            validation_summary_dir = os.path.join(out_dir, "summaries", "validation")
            validation_summary_writer = tf.summary.FileWriter(validation_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            def train_step(x_batch, y_batch):
                """A single training step"""
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy], feed_dict)
                time_str = datetime.datetime.now().isoformat()
                logging.critical("{}: step {}, loss {:g}, acc {:g}"
                                 .format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def validation_step(x_batch, y_batch, writer=None):
                """Evaluates model on a validation set"""
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: 1.0
                }
                step, summaries, scores, predictions, topKPreds, loss, accuracy, auc = sess.run(
                    [global_step, validation_summary_op, cnn.scores, cnn.predictions,
                     cnn.topKPreds, cnn.loss, cnn.accuracy, cnn.AUC], feed_dict)
                time_str = datetime.datetime.now().isoformat()
                logging.critical("{}: step {}, loss {:g}, acc {:g}, AUC {}"
                                 .format(time_str, step, loss, accuracy, auc))
                if writer:
                    writer.add_summary(summaries, step)

            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train_front, x_train_behind, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch_front, x_batch_behind, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    logging.info("\nEvaluation:")
                    validation_step(x_validation, y_validation,
                                    writer=validation_summary_writer)
                    logging.info("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    logging.critical("✔︎ Saved model checkpoint to {}\n".format(path))

    logging.info("✔︎ Done.")


if __name__ == '__main__':
    train_cnn()
