# -*- coding:utf-8 -*-

import os
import time
import datetime
import tensorflow as tf
import data_helpers
from text_cnn import TextCNN

logger = data_helpers.logger_fn('tflog', 'training-{}.log'.format(time.asctime()))

# Parameters
# ==================================================

TRAININGSET_DIR = 'Train.json'
VALIDATIONSET_DIR = 'Test.json'

# Data loading params
tf.flags.DEFINE_string("training_data_file", TRAININGSET_DIR, "Data source for the training data.")
tf.flags.DEFINE_string("validation_data_file", VALIDATIONSET_DIR, "Data source for the validation data.")

# Model Hyperparameterss
tf.flags.DEFINE_integer("pad_seq_len", 150, "Recommand padding Sequence length of data (depends on the data)")
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("embedding_type", 1, "The embedding type (default: 1)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 500, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("num_classes", 367, "Number of labels (depends on the task)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("gpu_options_allow_growth", True, "Allow gpu options growth")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
dilim = '-' * 100
logger.info('\n'.join([dilim, *['{:>50}|{:<50}'.format(attr.upper(), value)
                                for attr, value in sorted(FLAGS.__flags.items())], dilim]))


def train_cnn():
    """Training CNN model."""
    # Load sentences, labels, and training parameters
    logger.info('✔︎ Loading data...')

    logger.info('✔︎ Training data processing...')
    train_data = \
        data_helpers.load_data_and_labels(FLAGS.training_data_file, FLAGS.num_classes, FLAGS.embedding_dim)

    logger.info('✔︎ Validation data processing...')
    validation_data = \
        data_helpers.load_data_and_labels(FLAGS.validation_data_file, FLAGS.num_classes, FLAGS.embedding_dim)

    logger.info('Recommand padding Sequence length is: {}'.format(FLAGS.pad_seq_len))

    logger.info('✔︎ Training data padding...')
    x_train, y_train = \
        data_helpers.pad_data(train_data, FLAGS.pad_seq_len)

    logger.info('✔︎ Validation data padding...')
    x_validation, y_validation = \
        data_helpers.pad_data(validation_data, FLAGS.pad_seq_len)

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
                sequence_length=FLAGS.pad_seq_len,
                num_classes=FLAGS.num_classes,
                vocab_size=VOCAB_SIZE,
                embedding_size=FLAGS.embedding_dim,
                embedding_type=FLAGS.embedding_type,
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
            logger.info("✔︎ Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)

            # acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Validation summaries
            validation_summary_op = tf.summary.merge([loss_summary])
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
                _, step, summaries, loss = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss], feed_dict)
                time_str = datetime.datetime.now().isoformat()
                logger.info("{}: step {}, loss {:g}"
                                 .format(time_str, step, loss))
                train_summary_writer.add_summary(summaries, step)

            def validation_step(x_validation, y_validation, writer=None):
                """Evaluates model on a validation set"""
                batches_validation = data_helpers.batch_iter(
                    list(zip(x_validation, y_validation)), 8 * FLAGS.batch_size, FLAGS.num_epochs)
                eval_loss, eval_rec, eval_acc, eval_counter = 0.0, 0.0, 0.0, 0
                for batch_validation in batches_validation:
                    x_batch_validation, y_batch_validation = zip(*batch_validation)
                    feed_dict = {
                        cnn.input_x: x_batch_validation,
                        cnn.input_y: y_batch_validation,
                        cnn.dropout_keep_prob: 1.0
                    }
                    step, summaries, logits, cur_loss = sess.run(
                        [global_step, validation_summary_op, cnn.logits, cnn.loss], feed_dict)

                    predicted_labels = data_helpers.get_label_using_logits(logits, top_number=1)
                    cur_rec, cur_acc = 0.0, 0.0
                    for index, predicted_label in enumerate(predicted_labels):
                        rec_inc, acc_inc = data_helpers.cal_rec_and_acc(predicted_label, y_batch_validation[index])
                        cur_rec, cur_acc = cur_rec + rec_inc, cur_acc + acc_inc

                    cur_rec = cur_rec / len(y_batch_validation)
                    cur_acc = cur_acc / len(y_batch_validation)
                    
                    eval_loss, eval_rec, eval_acc, eval_counter = eval_loss + cur_loss, eval_rec + cur_rec, \
                                                                  eval_acc + cur_acc, eval_counter + 1
                    logger.info("✔︎ validation batch {} finished.".format(eval_counter))

                    if writer:
                        writer.add_summary(summaries, step)

                eval_loss = float(eval_loss / eval_counter)
                eval_rec = float(eval_rec / eval_counter)
                eval_acc = float(eval_acc / eval_counter)

                return eval_loss, eval_rec, eval_acc

            # Generate batches
            batches_train = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

            # Training loop. For each batch...
            for batch_train in batches_train:
                x_batch_train, y_batch_train = zip(*batch_train)
                train_step(x_batch_train, y_batch_train)
                current_step = tf.train.global_step(sess, global_step)

                if current_step % FLAGS.evaluate_every == 0:
                    logger.info("\nEvaluation:")
                    eval_loss, eval_rec, eval_acc = validation_step(x_validation, y_validation, writer=validation_summary_writer)
                    time_str = datetime.datetime.now().isoformat()
                    logger.info("{}: step {}, loss {:g}, rec {:g}, acc {:g}"
                                     .format(time_str, current_step, eval_loss, eval_rec, eval_acc))

                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    logger.info("✔︎ Saved model checkpoint to {}\n".format(path))

    logger.info("✔︎ Done.")


if __name__ == '__main__':
    train_cnn()
