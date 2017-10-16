# -*- coding:utf-8 -*-

import os
import time
import datetime
import logging
import tensorflow as tf
import data_helpers
from text_cnn import TextCNN

# Parameters
# ==================================================

TRAIN_OR_RESTORE = input("☛ Train or Restore?(T/R) \n")

while not (TRAIN_OR_RESTORE.isalpha() and TRAIN_OR_RESTORE.upper() in ['T', 'R']):
    TRAIN_OR_RESTORE = input('✘ The format of your input is illegal, please re-input: ')
logging.info('✔︎ The format of your input is legal, now loading to next step...')

TRAIN_OR_RESTORE = TRAIN_OR_RESTORE.upper()

CLASS_BIND = input("☛ Use Class Bind or Not?(Y/N) \n")
while not (CLASS_BIND.isalpha() and CLASS_BIND.upper() in ['Y', 'N']):
    CLASS_BIND = input('✘ The format of your input is illegal, please re-input: ')
logging.info('✔︎ The format of your input is legal, now loading to next step...')

CLASS_BIND = CLASS_BIND.upper()

if TRAIN_OR_RESTORE == 'T':
    logger = data_helpers.logger_fn('tflog', 'training-{}.log'.format(time.asctime()))
if TRAIN_OR_RESTORE == 'R':
    logger = data_helpers.logger_fn('tflog', 'restore-{}.log'.format(time.asctime()))

TRAININGSET_DIR = 'Train.json'
VALIDATIONSET_DIR = 'Validation_bind.json'

# Data loading params
tf.flags.DEFINE_string("training_data_file", TRAININGSET_DIR, "Data source for the training data.")
tf.flags.DEFINE_string("validation_data_file", VALIDATIONSET_DIR, "Data source for the validation data.")
tf.flags.DEFINE_string("train_or_restore", TRAIN_OR_RESTORE, "Train or Restore.")
tf.flags.DEFINE_string("use_classbind_or_not", CLASS_BIND, "Use the class bind info or not.")

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
tf.flags.DEFINE_integer("evaluate_every", 5000, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("num_classes", 367, "Number of labels (depends on the task)")
tf.flags.DEFINE_integer("top_num", 2, "Number of top K prediction classess (default: 3)")

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
    train_data = data_helpers.load_data_and_labels(FLAGS.training_data_file, FLAGS.num_classes, FLAGS.embedding_dim)

    logger.info('✔︎ Validation data processing...')
    validation_data = \
        data_helpers.load_data_and_labels(FLAGS.validation_data_file, FLAGS.num_classes, FLAGS.embedding_dim)

    logger.info('Recommand padding Sequence length is: {}'.format(FLAGS.pad_seq_len))

    logger.info('✔︎ Training data padding...')
    x_train, y_train = data_helpers.pad_data(train_data, FLAGS.pad_seq_len)

    logger.info('✔︎ Validation data padding...')
    x_validation, y_validation = data_helpers.pad_data(validation_data, FLAGS.pad_seq_len)

    y_validation_bind = validation_data.labels_bind

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
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=cnn.global_step, name="train_op")

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
            if FLAGS.train_or_restore == 'R':
                MODEL = input("☛ Please input the checkpoints model you want to restore, "
                              "it should be like(1490175368): ")  # The model you want to restore

                while not (MODEL.isdigit() and len(MODEL) == 10):
                    MODEL = input('✘ The format of your input is illegal, please re-input: ')
                logger.info('✔︎ The format of your input is legal, now loading to next step...')

                checkpoint_dir = 'runs/' + MODEL + '/checkpoints/'

                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", MODEL))
                logger.info("✔︎ Writing to {}\n".format(out_dir))
            else:
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

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            if FLAGS.train_or_restore == 'R':
                # Load cnn model
                logger.info("✔ Loading model...")
                checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
                logger.info(checkpoint_file)

                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)
            else:
                checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())

            current_step = sess.run(cnn.global_step)

            def train_step(x_batch, y_batch):
                """A single training step"""
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss = sess.run(
                    [train_op, cnn.global_step, train_summary_op, cnn.loss], feed_dict)
                time_str = datetime.datetime.now().isoformat()
                logger.info("{}: step {}, loss {:g}"
                                 .format(time_str, step, loss))
                train_summary_writer.add_summary(summaries, step)

            def validation_step(x_validation, y_validation, y_validation_bind, writer=None):
                """Evaluates model on a validation set"""
                batches_validation = data_helpers.batch_iter(
                    list(zip(x_validation, y_validation, y_validation_bind)), 8 * FLAGS.batch_size, FLAGS.num_epochs)
                eval_loss, eval_rec, eval_acc, eval_counter = 0.0, 0.0, 0.0, 0
                for batch_validation in batches_validation:
                    x_batch_validation, y_batch_validation, y_batch_validation_bind = zip(*batch_validation)
                    feed_dict = {
                        cnn.input_x: x_batch_validation,
                        cnn.input_y: y_batch_validation,
                        cnn.dropout_keep_prob: 1.0
                    }
                    step, summaries, logits, cur_loss = sess.run(
                        [cnn.global_step, validation_summary_op, cnn.logits, cnn.loss], feed_dict)

                    if FLAGS.use_classbind_or_not == 'Y':
                        predicted_labels = data_helpers.get_label_using_logits_and_classbind(
                            logits, y_batch_validation_bind, top_number=FLAGS.top_num)
                    if FLAGS.use_classbind_or_not == 'N':
                        predicted_labels = data_helpers.get_label_using_logits(logits, top_number=FLAGS.top_num)

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
                current_step = tf.train.global_step(sess, cnn.global_step)

                if current_step % FLAGS.evaluate_every == 0:
                    logger.info("\nEvaluation:")
                    eval_loss, eval_rec, eval_acc = validation_step(x_validation, y_validation, y_validation_bind,
                                                                    writer=validation_summary_writer)
                    time_str = datetime.datetime.now().isoformat()
                    logger.info("{}: step {}, loss {:g}, rec {:g}, acc {:g}"
                                .format(time_str, current_step, eval_loss, eval_rec, eval_acc))

                if current_step % FLAGS.checkpoint_every == 0:
                    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    logger.info("✔︎ Saved model checkpoint to {}\n".format(path))

    logger.info("✔︎ Done.")


if __name__ == '__main__':
    train_cnn()
