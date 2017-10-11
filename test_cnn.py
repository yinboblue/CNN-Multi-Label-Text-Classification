# -*- coding:utf-8 -*-

import os
import time
import numpy as np
import tensorflow as tf
import data_helpers

logger = data_helpers.logger_fn('tflog', 'test-{}.log'.format(time.asctime()))

MODEL = input("☛ Please input the model file you want to test, it should be like(1490175368): ")

while not (MODEL.isdigit() and len(MODEL) == 10):
    SUBSET = input('✘ The format of your input is illegal, it should be like(1490175368), please re-input: ')
logger.info('✔︎ The format of your input is legal, now loading to next step...')

TRAININGSET_DIR = 'Train.json'
VALIDATIONSET_DIR = 'Validation.json'
TESTSET_DIR = 'Test.json'
MODEL_DIR = 'runs/' + MODEL + '/checkpoints/'
SAVE_FILE = 'predictions.txt'

# Data loading params
tf.flags.DEFINE_string("training_data_file", TRAININGSET_DIR, "Data source for the training data.")
tf.flags.DEFINE_string("validation_data_file", VALIDATIONSET_DIR, "Data source for the validation data")
tf.flags.DEFINE_string("test_data_file", TESTSET_DIR, "Data source for the test data")
tf.flags.DEFINE_string("checkpoint_dir", MODEL_DIR, "Checkpoint directory from training run")

# Model Hyperparameters
tf.flags.DEFINE_integer("pad_seq_len", 150, "Recommand padding Sequence length of data (depends on the data)")
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("embedding_type", 1, "The embedding type (default: 1)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
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


def test_cnn():
    """Test CNN model."""

    # Load data
    logger.info("✔ Loading data...")

    logger.info('Recommand padding Sequence length is: {}'.format(FLAGS.pad_seq_len))

    logger.info('✔︎ Test data processing...')

    test_data = data_helpers.load_data_and_labels(FLAGS.test_data_file, FLAGS.num_classes, FLAGS.embedding_dim)

    logger.info('✔︎ Test data padding...')

    x_test, y_test = data_helpers.pad_data(test_data, FLAGS.pad_seq_len)

    y_test_bind = test_data.labels_bind

    # Build vocabulary
    VOCAB_SIZE = data_helpers.load_vocab_size(FLAGS.embedding_dim)
    pretrained_word2vec_matrix = data_helpers.load_word2vec_matrix(VOCAB_SIZE, FLAGS.embedding_dim)

    # Load cnn model
    logger.info("✔ Loading model...")
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    logger.info(checkpoint_file)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = FLAGS.gpu_options_allow_growth
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]

            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # pre-trained_word2vec
            pretrained_embedding = graph.get_operation_by_name("embedding/W").outputs[0]

            # Tensors we want to evaluate
            logits = graph.get_operation_by_name("output/logits").outputs[0]

            # Generate batches for one epoch
            batches = data_helpers.batch_iter(list(zip(x_test, y_test, y_test_bind)),
                                              FLAGS.batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predicitons = []
            eval_loss, eval_rec, eval_acc, eval_counter = 0.0, 0.0, 0.0, 0
            for batch_test in batches:
                x_batch_test, y_batch_test, y_batch_test_bind = zip(*batch_test)
                feed_dict = {
                    input_x: x_batch_test,
                    dropout_keep_prob: 1.0
                }
                batch_logits = sess.run(logits, feed_dict)
                predicted_labels = data_helpers.get_label_using_logits(batch_logits, y_batch_test_bind,
                                                                       top_number=FLAGS.top_num)
                all_predicitons = np.append(all_predicitons, predicted_labels)
                cur_rec, cur_acc = 0.0, 0.0
                for index, predicted_label in enumerate(predicted_labels):
                    rec_inc, acc_inc = data_helpers.cal_rec_and_acc(predicted_label, y_batch_test[index])
                    cur_rec, cur_acc = cur_rec + rec_inc, cur_acc + acc_inc

                cur_rec = cur_rec / len(y_batch_test)
                cur_acc = cur_acc / len(y_batch_test)

                eval_rec, eval_acc, eval_counter = eval_rec + cur_rec, eval_acc + cur_acc, eval_counter + 1
                logger.info("✔︎ validation batch {} finished.".format(eval_counter))

            eval_rec = float(eval_rec / eval_counter)
            eval_acc = float(eval_acc / eval_counter)
            logger.info("☛ Recall {:g}, Accuracy {:g}".format(eval_rec, eval_acc))
            np.savetxt(SAVE_FILE, list(zip(all_predicitons)), fmt='%s')

    logger.info("✔ Done.")


if __name__ == '__main__':
    test_cnn()
