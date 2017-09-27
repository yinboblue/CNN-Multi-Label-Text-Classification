# -*- coding:utf-8 -*-

import os
import multiprocessing
import gensim
import logging
import json
import numpy as np

from operator import itemgetter
from pylab import *
from gensim.models import word2vec
from tflearn.data_utils import pad_sequences


TEXT_DIR = 'content.txt'


def logger_fn(name, file, level=logging.INFO):
    tf_logger = logging.getLogger(name)
    tf_logger.setLevel(level)
    fh = logging.FileHandler(file, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    tf_logger.addHandler(fh)
    return tf_logger


def get_label_using_logits(logits, bind, top_number=1):
    labels_bind =[]
    predicted_labels = []
    logits = np.ndarray.tolist(logits)
    for index, item in enumerate(bind):
        result = []
        for i in item:
            result.append((i, logits[index][i]))
        labels_bind.append(result)
    for item in labels_bind:
        result = []
        index_list = sorted(item, key=itemgetter(1), reverse=True)
        index_list = index_list[:top_number]
        for label in index_list:
            result.append(label[0])
        predicted_labels.append(result)
    return predicted_labels


def cal_rec_and_acc(predicted_labels, labels):
    label_no_zero = []
    for index, label in enumerate(labels):
        if int(label) == 1:
            label_no_zero.append(index)
    count = 0
    logging.info("predicted_labels:{}, origin_labels: {}".format(predicted_labels, label_no_zero))
    for predicted_label in predicted_labels:
        if int(predicted_label) in label_no_zero:
            count += 1
    rec = count / len(label_no_zero)
    acc = count / len(predicted_labels)
    return rec, acc


def create_word2vec_model(embedding_size, input_file=TEXT_DIR):
    """
    Create the word2vec model based on the given embedding size and the corpus file.
    :param embedding_size: The embedding size
    :param input_file: The corpus file
    """
    word2vec_file = 'word2vec_' + str(embedding_size) + '.model'

    if os.path.isfile(word2vec_file):
        logging.info('☛ The word2vec model you want create already exists!')
    else:
        sentences = word2vec.LineSentence(input_file)
        # sg=0 means use CBOW model(default); sg=1 means use skip-gram model.
        model = gensim.models.Word2Vec(sentences, size=embedding_size, min_count=0,
                                       sg=0, workers=multiprocessing.cpu_count())
        model.save(word2vec_file)


def load_vocab_size(embedding_size):
    """
    Return the vocab size of the word2vec file.
    :param embedding_size: The embedding size
    :return: The vocab size of the word2vec file
    """
    word2vec_file = 'word2vec_' + str(embedding_size) + '.model'

    if os.path.isfile(word2vec_file):
        model = word2vec.Word2Vec.load(word2vec_file)
        return len(model.wv.vocab.items())
    else:
        logging.info("✘ The word2vec file doesn't exist. "
                     "Please use function <create_vocab_size(embedding_size)> to create it!")


def data_augmented(data_tokenindex, data_labels):
    """Data augmented"""
    aug_data = []
    aug_label = []
    aug_num = 0
    for i in range(len(data_tokenindex)):
        data_record = data_tokenindex[i]
        if len(data_record) == 1:  # 句子长度为 1，则不进行增广
            continue
        elif len(data_record) == 2:  # 句子长度为 2，则交换两个词的顺序
            data_record[0], data_record[1] = data_record[1], data_record[0]
            aug_data.append(data_record)
            aug_label.append(data_labels[i])
            aug_num += 1
        else:
            data_record = np.array(data_record)
            for num in range(len(data_record) - 1):  # 打乱词的次数，次数即生成样本的个数；次数根据句子长度而定
                data_shuffled = np.random.permutation(np.arange(len(data_record)))
                new_data_record = data_record[data_shuffled]
                aug_data.append(list(new_data_record))
                aug_label.append(data_labels[i])
                aug_num += 1

    class AugData:
        def __init__(self):
            pass

        @property
        def number(self):
            return aug_num

        @property
        def labels(self):
            return aug_label

        @property
        def tokenindex(self):
            return aug_data

    return AugData()


def data_word2vec(input_file, num_labels, word2vec_model):
    """
    Create the research data tokenindex based on the word2vec model file.
    Returns the class Data(includes the data tokenindex and data labels).
    :param input_file: The research data
    :param word2vec_model: The word2vec model file
    :return: The class Data(includes the data tokenindex and data labels)
    """

    vocab = dict([(k, v.index) for (k, v) in word2vec_model.wv.vocab.items()])

    def token_to_index(content):
        result = []
        for item in content:
            id = vocab.get(item)
            if id is None:
                id = 0
            result.append(id)
        return result

    def create_label(label_index):
        label = [0] * num_labels
        for item in label_index:
            label[int(item)] = 1
        return label

    if input_file.endswith('.json'):
        with open(input_file) as fin:
            content_indexlist = []
            labels = []
            labels_bind = []
            for index, eachline in enumerate(fin):
                content = []
                data = json.loads(eachline)
                features_content = data['features_content'].strip().split()
                label_index = data['knows_index'].strip().split()

                for item in features_content:
                    content.append(item)

                labels.append(create_label(label_index))
                content_indexlist.append(token_to_index(content))

                if 'knows_bind' in data.keys():
                    labels_bind.append(data['knows_bind'])

            total_line = index + 1

        class Data:
            def __init__(self):
                pass

            @property
            def number(self):
                return total_line

            @property
            def tokenindex(self):
                return content_indexlist

            @property
            def labels(self):
                return labels

            @property
            def labels_bind(self):
                if labels_bind:
                    return labels_bind
                else:
                    return None

        return Data()
    else:
        logging.info('✘ The research data is not a json file. '
                     'Please preprocess the research data into the json file.')


def load_word2vec_matrix(vocab_size, embedding_size):
    """
    Return the word2vec model matrix.
    :param vocab_size: The vocab size of the word2vec model file
    :param embedding_size: The embedding size
    :return: The word2vec model matrix
    """
    word2vec_file = 'word2vec_' + str(embedding_size) + '.model'

    if os.path.isfile(word2vec_file):
        model = gensim.models.Word2Vec.load(word2vec_file)
        vocab = dict([(k, v.index) for k, v in model.wv.vocab.items()])
        vector = np.zeros([vocab_size, embedding_size])
        for key, value in vocab.items():
            if len(key) > 0:
                vector[value] = model[key]
        return vector
    else:
        logging.info("✘ The word2vec file doesn't exist. "
                     "Please use function <create_vocab_size(embedding_size)> to create it!")


def load_data_and_labels(data_file, num_labels, embedding_size):
    """
    Loads research data from files, splits the data into words and generates labels.
    Returns split sentences, labels and the max sentence length of the research data.
    :param data_file: The research data
    :param embedding_size: The embedding size
    :returns: The class data and the max sentence length of the research data
    """
    word2vec_file = 'word2vec_' + str(embedding_size) + '.model'

    # Load word2vec model file
    if os.path.isfile(word2vec_file):
        model = word2vec.Word2Vec.load(word2vec_file)
    else:
        create_word2vec_model(embedding_size, TEXT_DIR)


    # Load data from files and split by words
    data = data_word2vec(input_file=data_file, num_labels=num_labels, word2vec_model=model)
    # aug_data = data_augmented(data_tokenindex=data.tokenindex, data_labels=data.labels)

    plot_seq_len(data_file, data)

    logging.info('Found {} texts.'.format(data.number))
    # logging.info('Augmented {} texts.'.format(aug_data.number))

    return data


def pad_data(data, pad_seq_len):
    """
    Padding each sentence of research data according to the max sentence length.
    Returns the padded data and data labels.
    :param data: The research data
    :param max_seq_len: The max sentence length of research data
    :returns: The padded data and data labels
    """
    pad_data = pad_sequences(data.tokenindex, maxlen=pad_seq_len, value=0.)
    labels = data.labels
    return pad_data, labels


def plot_seq_len(data_file, data, percentage=0.98):
    output_file = data_file.split('.')[0] + ' Sequence Length Distribution Histogram.png'
    result = dict()
    for x in data.tokenindex:
        if len(x) not in result.keys():
            result[len(x)] = 1
        else:
            result[len(x)] += 1
    freq_seq = [(key, result[key]) for key in sorted(result.keys())]
    x = []
    y = []
    avg = 0
    count = 0
    border_index = []
    print(data.number)
    for item in freq_seq:
        x.append(item[0])
        y.append(item[1])
        avg += item[0] * item[1]
        count += item[1]
        if count > data.number * percentage:
            border_index.append(item[0])
    avg = avg / data.number
    logging.info('The average of the data sequence length is {}'.format(avg))
    logging.info('The recommend of padding sequence length should more than {}'.format(border_index[0]))
    xlim(0, 200)
    plt.bar(x, y)
    plt.savefig(output_file)
    plt.close()


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    含有 yield 说明不是一个普通函数，是一个 Generator.
    函数效果：对 data，一共分成 num_epochs 个阶段（epoch），在每个 epoch 内，如果 shuffle=True，就将 data 重新洗牌，
    批量生成 (yield) 一批一批的重洗过的data，每批大小是 batch_size，一共生成 int(len(data)/batch_size)+1 批。
    Generate a  batch iterator for a dataset.
    :param data: The data
    :param batch_size: The size of the data batch
    :param num_epochs: The number of epoches
    :param shuffle: Shuffle or not
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
