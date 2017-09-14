# -*- coding:utf-8 -*-

import os
import multiprocessing
import numpy as np
import gensim
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

from pylab import mpl
from gensim.models import word2vec
from tflearn.data_utils import to_categorical, pad_sequences
from sklearn.manifold import TSNE

logging.getLogger().setLevel(logging.INFO)

mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

BASE_DIR = os.getcwd()
TEXT_DIR = BASE_DIR + '/content.txt'


def create_word2vec_model(embedding_size, input_file=TEXT_DIR):
    word2vec_file = BASE_DIR + '/word2vec_' + str(embedding_size) + '.model'

    if os.path.isfile(word2vec_file):
        logging.info('☛ The word2vec model you want create already exists!')
    else:
        sentences = word2vec.LineSentence(input_file)
        # sg=0 means use CBOW model(default); sg=1 means use skip-gram model.
        model = gensim.models.Word2Vec(sentences, size=embedding_size, min_count=0,
                                       sg=0, workers=multiprocessing.cpu_count())
        model.save(word2vec_file)


def load_vocab_size(embedding_size):
    word2vec_file = BASE_DIR + '/word2vec_' + str(embedding_size) + '.model'

    if os.path.isfile(word2vec_file):
        model = word2vec.Word2Vec.load(word2vec_file)
        return len(model.wv.vocab.items())
    else:
        logging.info("✘ The word2vec file doesn't exist. "
                     "Please use function <create_vocab_size(embedding_size)> to create it!")


def data_word2vec(input_file, word2vec_model):
    vocab = dict([(k, v.index) for (k, v) in word2vec_model.wv.vocab.items()])

    def token_to_index(content):
        result = []
        for item in content:
            if item != '<end>' and (len(item) > 0):
                id = vocab.get(item)
                if id is None:
                    id = 0
                result.append(id)
        return result

    with open(input_file) as fin:
        labels = []
        front_content_indexlist = []
        behind_content_indexlist = []
        for index, eachline in enumerate(fin):
            front_content = []
            behind_content = []
            line = eachline.strip().split('\t')
            label = line[2]
            content = line[3].strip().split(' ')

            end_tag = False
            for item in content:
                if item == '<end>':
                    end_tag = True
                if not end_tag:
                    front_content.append(item)
                if end_tag:
                    behind_content.append(item)

            labels.append(label)

            front_content_indexlist.append(token_to_index(front_content))
            behind_content_indexlist.append(token_to_index(behind_content[1:]))
        total_line = index + 1

    class Data:
        def __init__(self):
            pass

        @property
        def number(self):
            return total_line

        @property
        def labels(self):
            return labels

        @property
        def front_tokenindex(self):
            return front_content_indexlist

        @property
        def behind_tokenindex(self):
            return behind_content_indexlist

    return Data()


def load_word2vec_matrix(vocab_size, embedding_size):
    word2vec_file = BASE_DIR + '/word2vec_' + str(embedding_size) + '.model'

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


def load_data_and_labels(data_file, embedding_size):
    """
    Loads research data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    word2vec_file = BASE_DIR + '/word2vec_' + str(embedding_size) + '.model'

    # Load word2vec model file
    if os.path.isfile(word2vec_file):
        model = word2vec.Word2Vec.load(word2vec_file)
    else:
        create_word2vec_model(embedding_size, TEXT_DIR)

    # Load data from files and split by words
    data = data_word2vec(input_file=data_file, word2vec_model=model)
    max_front_len = max([len(x) for x in data.front_tokenindex])
    max_behind_len = max([len(x) for x in data.behind_tokenindex])
    max_seq_len = max(max_front_len, max_behind_len)
    logging.info('Found {} texts.'.format(data.number))
    return data, max_seq_len


def pad_data(data, max_seq_len):
    data_front = pad_sequences(data.front_tokenindex, maxlen=max_seq_len, value=0.)
    data_behind = pad_sequences(data.behind_tokenindex, maxlen=max_seq_len, value=0.)
    labels = to_categorical(data.labels, nb_classes=2)
    return data_front, data_behind, labels


def plot_word2vec(word2vec_file):
    model = gensim.models.Word2Vec.load(word2vec_file)
    data_x = []
    data_y = []
    for index, item in enumerate(model.wv.vocab):
        data_x.append(model[item])
        data_y.append(item)
    tsne = TSNE(n_components=2)
    x_tsne = tsne.fit_transform(data_x)

    def scatter(x, y):
        f = plt.figure(figsize=(50, 50))
        ax = plt.subplot(aspect='equal')
        sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40)
        plt.xlim(-25, 25)
        plt.ylim(-25, 25)
        ax.axis('off')
        ax.axis('tight')
        txts = []
        for i in range(len(y)):
            # Position of each label.
            if i % 20 == 0:
                txt = ax.text(x[i, 0], x[i, 1], y[i], fontsize=10)
                txt.set_path_effects([
                    PathEffects.Stroke(linewidth=5, foreground="w"),
                    PathEffects.Normal()])
                txts.append(txt)
        return f, ax, sc, txts

    scatter(x_tsne, data_y)
    plt.savefig('word_vector.png', dpi=150)


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
    :return:
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
