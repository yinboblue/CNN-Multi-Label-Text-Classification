# -*- coding:utf-8 -*-

import os
import multiprocessing
import numpy as np
import gensim
import logging
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

from pylab import mpl
from gensim import corpora
from gensim.models import word2vec
from gensim.corpora import Dictionary
from tflearn.data_utils import to_categorical, pad_sequences
from sklearn.manifold import TSNE

logging.getLogger().setLevel(logging.INFO)

mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

BASE_DIR = os.getcwd()
TEXT_DIR = BASE_DIR + '/content.txt'
VOCABULARY_DICT_DIR = BASE_DIR + '/math.dict'
WORD2VEC_DIR = BASE_DIR + '/math.model'


def create_vocab(text_file):
    texts = []
    with open(text_file, 'r') as fin:
        for eachline in fin:
            line = eachline.strip().split('\t')
            texts.append(line)
    vocab_dict = corpora.Dictionary(texts)
    print(vocab_dict)
    vocab_dict.save(VOCABULARY_DICT_DIR)
    return vocab_dict


# create_vocab(TEXT_DIR)

def word2vec_train(embedding_size, input_file=TEXT_DIR, output_file=WORD2VEC_DIR):
    sentences = word2vec.LineSentence(input_file)

    # sg=0 -> CBOW model; sg=1 -> skip-gram model.
    # 生成 embedding_size 的词向量 model
    model = gensim.models.Word2Vec(sentences, size=embedding_size, min_count=0,
                                   sg=0, workers=multiprocessing.cpu_count())
    model.save(output_file)


# word2vec_train(300)

def data_word2vec(input_file, dictionary):
    def token_to_index(content, dictionary):
        result = []
        for item in content:
            if item != '<end>' and (len(item) > 0):
                result.append(dictionary.token2id[item])
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

            front_content_indexlist.append(token_to_index(front_content, dictionary))
            behind_content_indexlist.append(token_to_index(behind_content[1:], dictionary))
        total_line = index + 1

    class Data:
        def __init__(self, total_line, labels, front_content_indexlist, behind_content_indexlist):
            self.number = total_line
            self.labels = labels
            self.front_tokenindex = front_content_indexlist
            self.behind_tokenindex = behind_content_indexlist

    return Data(total_line, labels, front_content_indexlist, behind_content_indexlist)


def load_word2vec_matrix(vocab_size, embedding_size):
    model = gensim.models.Word2Vec.load(WORD2VEC_DIR)
    vocab_dict = Dictionary.load(VOCABULARY_DICT_DIR)

    vector = np.zeros([vocab_size, embedding_size])
    for value, key in vocab_dict.items():
        if len(key) > 0:
            vector[value] = model[key]

    return vector


def max_seq_len_cal(content_indexlist):
    result = 0
    for item in content_indexlist:
        if len(item) > result:
            result = len(item)
    return result


def load_data_and_labels(data_file, MAX_SEQUENCE_LENGTH, EMBEDDING_SIZE):
    """
    Loads research data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load vocabulary dict file
    vocab_dict = create_vocab(TEXT_DIR)

    # Load word2vec file
    word2vec_train(EMBEDDING_SIZE, TEXT_DIR, WORD2VEC_DIR)

    # Load data from files and split by words
    data = data_word2vec(input_file=data_file, dictionary=vocab_dict)
    max_seq_len = max(max_seq_len_cal(data.front_tokenindex), max_seq_len_cal(data.behind_tokenindex))
    logging.info('Found %s texts.' % data.number)
    logging.info('Max sequence length is:', max_seq_len)
    data_front = pad_sequences(data.front_tokenindex, maxlen=MAX_SEQUENCE_LENGTH, value=0.)
    data_behind = pad_sequences(data.behind_tokenindex, maxlen=MAX_SEQUENCE_LENGTH, value=0.)
    labels = to_categorical(data.labels, nb_classes=2)
    logging.info('Shape of data front tensor:', data_front.shape)
    logging.info('Shape of data behind tensor:', data_behind.shape)
    logging.info('Shape of label tensor:', labels.shape)
    return data_front, data_behind, labels


def load_vocab_size(vocab_data_file=VOCABULARY_DICT_DIR):
    vocab_dict = Dictionary.load(vocab_data_file)
    return len(vocab_dict.items())


# pretrained_word2vec_matrix = load_word2vec_matrix(load_vocab_size(), embedding_size=128)
# print(pretrained_word2vec_matrix)
# print(type(pretrained_word2vec_matrix))
# print(pretrained_word2vec_matrix.shape)

def plot_word2vec():
    model = gensim.models.Word2Vec.load(WORD2VEC_DIR)
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
