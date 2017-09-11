# -*- coding: utf-8 -*-
import os
import time
import sys
import datetime
import itertools
import numpy as np
import tensorflow as tf
from collections import Counter
from gensim.models import Word2Vec
import json
import random
import math

import gflags

from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score,label_ranking_average_precision_score,precision_score

class LSTMnet(object):
    def __init__(self, hidden_neurons,keep_prob):
        self.hidden_neurons = hidden_neurons
        self.keep_prob = keep_prob # dropout keep prob
        # create rnn cell
        hidden_layers_lefts = []
        for idx, hidden_size in enumerate(self.hidden_neurons):
            lstm_layer_left = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
            hidden_layer_left = tf.contrib.rnn.DropoutWrapper(cell=lstm_layer_left,
                                                         output_keep_prob=self.keep_prob)
            hidden_layers_lefts.append(hidden_layer_left)
        self.cell = tf.contrib.rnn.MultiRNNCell(cells=hidden_layers_lefts, state_is_tuple=True)

    def dynamic_com(self,input_data,sequence_len):
        state_series_left,current_state_left = tf.nn.dynamic_rnn(cell=self.cell,
                                                                       inputs=input_data,
                                                                       sequence_length=sequence_len,
                                                                       dtype=tf.float32)
        return state_series_left,current_state_left


class TensorFlowLSTM(object):
    def __init__(self, config):
        self.hidden_neurons = hidden_neurons = config["hidden_neurons"]
        self.input_size = input_size = config["input_size"]
        self.keep_prob_value = config["keep_prob"]
        self.hold_value = config["hold_value"]
        self.batch_size = config["batch_size"]
        self.point_sum = config["point_sum"]
        self.l2_reg = config['l2_reg']
        self.data_qid_input = tf.placeholder(tf.float32, [None, None, self.input_size])
        self.data_qid_lengths = tf.placeholder(tf.int32, [None])
        self.label = tf.placeholder(tf.float32, [None,self.point_sum])
        self.keep_prob = tf.placeholder(tf.float32)  # dropout keep prob

        # self.label = tf.placeholder(tf.float32, [None])
        # create rnn cell
        with tf.variable_scope('name') as scope:
            self.cell = LSTMnet(hidden_neurons, self.keep_prob)
            state_series_qid, self.current_state_qid = self.cell.dynamic_com(self.data_qid_input,
                                                                               self.data_qid_lengths)
        # with tf.variable_scope("dynamic_rnn"):
        #     tf.get_variable_scope().reuse_variables()
        # hidden_layers_lefts = []
        # for idx, hidden_size in enumerate(hidden_neurons):
        #     lstm_layer_left = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
        #     hidden_layer_left = tf.contrib.rnn.DropoutWrapper(cell=lstm_layer_left,
        #                                                      output_keep_prob=self.keep_prob)
        #     hidden_layers_lefts.append(hidden_layer_left)
        # self.cell_left = tf.contrib.rnn.MultiRNNCell(cells=hidden_layers_lefts, state_is_tuple=True)
        #
        # hidden_layers_rights = []
        # for idx, hidden_size in enumerate(hidden_neurons):
        #     lstm_layer_right = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
        #     hidden_layer_right = tf.contrib.rnn.DropoutWrapper(cell=lstm_layer_right,
        #                                                           output_keep_prob=self.keep_prob)
        #     hidden_layers_rights.append(hidden_layer_right)
        # self.cell_right = tf.contrib.rnn.MultiRNNCell(cells=hidden_layers_rights, state_is_tuple=True)
        # # dynamic rnn
        #
        #
        # state_series_left, self.current_state_left = tf.nn.dynamic_rnn(cell=self.cell_left,
        #                                                          inputs=self.input_data_left,
        #                                                          sequence_length=self.sequence_len_left,
        #                                                          dtype=tf.float32,scope='left')
        # state_series_right, self.current_state_right = tf.nn.dynamic_rnn(cell=self.cell_right,
        #                                                          inputs=self.input_data_right,
        #                                                          sequence_length=self.sequence_len_right,
        #                                                          dtype=tf.float32,scope='right')

            # output layer
            #print(self.current_state_left)
            #print(self.current_state_right)
        def predict_layer(w1,b1,vec):
            res = tf.matmul(vec, w1) + b1
            # logits = tf.sigmoid(logits)

            return res

        def predict_layer_prob(w1,b1,vec):
            logits = tf.matmul(vec, w1) + b1
            res = tf.sigmoid(logits)
            return res

        pairvec_pos = self.current_state_qid[-1][1]

            #print(pairvec)
        output_w = tf.Variable(tf.truncated_normal([hidden_neurons[-1], self.point_sum], stddev=0.1), "out_w")
        output_b = tf.Variable(tf.constant(0.1, shape=[self.point_sum]), "out_b")

        self.pred = predict_layer_prob(output_w,output_b,pairvec_pos)


        self.loss = tf.reduce_mean(-tf.reduce_sum(self.label*tf.log(self.pred)+ (1-self.label)*tf.log((1-self.pred)),axis= 1))
        # self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label,
        #                                                                       logits=self.pred_all))
        rl2 = self.l2_reg * sum(
            tf.nn.l2_loss(tf_var)
            for tf_var in tf.trainable_variables()
            # if not ("noreg" in tf_var.name or "Bias" in tf_var.name)
        )
        self.loss += rl2
        self.lr = tf.Variable(0.0, trainable=False)
        trainable_vars = tf.trainable_variables()
        self.grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, trainable_vars), 5)

        optimizer = tf.train.AdagradOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(list(zip(self.grads, trainable_vars)))

        print("=" * 50)
        print("List of Variables:")
        for v in tf.trainable_variables():
            print(v.name)
        print("=" * 50)

    # def set_batchsize(self,batchsize):
    #     self.batch_size = batchsize
    # step on batch
    def step(self, sess, data_qid_input,data_qid_lengths,label,is_train):

        input_feed = {self.data_qid_input: data_qid_input,
                          self.data_qid_lengths: data_qid_lengths,
                          self.label: label
                          }

        if is_train:
            input_feed[self.keep_prob] = self.keep_prob_value
            train_loss, _, _ = sess.run([self.loss, self.train_op, self.pred], input_feed)
            return train_loss
        else:
            input_feed[self.keep_prob] = 1
            pred_all = sess.run(self.pred, input_feed)
            return pred_all

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))


def pad_sentences(sequences,wvmodel,num_hidden_wv = 100, dtype='float32',max_text_len=200):
    lengths = [min(len(s), max_text_len) for s in sequences]
    nb_samples = len(sequences)
    maxlen = np.max(lengths)
    x = (np.zeros( (nb_samples,maxlen,num_hidden_wv) )).astype(dtype)
    for idx, s in enumerate(sequences):
        for idy in range(lengths[idx]):
            wv = s[idy]
            if wv in wvmodel:
                x[idx, idy, :] = wvmodel[wv]
            else:
                x[idx, idy, :] = np.random.uniform(-5.0, 5.0, num_hidden_wv)

    return x,lengths


def gen_batch(all_data, batch_size):
    batchlist =[]
    data_size = len(all_data)
    num_batches_per_epoch = int(math.ceil(data_size / batch_size))

    shuffled_data = all_data
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        batchlist.append(shuffled_data[start_index:end_index])

    return  batchlist

def banch_convert(batch,wvmodel,qid_fenci,point_sum = 367,dtype = 'float32'):
    batch_seq = batch
    data_qid = [qid_fenci[x[0]].strip().split(' ') for x in batch_seq]
    data_label = [x[1] for x in batch_seq]
    batch_size = len(batch_seq)
    label = (np.zeros( (batch_size,point_sum) )).astype(dtype)
    for i in range(batch_size):
        for j in data_label[i]:
            label[i,j] = 1.0
    data_qid_input,data_qid_lengths = pad_sentences(sequences = data_qid,wvmodel = wvmodel)

    return  batch_size, label,data_qid_input,data_qid_lengths

def load_train_data(trainpath):
    data = []
    # windows
    fin = open(trainpath,'r')
    # linux
    # fin = open(trainpath, encoding='utf-8')
    for eachLine in fin:
        pid = eachLine.replace('\n', '').split('\t')
        points = pid[1].split(' ')
        list1 = [int(x) for x in points]
        data.append((pid[0],list1))

    fin.close()
    return data


def load_test_data(testpath):
    data = []
    # windows
    with open(testpath,'r') as fin:
        # linux
        # fin = open(trainpath, encoding='utf-8')
        for eachLine in fin:
            pid = eachLine.replace('\n', '').split('\t')
            points = pid[1].split(' ')
            list1 = [int(x) for x in points]
            data.append((pid[0],list1))

    return data


def load_qidfenci(wordpath):
    qidfenci = {}
    fin = open(wordpath, 'r')
    for eachLine in fin:
        js = json.loads(eachLine)
        qid = js['qid']
        fenci = js['fenci']
        qidfenci[qid] = fenci
    fin.close()
    return qidfenci




def run(filepath,sess):
    batch_size = 128
    point_sum = 367
    # epoch = 20
    # config and create model
    outfile = open(filepath+'result-KPP-max', 'w')
    # predfile = open(filepath + 'pred_result-pairsise-pic', 'w')
    config = {"hidden_neurons": [100],
              "keep_prob": 0.8,
              "hold_value": 1.0,
              "batch_size": batch_size,
              'l2_reg':0.00004,
              'point_sum':point_sum,
              "input_size": 100}
    model = TensorFlowLSTM(config)
    # sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    lr = 0.2
    lr_decay = 0.9
    # run epoch
    wvmodel = Word2Vec.load(filepath + 'word2vec/w2vmodel')
    traindata = load_train_data(filepath + 'Train.txt')
    testdata = load_test_data(filepath + 'Test.txt')
    qid_fenci = load_qidfenci(filepath + 'qid_fenci.json')
    for epoch in range(100):
        # train
        model.assign_lr(sess, lr * lr_decay ** epoch)
        overall_loss = 0
        st = time.time()
        batches = gen_batch(traindata, batch_size)
        print(len(batches))
        for eachbatch in batches:
            batchnum, data_label,data_qid_input,data_qid_lengths = banch_convert(
                batch=eachbatch, wvmodel=wvmodel, qid_fenci=qid_fenci)
            # model.set_batchsize(batchnum)
            loss = model.step(sess, data_qid_input=data_qid_input,data_qid_lengths=data_qid_lengths,
                              label=data_label,
                              is_train=True)
            overall_loss += loss
            print(("\r loss:{0}, time spent:{1}s".format(loss, time.time() - st)))
            sys.stdout.flush()

        print(("\r overall_loss:{0}, time spent:{1}s".format(overall_loss, time.time() - st)))
        sys.stdout.flush()

        # test
        testbatches = gen_batch(testdata, batch_size)
        pred_label =[]
        for eachbatch in testbatches:
            batchnum, data_label,data_qid_input,data_qid_lengths= banch_convert(
                batch=eachbatch, wvmodel=wvmodel, qid_fenci=qid_fenci)
            # model.set_batchsize(batchnum)
            pred_all = model.step(sess, data_qid_input=data_qid_input,data_qid_lengths=data_qid_lengths,
                                  label=data_label,
                              is_train=False)
            batch1 = eachbatch
            for idx in range(batchnum):
                set1 =set()
                for v in batch1[idx][1]:
                    set1.add(v)
                set2 = set()
                list1 = []
                for idy in range(point_sum):
                    list1.append((pred_all[idx,idy],idy))
                listmax =  sorted(list1, key=lambda asd: asd[0], reverse=True)
                maxlen = 5
                for k in range(maxlen):
                    set2.add(listmax[k][1])
                coms = set1&set2
                unions = set1|set2
                acc = float(len(coms))/float(len(unions))
                re = float(len(coms))/float(len(set1))
                pr = 0 if len(set2) ==0 else float(len(coms))/float(len(set2))
                pred_label.append((batch1[idx][0],acc,pr,re))

        d_size = len(pred_label)
        accsum = 0.0
        prsum = 0.0
        resum = 0.0
        for each in pred_label:
            accsum += each[1]
            prsum += each[2]
            resum += each[3]
        accuracy = accsum/d_size
        precision = prsum/d_size
        recall=resum/d_size
        print(("\n epoch = {0}, accuracy={1}, precision={2}, recall={3}".format(epoch, accuracy,precision,recall)))
        outfile.write(("\n epoch = {0}, accuracy, precision, recall\n{1}\t{2}\t{3}".format(epoch, accuracy,precision,recall)))

    outfile.close()
    # predfile.close()


def main(flags):
    #
    # TODO: Any code here.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    inputstr = "/home/huangzai/LSTMsim/Point_Prediction/"

    with tf.Session(config=config) as session:
        # session.run(tf.global_variables_initializer())
        #
        # TODO: Any code here.
        run(inputstr,sess=session)
    return 0


if __name__ == '__main__':
    global_flags = gflags.FLAGS
    gflags.DEFINE_boolean('help', False, 'Show this help.')
    gflags.DEFINE_string('gpu', '3', 'Which GPU to use.')
    #
    # TODO: Other FLAGS here.
    global_flags(sys.argv)
    if global_flags.help:
        print(global_flags.main_module_help())
        exit(0)
    os.environ['CUDA_VISIBLE_DEVICES'] = global_flags.gpu
    exit(main(global_flags))