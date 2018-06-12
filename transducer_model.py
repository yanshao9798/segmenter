# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.contrib.legacy_seq2seq as seq2seq
import toolbox
import batch as Batch
import numpy as np
import cPickle as pickle
import evaluation

import os

class Seq2seq(object):

    def __init__(self, trained_model):
        self.en_vec = None
        self.de_vec = None
        self.trans_output = None
        self.trans_labels = None
        self.feed_previouse = None
        self.trans_l_rate = None
        self.trained = trained_model
        self.decode_step = None
        self.encode_step = None

    def define(self, char_num, rnn_dim, emb_dim, max_x, max_y, write_trans_model=True):
        self.decode_step = max_y
        self.encode_step = max_x
        self.en_vec = [tf.placeholder(tf.int32, [None], name='en_input' + str(i)) for i in range(max_x)]
        self.trans_labels = [tf.placeholder(tf.int32, [None], name='de_input' + str(i)) for i in range(max_y)]
        weights = [tf.cast(tf.sign(ot_t), tf.float32) for ot_t in self.trans_labels]
        self.de_vec = [tf.zeros_like(self.trans_labels[0], tf.int32)] + self.trans_labels[:-1]
        self.feed_previous = tf.placeholder(tf.bool)
        self.trans_l_rate = tf.placeholder(tf.float32, [], name='learning_rate')
        seq_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_dim, state_is_tuple=True)
        self.trans_output, states = seq2seq.embedding_attention_seq2seq(self.en_vec, self.de_vec, seq_cell, char_num,
                                                                        char_num, emb_dim, feed_previous=self.feed_previous)

        loss = seq2seq.sequence_loss(self.trans_output, self.trans_labels, weights)
        optimizer = tf.train.AdagradOptimizer(learning_rate=self.trans_l_rate)

        params = tf.trainable_variables()
        gradients = tf.gradients(loss, params)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients, 5.0)
        self.trans_train = optimizer.apply_gradients(zip(clipped_gradients, params))

        self.saver = tf.train.Saver()

        if write_trans_model:
            param_dic = {}
            param_dic['char_num'] = char_num
            param_dic['rnn_dim'] = rnn_dim
            param_dic['emb_dim'] = emb_dim
            param_dic['max_x'] = max_x
            param_dic['max_y'] = max_y
            # print param_dic
            f_model = open(self.trained + '_model', 'w')
            pickle.dump(param_dic, f_model)
            f_model.close()

    def train(self, t_x, t_y, v_x, v_y, lrv, char2idx, sess, epochs, batch_size=10, reset=True):

        idx2char = {k: v for v, k in char2idx.items()}
        v_y_g = [np.trim_zeros(v_y_t) for v_y_t in v_y]
        gold_out = [toolbox.generate_trans_out(v_y_t, idx2char) for v_y_t in v_y_g]

        best_score = 0

        if reset or not os.path.isfile(self.trained + '_weights.index'):
            for epoch in range(epochs):
                Batch.train_seq2seq(sess, model=self.en_vec + self.trans_labels, decoding=self.feed_previous,
                                    batch_size=batch_size, config=self.trans_train, lr=self.trans_l_rate, lrv=lrv,
                                    data=[t_x] + [t_y])
                pred = Batch.predict_seq2seq(sess, model=self.en_vec + self.de_vec + self.trans_output,
                                             decoding=self.feed_previous, decode_len=self.decode_step,
                                             data=[v_x], argmax=True, batch_size=100)
                pred_out = [toolbox.generate_trans_out(pre_t, idx2char) for pre_t in pred]

                c_scores = evaluation.trans_evaluator(gold_out, pred_out)

                print 'epoch: %d' % (epoch + 1)

                print 'ACC: %f' % c_scores[0]
                print 'Token F score: %f' % c_scores[1]

                if c_scores[1] > best_score:
                    best_score = c_scores[1]
                    self.saver.save(sess, self.trained + '_weights', write_meta_graph=False)

        if best_score > 0 or not reset:
            self.saver.restore(sess, self.trained + '_weights')

    def tag(self, t_x, char2idx, sess, batch_size=100):

        t_x = [t_x_t[:self.encode_step] for t_x_t in t_x]
        t_x = toolbox.pad_zeros(t_x, self.encode_step)

        idx2char = {k: v for v, k in char2idx.items()}

        pred = Batch.predict_seq2seq(sess, model=self.en_vec + self.de_vec + self.trans_output, decoding=self.feed_previous,
                                         decode_len=self.decode_step, data=[t_x], argmax=True, batch_size=batch_size)
        pred_out = [toolbox.generate_trans_out(pre_t, idx2char) for pre_t in pred]

        return pred_out


