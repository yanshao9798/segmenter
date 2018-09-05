# -*- coding: utf-8 -*-
"""
@author: Yan Shao, yan.shao@lingfil.uu.se
"""
import reader
import toolbox
from model import Model
from transducer_model import Seq2seq
import sys
import argparse
import os
import codecs
import tensorflow as tf
import cPickle as pickle

from time import time

parser = argparse.ArgumentParser(description='A Universal Tokeniser. Written by Y. Shao, Uppsala University')
parser.add_argument('action', default='tag', choices=['train', 'test', 'tag'], help='train, test or tag')

parser.add_argument('-f', '--format', default='conll', help='Data format of different tasks, conll, mlp1 or mlp2')

parser.add_argument('-p', '--path', default=None, help='Path of the workstation')

parser.add_argument('-t', '--train', default=None, help='File for training')
parser.add_argument('-d', '--dev', default=None, help='File for validation')
parser.add_argument('-e', '--test', default=None, help='File for evaluation')
parser.add_argument('-r', '--raw', default=None, help='Raw file for tagging')

parser.add_argument('-m', '--model', default='trained_model', help='Name of the trained model')
parser.add_argument('-crf', '--crf', default=1, type=int, help='Using CRF interface')

parser.add_argument('-bt', '--bucket_size', default=50, type=int, help='Bucket size')
parser.add_argument('-sl', '--sent_limit', default=300, type=int, help='Long sentences will be chopped')

parser.add_argument('-tg', '--tags', default='BIES', help='Boundary Tagging, default is BIES')

parser.add_argument('-ed', '--emb_dimension', default=50, type=int, help='Dimension of the embeddings')
parser.add_argument('-emb', '--embeddings', default=None, help='Path and name of pre-trained char embeddings')

parser.add_argument('-ng', '--ngram', default=1, type=int, help='Using ngrams')

parser.add_argument('-cell', '--cell', default='gru', help='Use GRU as the recurrent cell', choices=['gru', 'lstm'])
parser.add_argument('-rnn', '--rnn_cell_dimension', default=200, type=int, help='Dimension of the RNN cells')
parser.add_argument('-layer', '--rnn_layer_number', default=1, type=int, help='Numbers of the RNN layers')

parser.add_argument('-dr', '--dropout_rate', default=0.5, type=float, help='Dropout rate')

parser.add_argument('-iter', '--epochs', default=30, type=int, help='Numbers of epochs')
parser.add_argument('-iter_trans', '--epochs_trans', default=50, type=int, help='Epochs for training the transducer')

parser.add_argument('-op', '--optimizer', default='adagrad', help='Optimizer')
parser.add_argument('-lr', '--learning_rate', default=0.2, type=float, help='Initial learning rate')
parser.add_argument('-lr_trans', '--learning_rate_trans', default=0.3, type=float, help='Initial learning rate')
parser.add_argument('-ld', '--decay_rate', default=0.05, type=float, help='Learning rate decay')
parser.add_argument('-mt', '--momentum', default=None, type=float, help='Momentum')

parser.add_argument('-ncp', '--no_clipping', default=False, action='store_true', help='Do not apply gradient clipping')

parser.add_argument("-tb","--train_batch", help="Training batch size", default=10, type=int)
parser.add_argument("-eb","--test_batch", help="Testing batch size", default=500, type=int)
parser.add_argument("-rb","--tag_batch", help="Tagging batch size", default=500, type=int)

parser.add_argument("-g","--gpu", help="the id of gpu, the default is 0", default=0, type=int)

parser.add_argument('-opth', '--output_path', default=None, help='Output path')

parser.add_argument('-sea', '--sea', help='Process languages like Vietamese', default=False, action='store_true')

parser.add_argument('-ss', '--sent_seg', help='Perform sentence seg', default=False, action='store_true')

parser.add_argument('-ens', '--ensemble', default=False, help='Ensemble several weights', action='store_true')

parser.add_argument('-sgl', '--segment_large', default=False, help='Segment (very) large file', action='store_true')

parser.add_argument('-lgs', '--large_size', default=10000, type=int, help='Segment (very) large file')

parser.add_argument('-ot', '--only_tokenised', default=False,
                    help='Only output the tokenised file when segment (very) large file', action='store_true')

parser.add_argument('-ts', '--train_size', default=-1, type=int, help='No. of sentences used for training')

parser.add_argument('-rs', '--reset', default=False, help='Delete and re-initialise the intermediate files',
                    action='store_true')
parser.add_argument('-rst', '--reset_trans', default=False, help='Retrain the transducers', action='store_true')

parser.add_argument('-isp', '--ignore_space', default=False, help='Ignore space delimiters', action='store_true')
parser.add_argument('-imt', '--ignore_mwt', default=False, help='Ignore multi-word tokens to be transcribed',
                    action='store_true')

parser.add_argument('-sb', '--segmentation_bias', default=-1, type=float,
                    help='Add segmentation bias to under(over)-splitting')

parser.add_argument('-tt', '--transduction_type', default='mix', choices=['mix', 'dict', 'trans', 'none'],
                    help='Different ways of transducing the non-segmental MWTs')

args = parser.parse_args()

sys = reload(sys)
sys.setdefaultencoding('utf-8')
print 'Encoding: ', sys.getdefaultencoding()

if args.action == 'train':
    assert args.path is not None
    path = args.path
    train_file = args.train
    dev_file = args.dev
    model_file = args.model
    print 'Reading data......'
    f_names = os.listdir(path)
    if train_file is None or dev_file is None:
        for f_n in f_names:
            if 'ud-train.conllu' in f_n or 'training.segd' in f_n or 'ud-sample.conllu' in f_n:
                train_file = f_n
            elif 'ud-dev.conllu' in f_n or 'development.segd' in f_n:
                dev_file = f_n
    assert train_file is not None
    is_space = True
    if 'Chinese' in path or 'Japanese' in path or args.format == 'mlp2':
        is_space = False

    if args.sea:
        is_space = 'sea'
    if args.reset or not os.path.isfile(path + '/raw_train.txt') or not os.path.isfile(path + '/raw_dev.txt'):
        cat = 'other'
        if 'Chinese' in path or 'Japanese' in path:
            cat = 'zh'
        for line in codecs.open(path + '/' + train_file, 'r', encoding='utf-8'):
            if len(line) < 2:
                break
            if '# sentence' in line or '# text' in line:
                cat = 'gold'

        if dev_file is None:
            reader.get_raw(path, train_file, '/raw_train.txt', cat, is_dev=False, form=args.format, is_space=is_space)
        else:
            reader.get_raw(path, train_file, '/raw_train.txt', cat, form=args.format, is_space=is_space)
            reader.get_raw(path, dev_file, '/raw_dev.txt', cat, form=args.format, is_space=is_space)

    if args.reset or not os.path.isfile(path + '/tag_train.txt') or not os.path.isfile(path + '/tag_dev.txt') or \
            not os.path.isfile(path + '/tag_dev_gold.txt'):
        if dev_file is None:
            raws_train = reader.raw(path + '/raw_train.txt')
            raws_dev = reader.raw(path + '/raw_dev.txt')
            sents_train, sents_dev = reader.gold(path + '/' + train_file, False, form=args.format, is_space=is_space)
        else:
            raws_train = reader.raw(path + '/raw_train.txt')
            sents_train = reader.gold(path + '/' + train_file, form=args.format, is_space=is_space)

            raws_dev = reader.raw(path + '/raw_dev.txt')
            sents_dev = reader.gold(path + '/' + dev_file, form=args.format, is_space=is_space)

        if is_space != 'sea':
            toolbox.raw2tags(raws_train, sents_train, path, 'tag_train.txt', ignore_space=args.ignore_space,
                             reset=args.reset, tag_scheme=args.tags, ignore_mwt=args.ignore_mwt)
            toolbox.raw2tags(raws_dev, sents_dev, path, 'tag_dev.txt', creat_dict=False, gold_path='tag_dev_gold.txt',
                             ignore_space=args.ignore_space, tag_scheme=args.tags, ignore_mwt=args.ignore_mwt)
        else:
            toolbox.raw2tags_sea(raws_train, sents_train, path, 'tag_train.txt', reset=args.reset, tag_scheme=args.tags)
            toolbox.raw2tags_sea(raws_dev, sents_dev, path, 'tag_dev.txt', gold_path='tag_dev_gold.txt',
                                 tag_scheme=args.tags)

    if args.reset or not os.path.isfile(path + '/chars.txt'):
        toolbox.get_chars(path, ['raw_train.txt', 'raw_dev.txt'], sea=is_space)

    char2idx, unk_chars_idx, idx2char, tag2idx, idx2tag, trans_dict = toolbox.get_dicts(path, args.sent_seg, args.tags,
                                                                                        args.crf)

    if args.embeddings is not None:
        print 'Reading embeddings...'
        short_emb = args.embeddings[args.embeddings.index('/') + 1: args.embeddings.index('.')]
        if args.reset or not os.path.isfile(path + '/' + short_emb + '_sub.txt'):
            toolbox.get_sample_embedding(path, args.embeddings, char2idx)
        emb_dim, emb, valid_chars = toolbox.read_sample_embedding(path, short_emb, char2idx)
        for vch in valid_chars:
            if char2idx[vch] in unk_chars_idx:
                unk_chars_idx.remove(char2idx[vch])
    else:
        emb_dim = args.emb_dimension
        emb = None

    train_x, train_y, max_len_train = toolbox.get_input_vec(path, 'tag_train.txt', char2idx, tag2idx,
                                                            limit=args.sent_limit, sent_seg=args.sent_seg,
                                                            is_space=is_space, train_size=args.train_size,
                                                            ignore_space=args.ignore_space)

    dev_x, max_len_dev = toolbox.get_input_vec_raw(path, 'raw_dev.txt', char2idx, limit=args.sent_limit,
                                                   sent_seg=args.sent_seg, is_space=is_space,
                                                   ignore_space=args.ignore_space)
    if args.sent_seg:
        print 'Joint sentence segmentation...'
    else:
        print 'Training set: %d instances; Dev set: %d instances.' % (len(train_x[0]), len(dev_x[0]))

    nums_grams = None
    ng_embs = None

    if args.ngram > 1 and (args.reset or  not os.path.isfile(path + '/' + str(args.ngram) + 'gram.txt')):
        toolbox.get_ngrams(path, args.ngram, is_space)

    ngram = toolbox.read_ngrams(path, args.ngram)

    if args.ngram > 1:
        gram2idx = toolbox.get_ngram_dic(ngram)
        train_gram = toolbox.get_gram_vec(path, 'tag_train.txt', gram2idx, limit=args.sent_limit,sent_seg=args.sent_seg,
                                          is_space=is_space, ignore_space=args.ignore_space)
        dev_gram = toolbox.get_gram_vec(path, 'raw_dev.txt', gram2idx, is_raw=True, limit=args.sent_limit,
                                        sent_seg=args.sent_seg, is_space=is_space, ignore_space=args.ignore_space)
        train_x += train_gram
        dev_x += dev_gram
        nums_grams = []
        for dic in gram2idx:
            nums_grams.append(len(dic.keys()))

    max_len = max(max_len_train, max_len_dev)

    b_train_x, b_train_y = toolbox.buckets(train_x, train_y, size=args.bucket_size)
    b_train_x, b_train_y, b_lens, b_count = toolbox.pad_bucket(b_train_x, b_train_y, max_len)

    b_dev_x = [toolbox.pad_zeros(dev_x_i, max_len) for dev_x_i in dev_x]

    b_dev_y_gold = [line.strip() for line in codecs.open(path + '/tag_dev_gold.txt', 'r', encoding='utf-8')]

    nums_tag = len(tag2idx)

    config = tf.ConfigProto(allow_soft_placement=True)
    gpu_config = "/gpu:" + str(args.gpu)

    transducer = None
    transducer_graph = None
    trans_model = None
    trans_init = None

    if len(trans_dict) > 200 and not args.ignore_mwt:
        transducer = toolbox.get_dict_vec(trans_dict, char2idx)
    t = time()

    initializer = tf.contrib.layers.xavier_initializer()

    if transducer is not None:
        transducer_graph = tf.Graph()
        with transducer_graph.as_default():
            with tf.variable_scope("transducer") as scope:
                trans_model = Seq2seq(path + '/' + model_file + '_transducer')
                print 'Defining transducer...'
                trans_model.define(char_num=len(char2idx), rnn_dim=args.rnn_cell_dimension, emb_dim=args.emb_dimension,
                                   max_x=len(transducer[0][0]), max_y=len(transducer[1][0]))
            trans_init = tf.global_variables_initializer()
        transducer_graph.finalize()

    print 'Initialization....'
    main_graph = tf.Graph()
    with main_graph.as_default():
        with tf.variable_scope("tagger") as scope:
            model = Model(nums_chars=len(char2idx) + 2, nums_tags=nums_tag, buckets_char=b_lens, counts=b_count,
                          crf=args.crf, ngram=nums_grams, batch_size=args.train_batch, sent_seg=args.sent_seg,
                          is_space=is_space, emb_path=args.embeddings, tag_scheme=args.tags)

            model.main_graph(trained_model=path + '/' + model_file + '_model', scope=scope,
                             emb_dim=emb_dim, cell=args.cell, rnn_dim=args.rnn_cell_dimension,
                             rnn_num=args.rnn_layer_number, drop_out=args.dropout_rate, emb=emb)
            t = time()

        model.config(optimizer=args.optimizer, decay=args.decay_rate, lr_v=args.learning_rate,
                     momentum=args.momentum, clipping=not args.no_clipping)
        init = tf.global_variables_initializer()

        print 'Done. Time consumed: %d seconds' % int(time() - t)

    main_graph.finalize()

    main_sess = tf.Session(config=config, graph=main_graph)

    if args.crf > 0:
        decode_graph = tf.Graph()
        with decode_graph.as_default():
            model.decode_graph()
        decode_graph.finalize()

        decode_sess = tf.Session(config=config, graph=decode_graph)

        sess = [main_sess, decode_sess]

    else:
        sess = [main_sess, None]

    with tf.device(gpu_config):

        if transducer is not None:
            print 'Building transducer...'
            t = time()
            trans_sess = tf.Session(config=config, graph=transducer_graph)
            trans_sess.run(trans_init)
            trans_model.train(transducer[0], transducer[1], transducer[2], transducer[3], args.learning_rate_trans,
                              char2idx, trans_sess, args.epochs_trans, batch_size=10, reset=args.reset_trans)
            sess.append(trans_sess)
            print 'Done. Time consumed: %d seconds' % int(time() - t)
            print 'Training the main segmenter..'
        main_sess.run(init)
        print 'Initialisation...'
        print 'Done. Time consumed: %d seconds' % int(time() - t)
        t = time()
        b_dev_raw = [line.strip() for line in codecs.open(path + '/raw_dev.txt', 'r', encoding='utf-8')]
        model.train(b_train_x, b_train_y, b_dev_x, b_dev_raw, b_dev_y_gold, idx2tag, idx2char, unk_chars_idx, trans_dict,
                    sess, args.epochs, path + '/' + model_file + '_weights', transducer=trans_model,
                    lr=args.learning_rate, decay=args.decay_rate, sent_seg=args.sent_seg, outpath=args.output_path)

else:

    assert args.path is not None
    assert args.model is not None
    path = args.path
    assert os.path.isfile(path + '/chars.txt')

    model_file = args.model

    if args.ensemble:
        if not os.path.isfile(path + '/' + model_file + '_1_model') or not os.path.isfile(path + '/' + model_file +
                                                                                          '_1_weights.index'):
            raise Exception('Not any model file or weights file under the name of ' + model_file + '.')
        fin = open(path + '/' + model_file + '_1_model', 'rb')
    else:
        if not os.path.isfile(path + '/' + model_file + '_model') or not os.path.isfile(path + '/' + model_file +
                                                                                        '_weights.index'):
            raise Exception('No model file or weights file under the name of ' + model_file + '.')
        fin = open(path + '/' + model_file + '_model', 'rb')

    weight_path = path + '/' + model_file

    param_dic = pickle.load(fin)
    fin.close()

    nums_chars = param_dic['nums_chars']
    nums_tags = param_dic['nums_tags']
    crf = param_dic['crf']
    emb_dim = param_dic['emb_dim']
    cell = param_dic['cell']
    rnn_dim = param_dic['rnn_dim']
    rnn_num = param_dic['rnn_num']
    drop_out = param_dic['drop_out']
    buckets_char = param_dic['buckets_char']
    nums_ngrams = param_dic['ngram']
    is_space = param_dic['is_space']
    sent_seg = param_dic['sent_seg']
    emb_path = param_dic['emb_path']
    tag_scheme = param_dic['tag_scheme']

    if args.embeddings is not None:
        emb_path = args.embeddings

    ngram = 1
    grams, gram2idx = None, None
    if nums_ngrams is not None:
        ngram = len(nums_ngrams) + 1

    char2idx, unk_chars_idx, idx2char, tag2idx, idx2tag, trans_dict = toolbox.get_dicts(path, sent_seg, tag_scheme, crf)

    trans_char_num = len(char2idx)

    if ngram > 1:
        grams = toolbox.read_ngrams(path, ngram)

    new_chars, new_grams = None, None

    test_x, test_y, raw_x, test_y_gold = None, None, None, None

    sub_dict = None

    max_step = None

    raw_file = None

    if args.action == 'test':
        test_file = args.test
        f_names = os.listdir(path)
        if test_file is None:
            for f_n in f_names:
                if 'ud-test.conllu' in f_n:
                    test_file = f_n
        assert test_file is not None

        cat = 'other'
        if 'Chinese' in path or 'Japanese' in path:
            cat = 'zh'
        for line in codecs.open(path + '/' + test_file, 'r', encoding='utf-8'):
            if len(line) < 2:
                break
            if '# sentence' in line or '# text' in line:
                cat = 'gold'
        reader.get_raw(path, test_file, 'raw_test.txt', cat, form=args.format)

        raws_test = reader.raw(path + '/raw_test.txt')
        test_y_gold = reader.test_gold(path + '/' + test_file, form=args.format, is_space=is_space,
                                       ignore_mwt=args.ignore_mwt)

        new_chars = toolbox.get_new_chars(path + '/raw_test.txt', char2idx, is_space)

        if emb_path is not None:
            valid_chars = toolbox.get_valid_chars(new_chars + char2idx.keys(), emb_path)
        else:
            valid_chars = None

        char2idx, idx2char, unk_chars_idx, sub_dict = toolbox.update_char_dict(char2idx, new_chars, unk_chars_idx, valid_chars)

        test_x, max_len_test = toolbox.get_input_vec_raw(path, 'raw_test.txt', char2idx, limit=args.sent_limit + 100,
                                                         sent_seg=sent_seg, is_space=is_space,
                                                         ignore_space=args.ignore_space)

        max_step = max_len_test

        if sent_seg:
            print 'Joint sentence segmentation...'
        else:
            print 'Test set: %d instances.' % len(test_x[0])

        if ngram > 1:
            gram2idx = toolbox.get_ngram_dic(grams)
            new_grams = toolbox.get_new_grams(path + '/' + test_file, gram2idx, is_space=is_space)

            test_grams = toolbox.get_gram_vec(path, 'raw_test.txt', gram2idx, is_raw=True, limit=args.sent_limit + 100,
                                              sent_seg=sent_seg, is_space=is_space, ignore_space=args.ignore_space)
            test_x += test_grams

        for k in range(len(test_x)):
            test_x[k] = toolbox.pad_zeros(test_x[k], max_step)

    elif args.action == 'tag':
        assert args.raw is not None

        raw_file = args.raw
        new_chars = toolbox.get_new_chars(raw_file, char2idx, is_space)

        if emb_path is not None:
            valid_chars = toolbox.get_valid_chars(new_chars, emb_path)
        else:
            valid_chars = None

        char2idx, idx2char, unk_chars_idx, sub_dict = toolbox.update_char_dict(char2idx, new_chars, unk_chars_idx,
                                                                               valid_chars)

        if not args.segment_large:

            if sent_seg:
                raw_x, raw_len = toolbox.get_input_vec_tag(None, raw_file, char2idx, limit=args.sent_limit + 100,
                                                           is_space=is_space)
            else:
                raw_x, raw_len = toolbox.get_input_vec_raw(None, raw_file, char2idx, limit=args.sent_limit + 100,
                                                           sent_seg=sent_seg, is_space=is_space)

            if sent_seg:
                print 'Joint sentence segmentation...'
            else:
                print 'Raw setences: %d instances.' % len(raw_x[0])

            max_step = raw_len

        else:

            max_step = args.sent_limit

        if ngram > 1:
            gram2idx = toolbox.get_ngram_dic(grams)
            new_grams = toolbox.get_new_grams(raw_file, gram2idx, is_raw=True, is_space=is_space)

            if not args.segment_large:
                if sent_seg:
                    raw_grams = toolbox.get_gram_vec_tag(None, raw_file, gram2idx, limit=args.sent_limit + 100,
                                                         is_space=is_space)
                else:
                    raw_grams = toolbox.get_gram_vec(None, raw_file, gram2idx, is_raw=True, limit=args.sent_limit + 100,
                                                     sent_seg=sent_seg, is_space=is_space)

                raw_x += raw_grams

        if not args.segment_large:
            for k in range(len(raw_x)):
                raw_x[k] = toolbox.pad_zeros(raw_x[k], max_step)

    config = tf.ConfigProto(allow_soft_placement=True)
    gpu_config = "/gpu:" + str(args.gpu)

    transducer = None
    transducer_graph = None
    trans_model = None
    trans_init = None

    if len(trans_dict) > 200:
        transducer = toolbox.get_dict_vec(trans_dict, char2idx)
    t = time()

    initializer = tf.contrib.layers.xavier_initializer()

    if transducer is not None:
        transducer_graph = tf.Graph()
        with transducer_graph.as_default():
            with tf.variable_scope("transducer") as scope:
                trans_model = Seq2seq(path + '/' + model_file + '_transducer')
                trans_fin = open(path + '/' + model_file + '_transducer_model', 'rb')
                trans_param_dic = pickle.load(trans_fin)
                trans_fin.close()

                tr_char_num = trans_param_dic['char_num']
                tr_rnn_dim = trans_param_dic['rnn_dim']
                tr_emb_dim = trans_param_dic['emb_dim']
                tr_max_x = trans_param_dic['max_x']
                tr_max_y = trans_param_dic['max_y']

                print 'Defining transducer...'
                trans_model.define(char_num=tr_char_num, rnn_dim=tr_rnn_dim, emb_dim=tr_emb_dim,
                                   max_x=tr_max_x, max_y=tr_max_y, write_trans_model=False)
            trans_init = tf.global_variables_initializer()
        transducer_graph.finalize()

    print 'Initialization....'
    main_graph = tf.Graph()
    with main_graph.as_default():
        with tf.variable_scope("tagger") as scope:
            model = Model(nums_chars=nums_chars, nums_tags=nums_tags, buckets_char=[max_step], counts=[200],
                          crf=crf, ngram=nums_ngrams, batch_size=args.tag_batch, is_space=is_space)

            model.main_graph(trained_model=None, scope=scope, emb_dim=emb_dim, cell=cell,
                             rnn_dim=rnn_dim, rnn_num=rnn_num, drop_out=drop_out)

        model.define_updates(new_chars=new_chars, emb_path=emb_path, char2idx=char2idx)

        init = tf.global_variables_initializer()

        print 'Done. Time consumed: %d seconds' % int(time() - t)
    main_graph.finalize()

    idx=None

    if args.ensemble:
        idx = 1
        main_sess = []
        while os.path.isfile(path + '/' + model_file + '_' + str(idx) + '_weights.index'):
            main_sess.append(tf.Session(config=config, graph=main_graph))
            idx += 1
    else:
        main_sess = tf.Session(config=config, graph=main_graph)

    if crf:
        decode_graph = tf.Graph()

        with decode_graph.as_default():
            model.decode_graph()
        decode_graph.finalize()

        decode_sess = tf.Session(config=config, graph=decode_graph)

        sess = [main_sess, decode_sess]

    else:
        sess = [main_sess, None]

    with tf.device(gpu_config):
        ens_model = None
        print 'Loading weights....'
        if args.ensemble:
            for i in range(1, idx):
                print 'Ensemble: ' + str(i)
                main_sess[i - 1].run(init)
                model.run_updates(main_sess[i - 1], weight_path + '_' + str(i) + '_weights')
        else:
            main_sess.run(init)
            model.run_updates(main_sess, weight_path + '_weights')

        if transducer is not None:
            print 'Loading transducer...'
            t = time()
            trans_sess = tf.Session(config=config, graph=transducer_graph)
            trans_sess.run(trans_init)
            if os.path.isfile(path + '/' + model_file + '_transducer_weights'):
                trans_weight_path = path + '/' + model_file + '_transducer_weights'
                trans_weight_path = trans_weight_path.replace('//', '/')
                trans_model.saver.restore(trans_sess, trans_weight_path)
            sess.append(trans_sess)

        if args.action == 'test':
            test_y_raw = [line.strip() for line in codecs.open(path + '/raw_test.txt', 'rb', encoding='utf-8')]
            model.test(test_x, test_y_raw, test_y_gold, idx2tag, idx2char, unk_chars_idx, sub_dict, trans_dict, sess,
                       transducer=trans_model, ensemble=args.ensemble, batch_size=args.test_batch, sent_seg=sent_seg,
                       bias=args.segmentation_bias, outpath=args.output_path, trans_type=args.transduction_type)

        if args.action == 'tag':

            if not args.segment_large:
                raw_sents = []
                for line in codecs.open(raw_file, 'rb', encoding='utf-8'):
                    line = line.strip()
                    if len(line) > 0:
                        raw_sents.append(line)
                model.tag(raw_x, raw_sents, idx2tag, idx2char, unk_chars_idx, sub_dict, trans_dict, sess,
                          transducer=trans_model, outpath=args.output_path, ensemble=args.ensemble,
                          batch_size=args.tag_batch, sent_seg=sent_seg, seg_large=args.segment_large, form=args.format)
            else:
                count = 0
                c_line = 0
                l_writer = codecs.open(args.output_path, 'w', encoding='utf-8')
                out = []
                with codecs.open(raw_file, 'r', encoding='utf-8') as l_file:
                    lines = []
                    for line in l_file:
                        line = line.strip()
                        if len(line) > 0:
                            lines.append(line)
                        else:
                            c_line += 1
                        if c_line >= args.large_size:
                            count += len(lines)
                            c_line = 0
                            print count
                            if args.sent_seg:
                                raw_x, _ = toolbox.get_input_vec_tag(None, None, char2idx, lines=lines,
                                                                     limit=args.sent_limit, is_space=is_space)
                            else:
                                raw_x, _ = toolbox.get_input_vec_raw(None, None, char2idx, lines=lines,
                                                                     limit=args.sent_limit, sent_seg=sent_seg,
                                                                     is_space=is_space)
                            if ngram > 1:
                                if sent_seg:
                                    raw_grams = toolbox.get_gram_vec_tag(None, None, gram2idx, lines=lines,
                                                                         limit=args.sent_limit, is_space=is_space)
                                else:
                                    raw_grams = toolbox.get_gram_vec(None, None, gram2idx, lines=lines, is_raw=True,
                                                                     limit=args.sent_limit, sent_seg=sent_seg,
                                                                     is_space=is_space)
                                raw_x += raw_grams

                            for k in range(len(raw_x)):
                                raw_x[k] = toolbox.pad_zeros(raw_x[k], max_step)

                            predition, multi = model.tag(raw_x, lines, idx2tag, idx2char, unk_chars_idx, sub_dict,
                                                         trans_dict, sess, transducer=trans_model,
                                                         outpath=args.output_path, ensemble=args.ensemble,
                                                         batch_size=args.tag_batch, sent_seg=sent_seg,
                                                         seg_large=args.segment_large, form=args.format)

                            if args.only_tokenised:
                                for l_out in predition:
                                    if len(l_out.strip()) > 0:
                                        l_writer.write(l_out + '\n')
                            else:
                                for tagged_t, multi_t in zip(predition, multi):
                                    if len(tagged_t.strip()) > 0:
                                        l_writer.write('#sent_tok: ' + tagged_t + '\n')
                                        idx = 1
                                        tgs = multi_t.split('  ')
                                        pl = ''
                                        for _ in range(8):
                                            pl += '\t' + '_'
                                        for tg in tgs:
                                            if '!#!' in tg:
                                                segs = tg.split('!#!')
                                                l_writer.write(str(idx) + '-' + str(int(segs[1]) + idx - 1) + '\t' +
                                                               segs[0] + pl + '\n')
                                            else:
                                                l_writer.write(str(idx) + '\t' + tg + pl + '\n')
                                                idx += 1
                                        l_writer.write('\n')
                            lines = []
                    if len(lines) > 0:

                        if args.sent_seg:
                            raw_x, _ = toolbox.get_input_vec_tag(None, None, char2idx, lines=lines,
                                                                      limit=args.sent_limit, is_space=is_space)
                        else:
                            raw_x, _ = toolbox.get_input_vec_raw(None, None, char2idx, lines=lines,
                                                                      limit=args.sent_limit, sent_seg=sent_seg,
                                                                      is_space=is_space)
                        if ngram > 1:
                            if sent_seg:
                                raw_grams = toolbox.get_gram_vec_tag(None, None, gram2idx, lines=lines,
                                                                     limit=args.sent_limit, is_space=is_space)
                            else:
                                raw_grams = toolbox.get_gram_vec(None, None, gram2idx, lines=lines, is_raw=True,
                                                                 limit=args.sent_limit, sent_seg=sent_seg,
                                                                 is_space=is_space)
                            raw_x += raw_grams

                        for k in range(len(raw_x)):
                            raw_x[k] = toolbox.pad_zeros(raw_x[k], max_step)

                        prediction, multi = model.tag(raw_x, lines, idx2tag, idx2char, unk_chars_idx, sub_dict,
                                                      trans_dict, sess, transducer=trans_model,
                                                      outpath=args.output_path, ensemble=args.ensemble,
                                                      batch_size=args.tag_batch, sent_seg=sent_seg,
                                                      seg_large=args.segment_large, form=args.format)

                        if args.only_tokenised:
                            for l_out in prediction:
                                if len(l_out.strip()) > 0:
                                    l_writer.write(l_out + '\n')
                        else:
                            for tagged_t, multi_t in zip(prediction, multi):
                                if len(tagged_t.strip()) > 0:
                                    l_writer.write('#sent_tok: ' + tagged_t + '\n')
                                    idx = 1
                                    tgs = multi_t.split('  ')
                                    pl = ''
                                    for _ in range(8):
                                        pl += '\t' + '_'
                                    for tg in tgs:
                                        if '!#!' in tg:
                                            segs = tg.split('!#!')
                                            l_writer.write(str(idx) + '-' + str(int(segs[1]) + idx - 1) + '\t' +
                                                           segs[0] + pl + '\n')
                                        else:
                                            l_writer.write(str(idx) + '\t' + tg + pl + '\n')
                                            idx += 1
                                    l_writer.write('\n')
                l_writer.close()

        print 'Done.'
