# -*- coding: utf-8 -*-
import random
import toolbox
import numpy as np


def train(sess, model, batch_size, config, lr, lrv, data, dr=None, drv=None, verbose=False):
    assert len(data) == len(model)
    num_items = len(data)
    samples = zip(*data)
    random.shuffle(samples)
    start_idx = 0
    n_samples = len(samples)
    model.append(lr)
    if dr is not None:
        model.append(dr)
    while start_idx < len(samples):
        if verbose:
            print '%d' % (start_idx * 100 / n_samples) + '%'
        next_batch_samples = samples[start_idx:start_idx + batch_size]
        real_batch_size = len(next_batch_samples)
        if real_batch_size < batch_size:
            next_batch_samples.extend(samples[:batch_size - real_batch_size])
        holders = []
        for item in range(num_items):
            holders.append([s[item] for s in next_batch_samples])
        holders.append(lrv)
        if dr is not None:
            holders.append(drv)
        sess.run(config, feed_dict={m: h for m, h in zip(model, holders)})
        start_idx += batch_size


def softmax(x):
    dim = len(list(x.shape)) - 1
    anp = np.exp(x - np.max(x, axis=dim, keepdims=True))
    return anp / np.sum(anp, axis=dim, keepdims=True)


def predict(sess, model, data, dr=None, transitions=None, crf=True, decode_sess=None, scores=None, decode_holders=None,
            argmax=True, batch_size=100, ensemble=False, verbose=False):
    en_num = None
    if ensemble:
        en_num = len(sess)
    num_items = len(data)
    input_v = model[:num_items]
    if dr is not None:
        input_v.append(dr)
    predictions = model[num_items:]
    output = [[] for _ in range(len(predictions))]
    samples = zip(*data)
    start_idx = 0
    n_samples = len(samples)
    if crf > 0:
        trans = []
        for i in range(len(predictions)):
            if ensemble:
                en_trans = 0
                for en_sess in sess:
                    en_trans += en_sess.run(transitions[i])
                trans.append(en_trans/en_num)
            else:
                trans.append(sess.run(transitions[i]))
    while start_idx < n_samples:
        if verbose:
            print '%d' % (start_idx*100/n_samples) + '%'
        next_batch_input = samples[start_idx:start_idx + batch_size]
        batch_size = len(next_batch_input)
        holders= []
        for item in range(num_items):
            holders.append([s[item] for s in next_batch_input])
        if dr is not None:
            holders.append(0.0)
        length = np.sum(np.sign(holders[0]), axis=1)
        if crf > 0:
            assert transitions is not None and len(transitions) == len(predictions) and len(scores) == len(decode_holders)
            for i in range(len(predictions)):
                if ensemble:
                    en_obs = 0
                    for en_sess in sess:
                        en_obs += en_sess.run(predictions[i], feed_dict={i: h for i, h in zip(input_v, holders)})
                    ob = en_obs/en_num
                else:
                    ob = sess.run(predictions[i], feed_dict={i: h for i, h in zip(input_v, holders)})
                pre_values = [ob, trans[i], length, batch_size]
                assert len(pre_values) == len(decode_holders[i])
                max_scores, max_scores_pre = decode_sess.run(scores[i], feed_dict={i: h for i, h in zip(decode_holders[i], pre_values)})
                output[i].extend(toolbox.viterbi(max_scores, max_scores_pre, length, batch_size))
        elif argmax:
            for i in range(len(predictions)):
                pre = sess.run(predictions[i], feed_dict={i: h for i, h in zip(input_v, holders)})
                dim_axis = len(list(pre.shape)) - 1
                if argmax is True:
                    pre = np.argmax(pre, axis= dim_axis)
                else:
                    pre = softmax(pre)
                    pre[:, :, 0][pre[:, :, 0] > argmax] = 1
                    pre[:, :, 0][pre[:, :, 0] <= argmax] = 0
                    pre = np.argmax(pre, axis=dim_axis)
                pre = pre.tolist()
                if dim_axis > 1:
                    pre = toolbox.trim_output(pre, length)
                output[i].extend(pre)
        else:
            for i in range(len(predictions)):
                pre = sess.run(predictions[i], feed_dict={i: h for i, h in zip(input_v, holders)})
                #pre = softmax(pre)
                dim_axis = len(list(pre.shape)) - 1
                if dim_axis > 1:
                    pre = toolbox.trim_output(pre, length)
                output[i].extend(pre)
        start_idx += batch_size
    return output


def train_seq2seq(sess, model, decoding, batch_size, config, lr, lrv, data, dr=None, drv=None, verbose=False):
    #assert len(data) == len(model)
    samples = zip(*data)
    random.shuffle(samples)
    start_idx = 0
    n_samples = len(samples)
    model.append(lr)
    model.append(decoding)
    if dr is not None:
        model.append(dr)
    while start_idx < len(samples):
        if verbose:
            print '%d' % (start_idx * 100 / n_samples) + '%'
        next_batch_samples = samples[start_idx:start_idx + batch_size]
        real_batch_size = len(next_batch_samples)
        if real_batch_size < batch_size:
            next_batch_samples.extend(samples[:batch_size - real_batch_size])
        holders = []
        next_batch_samples = zip(*next_batch_samples)
        for n_batch in next_batch_samples:
            n_batch = np.asarray(n_batch).T
            for b in n_batch:
                holders.append(b)
        holders.append(lrv)
        holders.append(False)
        if dr is not None:
            holders.append(drv)
        sess.run(config, feed_dict={m: h for m, h in zip(model, holders)})
        start_idx += batch_size


def predict_seq2seq(sess, model, decoding, data, decode_len, dr=None, argmax=True, batch_size=100, ensemble=False, verbose=False):
    num_items = len(data)
    in_len = len(data[0][0])
    input_v = model[:num_items*in_len + decode_len]
    input_v.append(decoding)
    if dr is not None:
        input_v.append(dr)
    predictions = model[num_items*in_len + decode_len:]
    output = []
    samples = zip(*data)
    start_idx = 0
    n_samples = len(samples)
    while start_idx < n_samples:
        if verbose:
            print '%d' % (start_idx * 100 / n_samples) + '%'
        next_batch_input = samples[start_idx:start_idx + batch_size]
        batch_size = len(next_batch_input)
        holders = []
        next_batch_input = zip(*next_batch_input)
        for n_batch in next_batch_input:
            n_batch = np.asarray(n_batch).T
            for b in n_batch:
                holders.append(b)
        for i in range(decode_len):
            holders.append(np.zeros(batch_size, dtype='int32'))
        holders.append(True)
        if dr is not None:
            holders.append(0.0)
        if argmax:
            pre = sess.run(predictions, feed_dict={i: h for i, h in zip(input_v, holders)})
            pre = [np.argmax(pre_t, axis=1) for pre_t in pre]
            pre = np.asarray(pre).T.tolist()
            pre = [np.trim_zeros(pre_t) for pre_t in pre]
            output += pre
        else:
            pre = sess.run(predictions, feed_dict={i: h for i, h in zip(input_v, holders)})
            output += pre
        start_idx += batch_size
    return output