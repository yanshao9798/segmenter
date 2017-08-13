# -*- coding: utf-8 -*-
import copy


def eq_tokens(gt, st):
    if gt.strip() in ["``", "''"] and st.strip() == '"':
        return True
    else:
        return gt == st

lcs = {}


def score(gtokens, stokens):
    for i in range(0, len(gtokens) + 1):
        lcs[(i, 0)] = 0
    for j in range(0, len(stokens) + 1):
        lcs[(0, j)] = 0
    for i in range(1, len(gtokens) + 1):
        for j in range(1, len(stokens) + 1):
            if eq_tokens(gtokens[i - 1], stokens[j - 1]):
                lcs[(i, j)] = lcs[(i - 1, j - 1)] + 1
            else:
                if lcs[(i - 1, j)] >= lcs[(i, j - 1)]:
                    lcs[(i, j)] = lcs[(i - 1, j)]
                else:
                    lcs[(i, j)] = lcs[(i, j - 1)]

    tp = lcs[(len(gtokens), len(stokens))]
    g = len(gtokens)
    s = len(stokens)

    #precision = float(tp)/float(s)
    #recall = float(tp)/float(g)
    #f1score=(2*precision*recall)/(precision + recall)

    return tp, g, s


def exact_match(g, pre):
    acc = 0
    for g_t, p_t in zip(g, pre):
        if g_t == p_t:
            acc += 1
    return acc


def evaluator(prediction, gold, prediction_raw=None, gold_raw=None, verbose=False):
    if prediction_raw is None or gold_raw is None:
        prediction = prediction[0]
        assert len(prediction) == len(gold)
        tp, s, g, pp, rp, tw = 0, 0, 0, 0, 0, 0
        for pre, gd in zip(prediction, gold):
            pre_t = pre.split('  ')
            gd_t = gd.split('  ')
            tt, t_g, t_s = score(gd_t, pre_t)
            tp += tt
            g += t_g
            s += t_s
            if verbose:
                pp += len(pre_t)
                rp += len(gd_t)
                sl = len(''.join(pre_t))
                tw += (1 + sl) * sl / 2
        precision = float(tp) / float(s)
        recall = float(tp) / float(g)
        if precision == 0 and recall == 0:
            f1score = 0
        else:
            f1score = (2 * precision * recall) / (precision + recall)

        if verbose:
            tnr = 1 - float(pp - tp) / float(tw - rp)
            return precision, recall, f1score, tnr
        else:
            return precision, recall, f1score
    else:
        prediction = copy.copy(prediction[0])
        prediction_raw = copy.copy(prediction_raw[0])
        gold = copy.copy(gold)
        gold_raw = copy.copy(gold_raw)
        prediction_raw = ["".join(pre.split()) for pre in prediction_raw]
        gold_raw = ["".join(gd.split()) for gd in gold_raw]
        assert len(prediction) == len(prediction_raw)
        assert len(gold) == len(gold_raw)
        n_prediction = len(prediction)
        n_gold = len(gold)
        correct = 0
        l_prediction = 0
        l_gold = 0
        pre_tokens = []
        gd_tokens = []
        tp, s, g = 0, 0, 0
        last_correct = True
        while prediction_raw and gold_raw and prediction and gold:
            if prediction_raw[0] == gold_raw[0] or (last_correct and len(prediction_raw[0]) == len(gold_raw[0])):  # words right
                correct += 1
                l_prediction += len(prediction_raw[0])  # move
                l_gold += len(gold_raw[0])
                pre_tokens = prediction[0].split('  ')
                gd_tokens = gold[0].split('  ')
                tt, t_g, t_s = score(gd_tokens, pre_tokens)
                tp += tt
                g += t_g
                s += t_s
                pre_tokens = []
                gd_tokens = []
                prediction_raw.pop(0)
                gold_raw.pop(0)
                prediction.pop(0)
                gold.pop(0)
                last_correct = True
            else:
                if l_prediction == l_gold:
                    if len(gd_tokens) < 1000:
                        tt, t_g, t_s = score(gd_tokens, pre_tokens)
                        tp += tt
                        g += t_g
                        s += t_s
                    else:
                        g += len(gd_tokens)
                        s += len(pre_tokens)
                    l_prediction += len(prediction_raw[0])  # move
                    l_gold += len(gold_raw[0])
                    pre_tokens = prediction[0].split('  ')
                    gd_tokens = gold[0].split('  ')
                    prediction_raw.pop(0)
                    gold_raw.pop(0)
                    prediction.pop(0)
                    gold.pop(0)
                    last_correct = False
                elif l_prediction < l_gold:
                    l_prediction += len(prediction_raw[0])
                    pre_tokens += prediction[0].split('  ')
                    prediction_raw.pop(0)
                    prediction.pop(0)
                    last_correct = False
                elif l_prediction > l_gold:
                    gd_tokens += gold[0].split('  ')
                    l_gold += len(gold_raw[0])  # move
                    gold_raw.pop(0)
                    gold.pop(0)
                    last_correct = False

        if correct > 0:
            sent_precision = float(correct) / float(n_prediction)
            sent_recall = float(correct) / float(n_gold)
            sent_f1score = (2 * sent_precision * sent_recall) / (sent_precision + sent_recall)
        else:
            sent_precision = 0
            sent_recall = 0
            sent_f1score = 0

        if tp > 0:
            precision = float(tp) / float(s)
            recall = float(tp) / float(g)
            f1score = (2 * precision * recall) / (precision + recall)
        else:
            precision = 0
            recall = 0
            f1score = 0

        return precision, recall, f1score, sent_precision, sent_recall, sent_f1score


def sent_evaluator(prediction, gold):
    prediction = ["".join(pre.split()) for pre in prediction]
    gold = ["".join(gd.split()) for gd in gold]
    n_prediction = len(prediction)
    n_gold = len(gold)
    correct = 0
    l_prediction = 0
    l_gold = 0
    while prediction and gold:
        if prediction[0] == gold[0]:  # words right
            correct += 1
            l_prediction = 0  # move
            l_gold = 0
            prediction.pop(0)
            gold.pop(0)
        else:
            if l_prediction == l_gold:
                l_prediction = 0  # move
                l_gold = 0
                prediction.pop(0)
                gold.pop(0)
            elif l_prediction < l_gold:
                l_prediction += len(prediction[0])
                prediction.pop(0)
            elif l_prediction > l_gold:
                l_gold += len(gold[0])  # move
                gold.pop(0)
    if correct > 0:
        precision = float(correct) / float(n_prediction)
        recall = float(correct) / float(n_gold)
        f1score = (2 * precision * recall) / (precision + recall)
    else:
        precision = 0
        recall = 0
        f1score = 0
    return precision, recall, f1score


def trans_evaluator(prediction, gold):
    assert len(prediction) == len(gold)
    acc = float(exact_match(prediction, gold))/len(prediction)
    tp, s, g = 0, 0, 0
    for pre, gd in zip(prediction, gold):
        pre_t = pre.split(' ')
        gd_t = gd.split(' ')
        tt, t_g, t_s = score(gd_t, pre_t)
        tp += tt
        g += t_g
        s += t_s
    precision = float(tp) / float(s)
    recall = float(tp) / float(g)
    if precision == 0 and recall == 0:
        f1score = 0
    else:
        f1score = (2 * precision * recall) / (precision + recall)
    return acc, f1score
