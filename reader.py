# -*- coding: utf-8 -*-
"""
@author: Yan Shao, yan.shao@lingfil.uu.se
"""
import codecs


def conll(path, is_dev=True):
    sents = []
    sent = []
    cter = 0
    sents_dev = None
    if not is_dev:
        sents_dev = []
    for line in codecs.open(path, 'rb', encoding='utf8'):
        line = line.strip()
        segs = line.split('\t')
        if len(segs) == 10:
            if '.' not in segs[0]:
                sent.append(tuple(segs))
        elif len(sent) > 0:
            if not is_dev and cter == 9:
                sents_dev.append(sent)
                cter = 0
            else:
                sents.append(sent)
                cter += 1
            sent = []
    if is_dev:
        return sents
    else:
        return sents, sents_dev


def raw(path):
    sents = []
    for line in codecs.open(path, 'rb', encoding='utf-8'):
        line = line.strip()
        sents.append(line)
    return sents


def conll_gold(path):
    sents = []
    st = ''
    for line in codecs.open(path, 'rb', encoding='utf-8'):
        line = line.strip()
        segs = line.split('\t')
        if len(segs) == 10:
            if '.' not in segs[0] and '-' not in segs[0]:
                st += '  ' + segs[1]
        elif len(st) > 0:
            sents.append(st.strip())
            st = ''
    return sents

