# -*- coding: utf-8 -*-
"""
@author: Yan Shao, yan.shao@lingfil.uu.se
"""
import codecs
import sys
import numpy as np
import random
import os
import math

sys = reload(sys)
sys.setdefaultencoding('utf-8')

punc = ['!', ')', ',', '.', ';', ':', '?', '»', '...', '..', '....', '%', 'º', '²', '°', '¿', '¡', '(', '«',
        '"', '\'', '-', '。', '·', '।', '۔']

def pre_token(line):
    out = []
    for seg in line.split(' '):
        f_out = []
        b_out = []
        while len(seg) > 0 and (seg[0] in punc or ('0' <= seg[0] <= '9')):
            f_out.append(seg[0])
            seg = seg[1:]
        while len(seg) > 0 and (seg[-1] in punc or ('0' <= seg[-1] <= '9')):
            b_out = [seg[-1]] + b_out
            seg = seg[:-1]
        if len(seg) > 0:
            out += f_out + [seg] + b_out
        else:
            out += f_out + b_out
    return out


def get_chars(path, filelist, sea=False):
    char_set = {}
    out_char = codecs.open(path + '/chars.txt', 'w', encoding='utf-8')
    for i, file_name in enumerate(filelist):
        for line in codecs.open(path + '/' + file_name, 'rb', encoding='utf-8'):
            line = line.strip()
            if sea=='sea':
                line = pre_token(line)
            for ch in line:
                if ch in char_set:
                    if i == 0:
                        char_set[ch] += 1
                else:
                    char_set[ch] = 1
    for k, v in char_set.items():
        out_char.write(k + '\t' + str(v) + '\n')
    out_char.close()


def get_dicts(path, sent_seg, tag_scheme='BIES', crf=1):
    char2idx = {'<P>': 0, '<UNK>': 1, '<#>': 2}
    unk_chars = []
    idx = 3
    for line in codecs.open(path + '/chars.txt', 'r', encoding='utf-8'):
        segs = line.split('\t')
        if len(segs[0].strip()) == 0:
            if ' ' not in char2idx:
                char2idx[' '] = idx
                idx += 1
        else:
            char2idx[segs[0]] = idx
            if int(segs[1]) == 1:
                unk_chars.append(idx)
            idx += 1
    idx2char = {k: v for v, k in char2idx.items()}
    if tag_scheme == 'BI':
        if crf > 0:
            tag2idx = {'<P>': 0, 'B': 1, 'I': 2}
        else:
            tag2idx = {'B': 0, 'I': 1}
    else:
        if crf > 0:
            tag2idx = {'<P>': 0, 'B': 1, 'I': 2, 'E': 3, 'S': 4}
            idx = 5
        else:
            tag2idx = {'B': 0, 'I':1, 'E':2, 'S':3}
            idx = 4
        for line in codecs.open(path + '/tags.txt', 'r', encoding='utf-8'):
            line = line.strip()
            if line not in tag2idx:
                tag2idx[line] = idx
                idx += 1
        if sent_seg:
            tag2idx['T'] = idx
            tag2idx['U'] = idx + 1
    idx2tag = {k: v for v, k in tag2idx.items()}

    trans_dict = {}
    key = ''
    if os.path.isfile(path + '/dict.txt'):
        for line in codecs.open(path + '/dict.txt', 'r', encoding='utf-8'):
            line = line.strip()
            if len(line) > 0:
                segs = line.split('\t')
                if len(segs) == 1:
                    key = segs[0]
                    trans_dict[key] = None
                elif len(segs) == 2:
                    if trans_dict[key] is None:
                        trans_dict[key] = segs[0].replace(' ', '  ')

    return char2idx, unk_chars, idx2char, tag2idx, idx2tag, trans_dict


def ngrams(raw, gram, is_space):
    gram_set = {}
    li = gram/2
    ri = gram - li - 1
    p = '<PAD>'
    last_line = ''
    is_first = True
    for line in raw:
        for i in range(len(line)):
            if i - li < 0:
                if is_space != 'sea':
                    lp = p * (li - i) + line[:i]
                else:
                    lp = [p] * (li - i) + line[:i]
            else:
                lp = line[i - li:i]
            if i + ri + 1 > len(line):
                if is_space != 'sea':
                    rp = line[i:] + p*(i + ri + 1 - len(line))
                else:
                    rp = line[i:] + [p] * (i + ri + 1 - len(line))
            else:
                rp = line[i:i+ri+1]
            ch = lp + rp
            if is_space == 'sea':
                ch = '_'.join(ch)
            if ch in gram_set:
                gram_set[ch] += 1
            else:
                gram_set[ch] = 1
        if is_first:
            is_first = False
        else:
            if is_space is True:
                last_line += ' '
            start_idx = len(last_line) - ri
            if start_idx < 0:
                start_idx = 0
            end_idx = li + len(last_line)
            j_line = last_line + line
            for i in range(start_idx, end_idx):
                if i - li < 0:
                    if is_space != 'sea':
                        j_lp = p * (-i) + j_line[start_idx:i]
                    else:
                        j_lp = [p] * (-i) + j_line[start_idx:i]
                else:
                    j_lp = j_line[i - li:i]
                if i + ri + 1 > len(j_line):
                    if is_space != 'sea':
                        j_rp = j_line[i:end_idx] + p * (ri + i + 1 - len(j_line))
                    else:
                        j_rp = j_line[i:end_idx] + [p] * (ri + i + 1 - len(j_line))
                else:
                    j_rp = j_line[i:ri + 1 + i]
                j_ch = j_lp + j_rp
                if is_space == 'sea':
                    ch = '_'.join(j_ch)
                if ch in gram_set:
                    gram_set[ch] += 1
                else:
                    gram_set[ch] = 1
        last_line = line
    return gram_set


def get_ngrams(path, ng, is_space):
    raw = []
    for line in codecs.open(path + '/raw_train.txt', 'r', encoding='utf-8'):
        if is_space == 'sea':
            segs = pre_token(line.strip())
        else:
            segs = line.strip()
        raw.append(segs)
    if ng > 1:
        for i in range(2, ng + 1):
            out_gram = codecs.open(path + '/' + str(i) + 'gram.txt', 'w', encoding='utf-8')
            grams = ngrams(raw, i, is_space)
            for k, v in grams.items():
                out_gram.write(k + '\t' + str(v) + '\n')
            out_gram.close()


def read_ngrams(path, ng):
    ngs = []
    for i in range(2, ng + 1):
        ng = {}
        for line in codecs.open(path + '/' + str(i) + 'gram.txt', 'r', encoding='utf-8'):
            line = line.rstrip()
            segs = line.split('\t')
            while len(segs[0]) < i:
                segs[0] += ' '
            ng[segs[0]] = int(segs[1])
        ngs.append(ng)
    return ngs


def get_sample_embedding(path, emb, chars2idx):
    chars = chars2idx.keys()
    short_emb = emb[emb.index('/') + 1: emb.index('.')]
    emb_dic = {}
    valid_chars=[]
    for line in codecs.open(emb, 'rb', encoding='utf-8'):
        line = line.strip()
        sets = line.split(' ')
        emb_dic[sets[0]] = np.asarray(sets[1:], dtype='float32')
    fout = codecs.open(path + '/' + short_emb + '_sub.txt', 'w', encoding='utf-8')
    for ch in chars:
        p_line = ch
        if ch in emb_dic:
            valid_chars.append(ch)
            for emb in emb_dic[ch]:
                p_line += ' ' + unicode(emb)
            fout.write(p_line + '\n')
    fout.close()


def read_sample_embedding(path, short_emb, char2idx):
    emb_values = []
    valid_chars = []
    emb_dic={}
    for line in codecs.open(path + '/' + short_emb + '_sub.txt', 'rb', encoding='utf-8'):
        first_ch = line[0]
        line = line.rstrip()
        sets = line.split(' ')
        if first_ch == ' ':
            emb_dic[' '] = np.asarray(sets, dtype='float32')
        else:
            emb_dic[sets[0]] = np.asarray(sets[1:], dtype='float32')
    emb_dim = len(emb_dic.items()[0][1])
    for ch in char2idx.keys():
        if ch in emb_dic:
            emb_values.append(emb_dic[ch])
            valid_chars.append(ch)
        else:
            rand = np.random.uniform(-math.sqrt(float(3) / emb_dim), math.sqrt(float(3) / emb_dim), emb_dim)
            emb_values.append(np.asarray(rand, dtype='float32'))
    emb_dim = len(emb_values[0])
    return emb_dim, emb_values, valid_chars


def get_sent_raw(path, fname, is_space=True):
    long_line = ''
    for line in codecs.open(path + '/' + fname, 'r', encoding='utf-8'):
        line = line.strip()
        if is_space:
            long_line += ' ' + line
        else:
            long_line += line
    if is_space:
        long_line = long_line[1:]

    return long_line


def chop(line, ad_s, limit):
    out = []
    chopped = False
    while len(line) > 0:
        if chopped:
            s_line = line[:limit - 1]
            s_line = [ad_s] + s_line
        else:
            chopped = True
            s_line = line[:limit]
        out.append(s_line)
        line = line[limit - 10:]
        if len(line) < 10:
            line = ''
    while len(out) > 0 and len(out[-1]) < limit-1:
        out[-1].append(0)
    return out


def get_input_vec(path, fname, char2idx, tag2idx, limit=500, sent_seg=False, is_space=True, train_size=-1, ignore_space=False):
    ct = 0
    max_len = 0
    space_idx = None
    if is_space is True:
        assert ' ' in char2idx
        space_idx = char2idx[' ']
    x_indices = []
    y_indices = []
    s_count = 0
    l_count = 0
    x = []
    y = []

    n_sent = 0

    if sent_seg:
        for line in codecs.open(path + '/' + fname, 'r', encoding='utf-8'):
            line = line.strip()
            if len(line) == 0:
                ct = 0
            elif ct == 0:
                if is_space == 'sea':
                    line = pre_token(line)
                for ch in line:
                    if len(ch.strip()) == 0:
                        x.append(char2idx[' '])
                    elif ch in char2idx:
                        x.append(char2idx[ch])
                    else:
                        x.append(char2idx['<UNK>'])
                if is_space is True and not ignore_space:
                    x = [space_idx] + x
                x_indices += x
                x = []
                ct = 1
            elif ct == 1:
                for ch in line:
                    y.append(tag2idx[ch])
                if y[-1] == tag2idx['S']:
                    y[-1] = tag2idx['T']
                else:
                    y[-1] = tag2idx['U']
                if is_space is True and not ignore_space:
                    y = [tag2idx['X']] + y
                y_indices += y
                y = []
                n_sent += 1
            if 0 < train_size <= n_sent:
                break
        x_indices = chop(x_indices, char2idx['<#>'], limit)
        y_indices = chop(y_indices, tag2idx['I'], limit)
        max_len = limit
    else:
        for line in codecs.open(path + '/' + fname, 'r', encoding='utf-8'):
            line = line.strip()
            if len(line) == 0:
                ct = 0
            elif ct == 0:
                if is_space == 'sea':
                    line = pre_token(line)
                max_len = max(max_len, len(line))
                s_count += 1
                if len(line) > limit:
                    l_count += 1
                chopped = False
                while len(line) > 0:
                    s_line = line[:limit - 1]
                    line = line[limit - 10:]
                    if len(line) < 10:
                        line = ''
                    if not chopped:
                        chopped = True
                    else:
                        x.append(char2idx['<#>'])
                    for ch in s_line:
                        if len(ch.strip()) == 0:
                            x.append(char2idx[' '])
                        elif ch in char2idx:
                            x.append(char2idx[ch])
                        else:
                            x.append(char2idx['<UNK>'])
                    x_indices.append(x)
                    x = []
                ct = 1
            elif ct == 1:
                chopped = False
                while len(line) > 0:
                    s_line = line[:limit - 1]
                    line = line[limit - 10:]
                    if len(line) < 10:
                        line = ''
                    if not chopped:
                        chopped = True
                    else:
                        y.append(tag2idx['I'])
                    for ch in s_line:
                        y.append(tag2idx[ch])
                    y_indices.append(y)
                    y = []
                n_sent += 1
            if 0 < train_size <= n_sent:
                break
        max_len = min(max_len, limit)
        if l_count > 0:
            print '%d (out of %d) sentences are chopped.' % (l_count, s_count)
    return [x_indices], [y_indices], max_len


def get_input_vec_sent(path, fname, char2idx, win_size, is_space=True):
    pre_line = ''
    c_line = ''
    x = []
    y = []
    is_first = True
    for line in codecs.open(path + '/' + fname, 'r', encoding='utf-8'):
        line = line.strip()
        if is_space == 'sea':
            line = pre_token(line)
        start_idx = len(pre_line)
        if is_space is True:
            j_line = pre_line + ' ' + c_line + ' ' + line
            end_idx = start_idx + len(c_line) + 1
            if is_first:
                is_first = False
                j_line = j_line[1:]
                end_idx -= 1
        else:
            j_line = pre_line + c_line + line
            end_idx = start_idx + len(c_line)
        for i in range(start_idx, end_idx):
            indices = []
            for j in range(i - win_size, i + win_size + 1):
                if j < 0 or j >= len(j_line):
                    indices.append(char2idx['<P>'])
                else:
                    if j_line[j] in char2idx:
                        indices.append(char2idx[j_line[j]])
                    else:
                        indices.append(char2idx['<UNK>'])
            x.append(indices)
            if i == end_idx - 1:
                y.append(1)
            else:
                y.append(0)
        pre_line = c_line
        c_line = line
    if is_space is True:
        j_line = pre_line + ' ' + c_line
    else:
        j_line = pre_line + c_line
    start_idx = len(pre_line)
    end_idx = start_idx + len(c_line)
    for i in range(start_idx, end_idx):
        indices = []
        for j in range(i - win_size, i + win_size + 1):
            if j < 0 or j >= len(j_line):
                indices.append(char2idx['<P>'])
            else:
                if j_line[j] in char2idx:
                    indices.append(char2idx[j_line[j]])
                else:
                    indices.append(char2idx['<UNK>'])
        x.append(indices)
        if i == end_idx - 1:
            y.append(1)
        else:
            y.append(0)

    assert len(x) == len(y)
    return x, y


def get_input_vec_sent_raw(raws, char2idx, win_size):
    x = []
    for i in range(len(raws)):
        indices = []
        for j in range(i - win_size, i + win_size + 1):
            if j < 0 or j >= len(raws):
                indices.append(char2idx['<P>'])
            else:
                if raws[j] in char2idx:
                    indices.append(char2idx[raws[j]])
                else:
                    indices.append(char2idx['<UNK>'])
        x.append(indices)
    return x


def get_input_vec_raw(path, fname, char2idx, lines=None, limit=500, sent_seg=False, is_space=True, ignore_space=False):
    max_len = 0
    space_idx = None
    is_first = True
    if is_space is True:
        assert ' ' in char2idx
        space_idx = char2idx[' ']
    x_indices = []
    s_count = 0
    l_count = 0
    x = []
    if lines is None:
        assert fname is not None
        if path is None:
            real_path = fname
        else:
            real_path = path + '/' + fname
        lines = codecs.open(real_path, 'r', encoding='utf-8')
    if sent_seg:
        for line in lines:
            line = line.strip()
            if is_space == 'sea':
                line = pre_token(line)
            elif ignore_space:
                line = ''.join(line.split())
            for ch in line:
                if len(ch.strip()) == 0:
                    x.append(char2idx[' '])
                elif ch in char2idx:
                    x.append(char2idx[ch])
                else:
                    x.append(char2idx['<UNK>'])
            if is_space is True and not ignore_space:
                if is_first:
                    is_first = False
                else:
                    x = [space_idx] + x
            x_indices += x
            x = []
        x_indices = chop(x_indices, char2idx['<#>'], limit)
        max_len = limit
    else:
        for line in lines:
            line = line.strip()
            if len(line) > 0:
                if is_space == 'sea':
                    line = pre_token(line)
                elif ignore_space:
                    line = ''.join(line.split())
                max_len = max(max_len, len(line))
                s_count += 1

                for ch in line:
                    if len(ch.strip()) == 0:
                        x.append(char2idx[' '])
                    elif ch in char2idx:
                        x.append(char2idx[ch])
                    else:
                        x.append(char2idx['<UNK>'])

                if len(line) > limit:
                    l_count += 1
                    chop_x = chop(x, char2idx['<#>'], limit)
                    x_indices += chop_x
                else:
                    x_indices.append(x)
                x = []
        max_len = min(max_len, limit)
        if l_count > 0:
            print '%d (out of %d) sentences are chopped.' % (l_count, s_count)
    return [x_indices], max_len


def get_input_vec_tag(path, fname, char2idx, lines=None, limit=500, is_space=True):
    space_idx = None
    if is_space is True:
        assert ' ' in char2idx
        space_idx = char2idx[' ']
    x_indices = []
    out = []
    x = []
    is_first = True
    if lines is None:
        assert fname is not None
        if path is None:
            real_path = fname
        else:
            real_path = path + '/' + fname
        lines = codecs.open(real_path, 'r', encoding='utf-8')
    for line in lines:
        line = line.strip()
        if len(line) > 0:
            if is_space == 'sea':
                line = pre_token(line)
            if len(line) > 0:
                for ch in line:
                    if len(ch.strip()) == 0:
                        x.append(char2idx[' '])
                    elif ch in char2idx:
                        x.append(char2idx[ch])
                    else:
                        x.append(char2idx['<UNK>'])
                if is_space is True:
                    if is_first:
                        is_first = False
                    else:
                        x = [space_idx] + x
                x_indices += x
                x = []
            elif len(x_indices) > 0:
                x_indices = chop(x_indices, char2idx['<#>'], limit)
                out += x_indices
                x_indices = []
                is_first = True

    if len(x_indices) > 0:
        x_indices = chop(x_indices, char2idx['<#>'], limit)
        out += x_indices

    return [out], limit

def get_vecs(str, char2idx):
    out = []
    for ch in str:
        if ch in char2idx:
            out.append(char2idx[ch])
    return out


def get_dict_vec(trans_dict, char2idx):
    max_x, max_y = 0, 0
    x = []
    y = []
    for k, v in trans_dict.items():
        x.append(get_vecs(k, char2idx))
        y.append(get_vecs(v.replace('  ', ' '), char2idx) + [2])
        if len(k) > max_x:
            max_x = len(k)
        if len(v) > max_y:
            max_y = len(v)
    max_x += 5
    max_y += 5
    x = pad_zeros(x, max_x)
    y = pad_zeros(y, max_y)
    assert len(x) == len(y)
    num = len(x)
    xy = zip(x, y)
    random.shuffle(xy)
    xy = zip(*xy)
    t_x = xy[0][:int(num * 0.95)]
    t_y = xy[1][:int(num * 0.95)]
    v_x = xy[0][int(num * 0.95):]
    v_y = xy[1][int(num * 0.95):]
    return t_x, t_y, v_x, v_y


def get_ngram_dic(ng):
    gram_dics = []
    for i, gram in enumerate(ng):
        g_dic = {'<P>': 0, '<UNK>': 1, '<#>': 2}
        idx = 3
        for g in gram.keys():
            if gram[g] > 1:
                g_dic[g] = idx
            else:
                g_dic[g] = 1
            idx += 1
        gram_dics.append(g_dic)
    return gram_dics


def gram_vec(raw, dic, limit=500, sent_seg=False, is_space=True):
    out = []
    if is_space == 'sea':
        ngram = len(dic.keys()[0].split('_'))
    else:
        ngram = 0
        for k in dic.keys():
            if '<PAD>' not in k:
                ngram = len(k)
                break
    li = ngram/2
    ri = ngram - li - 1
    p = '<PAD>'
    indices = []
    is_first = True
    if sent_seg:
        last_line = ''
        for line in raw:
            for i in range(len(line)):
                if i - li < 0:
                    if is_space != 'sea':
                        lp = p * (li - i) + line[:i]
                    else:
                        lp = [p] * (li - i) + line[:i]
                else:
                    lp = line[i - li:i]
                if i + ri + 1 > len(line):
                    if is_space != 'sea':
                        rp = line[i:] + p * (i + ri + 1 - len(line))
                    else:
                        rp = line[i:] + [p] * (i + ri + 1 - len(line))
                else:
                    rp = line[i:i + ri + 1]
                ch = lp + rp
                if is_space == 'sea':
                    ch = '_'.join(ch)
                if ch in dic:
                    indices.append(dic[ch])
                else:
                    indices.append(dic['<UNK>'])
            if is_first:
                is_first = False
            else:
                start_idx = len(last_line) - ri
                if start_idx < 0:
                    start_idx = 0
                if is_space:
                    last_line += ' '
                j_line = last_line + line
                end_idx = len(last_line) + li
                j_indices = []
                for i in range(start_idx, end_idx):
                    if i - li < 0:
                        if is_space != 'sea':
                            j_lp = p * (-i) + j_line[start_idx:i]
                        else:
                            j_lp = [p] * (-i) + j_line[start_idx:i]
                    else:
                        j_lp = j_line[i - li:i]
                    if i + ri + 1 > len(j_line):
                        if is_space != 'sea':
                            j_rp = j_line[i:end_idx] + p * (ri + i + 1 - len(j_line))
                        else:
                            j_rp = j_line[i:end_idx] + [p] * (ri + i + 1 - len(j_line))
                    else:
                        j_rp = j_line[i:ri + 1 + i]
                    j_ch = j_lp + j_rp
                    if is_space == 'sea':
                        j_ch = '_'.join(j_ch)
                    if j_ch in dic:
                        j_indices.append(dic[j_ch])
                    else:
                        j_indices.append(dic['<UNK>'])
                if ri > 0:
                    out = out[: - ri] + j_indices[:ri]
                if is_space:
                    indices = j_indices[ - (li + 1):] + indices[li:]
                else:
                    indices = j_indices[ - li:] + indices[li:]
            out += indices
            indices = []
            last_line = line
        out = chop(out, dic['<#>'], limit)

    else:
        for line in raw:
            chopped = False
            while len(line) > 0:
                s_line = line[:limit - 1]
                line = line[limit - 10:]
                if len(line) < 10:
                    line = ''
                if not chopped:
                    chopped = True
                else:
                    indices.append(dic['<#>'])
                for i in range(len(s_line)):
                    if i - li < 0:
                        if is_space != 'sea':
                            lp = p * (li - i) + s_line[:i]
                        else:
                            lp = [p] * (li - i) + s_line[:i]
                    else:
                        lp = s_line[i - li:i]
                    if i + ri + 1 > len(s_line):
                        if is_space != 'sea':
                            rp = s_line[i:] + p * (i + ri + 1 - len(s_line))
                        else:
                            rp = s_line[i:] + [p] * (i + ri + 1 - len(s_line))
                    else:
                        rp = s_line[i:i + ri + 1]
                    ch = lp + rp
                    if is_space == 'sea':
                        ch = '_'.join(ch)
                    if ch in dic:
                        indices.append(dic[ch])
                    else:
                        indices.append(dic['<UNK>'])
                out.append(indices)
                indices = []
    return out


def get_gram_vec(path, fname, gram2index, lines=None, is_raw=False, limit=500, sent_seg=False, is_space=True, ignore_space=False):
    raw = []
    i = 0
    if lines is None:
        assert fname is not None
        if path is None:
            real_path = fname
        else:
            real_path = path + '/' + fname
        lines = codecs.open(real_path, 'r', encoding='utf-8')
    for line in lines:
        line = line.strip()
        if is_space == 'sea':
            line = pre_token(line)
        elif ignore_space:
            line = ''.join(line.split())
        if i == 0 or is_raw:
            raw.append(line)
            i += 1
        if len(line) > 0:
            i += 1
        else:
            i = 0
    out = []
    for g_dic in gram2index:
        out.append(gram_vec(raw, g_dic, limit, sent_seg, is_space))
    return out


def get_gram_vec_tag(path, fname, gram2index, lines=None, limit=500, is_space=True, ignore_space=False):
    raw = []
    out = [[] for _ in range(len(gram2index))]
    if lines is None:
        assert fname is not None
        if path is None:
            real_path = fname
        else:
            real_path = path + '/' + fname
        lines = codecs.open(real_path, 'r', encoding='utf-8')
    for line in lines:
        line = line.strip()
        if is_space == 'sea':
            line = pre_token(line)
        elif ignore_space:
            line = ''.join(line.split())
        if len(line) > 0:
            raw.append(line)
        else:
            for i, g_dic in enumerate(gram2index):
                out[i] += gram_vec(raw, g_dic, limit, True, is_space)
            raw = []
    if len(raw) > 0:
        for i, g_dic in enumerate(gram2index):
            out[i] += gram_vec(raw, g_dic, limit, True, is_space)
    return out


def read_vocab_tag(path):
    '''
    Read tags from index files and create dictionaries
    :param path:
    :return tag2idx, idx2tag
    '''
    tag2idx = {}
    for i, line in enumerate(codecs.open(path, 'rb', encoding='utf-8')):
        line = line.strip()
        tag2idx[line] = i
    idx2tag = {k: v for v, k in tag2idx.items()}
    return tag2idx, idx2tag


def get_tags(can, action='sep', tag_scheme='BIES'):
    tags = []
    if tag_scheme == 'BI':
        for i in range(len(can)):
            if i == 0:
                if action == 'sep':
                    tags.append('B')
                else:
                    tags.append('K')
            else:
                if action == 'sep':
                    tags.append('I')
                else:
                    tags.append('Z')
    else:
        for i in range(len(can)):
            if len(can) == 1:
                if action == 'sep':
                    tags.append('S')
                else:
                    tags.append('D')
            elif i == 0:
                if action == 'sep':
                    tags.append('B')
                else:
                    tags.append('K')
            elif i == len(can) - 1:
                if action == 'sep':
                    tags.append('E')
                else:
                    tags.append('J')
            else:
                if action == 'sep':
                    tags.append('I')
                else:
                    tags.append('Z')
    return tags


def get_gold(sent):
    line = ''
    for tk in sent:
        if '-' not in tk[0]:
            line += '  ' + tk[1]
    return line[2:]


def update_dict(trans_dic, can, trans):
    if can not in trans_dic:
        trans_dic[can] = {}
    if trans not in trans_dic[can]:
        trans_dic[can][trans] = 1
    else:
        trans_dic[can][trans] += 1
    return trans_dic


def raw2tags(raw, sents, path, train_file, creat_dict=True, gold_path=None, ignore_space=False, reset=False, tag_scheme='BIES'):
    wt = codecs.open(path + '/' + train_file, 'w', encoding='utf-8')
    if creat_dict:
        wd = codecs.open(path + '/dict.txt', 'w', encoding='utf-8')
    wg = None
    if gold_path is not None:
        wg = codecs.open(path + '/' + gold_path, 'w', encoding='utf-8')
    wtg = None
    if reset or not os.path.isfile(path + '/tags.txt'):
        wtg = codecs.open(path + '/tags.txt', 'w', encoding='utf-8')
    final_dic = {}
    assert len(raw) == len(sents)
    invalid = 0
    s_tags = set()

    def matched(can, sent_l, tags, trans_dic):
        if '-' in sent_l[0][0]:
            nums = sent_l[0][0].split('-')
            count = int(nums[1]) - int(nums[0])
            sent_l.pop(0)
            segs = []
            while count >= 0:
                segs.append(sent_l[0][1])
                sent_l.pop(0)
                count -= 1
            j_seg = ''.join(segs)
            if j_seg == can:
                for seg in segs:
                    tags += get_tags(seg, tag_scheme=tag_scheme)
            elif can.replace('-', '') == j_seg:
                for c_split in can.split('-'):
                    tags += get_tags(c_split, tag_scheme=tag_scheme)
                    if tag_scheme == 'BI':
                        tags.append('I')
                    else:
                        tags.append('X')
                tags.pop()
            else:
                tags += get_tags(can, action='trans', tag_scheme=tag_scheme)
                trans = ' '.join(segs)
                trans_dic = update_dict(trans_dic, can, trans)
        else:
            tags += get_tags(can, tag_scheme=tag_scheme)
            sent_l.pop(0)

        return tags, trans_dic

    for raw_l, sent_l in zip(raw, sents):
        if ignore_space:
            raw_l = ''.join(raw_l.split())
        tags = []
        cans = raw_l.split(' ')
        trans_dic = {}
        gold = get_gold(sent_l)
        pre = ''
        for can in cans:
            t_can = can.strip()
            purged = len(can) - len(t_can)
            if purged > 0:
                can = t_can
            while purged > 0:
                if tag_scheme == 'BI':
                    tags.append('I')
                else:
                    tags.append('X')
                purged -= 1
            done = False
            if len(pre) > 0:
                can = pre + ' ' + can
            while not done:
                if can == sent_l[0][1]:
                    tags, trans_dic = matched(can, sent_l, tags, trans_dic)
                    done = True
                    pre = ''
                else:
                    if len(can) >= len(sent_l[0][1]):
                        s_l = len(sent_l[0][1])
                        s_can = can[:s_l]
                        if s_can != sent_l[0][1]:
                            done = True
                        tags, trans_dic = matched(s_can, sent_l, tags, trans_dic)
                        can = can[s_l:]
                        if len(can) == 0:
                            done = True
                            pre = ''
                    else:
                        pre = can
                        done = True
            if len(pre) == 0:
                if tag_scheme == 'BI':
                    tags.append('I')
                else:
                    tags.append('X')
        if len(tags) > 0:
            tags.pop()
        if len(tags) == len(raw_l):
            for tg in tags:
                s_tags.add(tg)
            wt.write(raw_l + '\n')
            wt.write(''.join(tags) + '\n')
            wt.write('\n')
            for key in trans_dic:
                if key not in final_dic:
                    final_dic[key] = trans_dic[key]
                else:
                    for tr in trans_dic[key]:
                        if tr in final_dic[key]:
                            final_dic[key][tr] += trans_dic[key][tr]
                        else:
                            final_dic[key][tr] = trans_dic[key][tr]
        else:
            invalid += 1
        if wg is not None:
            wg.write(gold + '\n')
    if wg is not None:
        wg.close()
    if wtg is not None:
        for stg in s_tags:
            wtg.write(stg + '\n')
        wtg.close()
    if creat_dict:
        for key in final_dic:
            wd.write(key + '\n')
            s_dic = sorted(final_dic[key].items(), key=lambda x: x[1], reverse=True)
            for i in s_dic:
                wd.write(i[0] + '\t' + str(i[1]) + '\n')
            wd.write('\n')
    wt.close()
    print 'invalid sentences: ', invalid, len(raw)


def raw2tags_sea(raw, sents, path, train_file, gold_path=None, tag_scheme='BIES'):
    wt = codecs.open(path + '/' + train_file, 'w', encoding='utf-8')
    wg = None
    if gold_path is not None:
        wg = codecs.open(path + '/' + gold_path, 'w', encoding='utf-8')
    assert len(raw) == len(sents)
    invalid = 0
    wtg = None
    if not os.path.isfile(path + '/tags.txt'):
        wtg = codecs.open(path + '/tags.txt', 'w', encoding='utf-8')

    s_tags = set()

    def matched(can, sent_l, tags):
        segs = can.split(' ')
        sent_l.pop(0)
        if len(segs) == 1:
            tags.append('S')
        elif len(segs) > 1:
            if tag_scheme == 'BI':
                tags += ['B'] + ['I'] * (len(segs) - 1)
            else:
                mid_t = ['I'] * (len(segs) - 2)
                tags += ['B'] + mid_t + ['E']
        return tags

    for raw_l, sent_l in zip(raw, sents):
        tags = []
        cans = pre_token(raw_l)
        gold = get_gold(sent_l)
        pre = ''
        for can in cans:
            t_can = can.strip()
            purged = len(can) - len(t_can)
            if purged > 0:
                can = t_can
            while purged > 0:
                if tag_scheme == 'BI':
                    tags.append('I')
                else:
                    tags.append('X')
                purged -= 1
            if len(pre) > 0:
                can = pre + ' ' + can
            j_can = ''.join(can.split())
            if sent_l:
                j_sent = ''.join(sent_l[0][1].split())
            if j_can == j_sent:
                tags = matched(can, sent_l, tags)
                pre = ''
            else:
                assert len(j_can) < len(j_sent)
                pre = can
        if len(tags) == len(cans):
            for tg in tags:
                s_tags.add(tg)
            wt.write(raw_l + '\n')
            wt.write(''.join(tags) + '\n')
            wt.write('\n')
        else:
            invalid += 1
        if wg is not None:
            wg.write(gold + '\n')
    if wg is not None:
        wg.close()
    if wtg is not None:
        for stg in s_tags:
            wtg.write(stg + '\n')
        wtg.close()
    wt.close()

    print 'invalid sentences: ', invalid, len(raw)


def pad_zeros(l, max_len):
    padded = None
    if type(l) is list:
        padded = []
        for item in l:
            if len(item) <= max_len:
                padded.append(np.pad(item, (0, max_len - len(item)), 'constant', constant_values=0))
            else:
                padded.append(np.asarray(item[:max_len]))
        padded = np.asarray(padded)
    elif type(l) is dict:
        padded = {}
        for k, v in l.iteritems():
            padded[k] = [np.pad(item, (0, max_len - len(item)), 'constant', constant_values=0) for item in v]
    return padded

def unpad_zeros(l):
    out = []
    for tags in l:
        out.append([np.trim_zeros(line) for line in tags])
    return out


def buckets(x, y, size=50):
    assert len(x[0]) == len(y[0])
    num_inputs = len(x)
    samples = x + y
    num_items = len(samples)
    xy = zip(*samples)
    xy.sort(key=lambda i: len(i[0]))
    t_len = size
    idx = 0
    bucks = [[[]] for _ in range(num_items)]
    for item in xy:
        if len(item[0]) > t_len:
            if len(bucks[0][idx]) > 0:
                for buck in bucks:
                    buck.append([])
                idx += 1
            while len(item[0]) > t_len:
                t_len += size
        for i in range(num_items):
            #print item[i]
            bucks[i][idx].append(item[i])

    return bucks[:num_inputs], bucks[num_inputs:]


def pad_bucket(x, y, limit, bucket_len_c=None):
    assert len(x[0]) == len(y[0])
    num_inputs = len(x)
    num_tags = len(y)
    padded = [[] for _ in range(num_tags + num_inputs)]
    bucket_counts = []
    samples = x + y
    xy = zip(*samples)
    if bucket_len_c is None:
        bucket_len_c = []
        for i, item in enumerate(xy):
            max_len = len(item[0][-1])
            if i == len(xy) - 1:
                max_len = limit
            bucket_len_c.append(max_len)
            bucket_counts.append(len(item[0]))
            for idx in range(num_tags + num_inputs):
                padded[idx].append(pad_zeros(item[idx], max_len))
        print 'Number of buckets: ', len(bucket_len_c)
    else:
        idy = 0
        for item in xy:
            max_len = len(item[0][-1])
            while idy < len(bucket_len_c) and max_len > bucket_len_c[idy]:
                idy += 1
            bucket_counts.append(len(item[0]))
            if idy >= len(bucket_len_c):
                for idx in range(num_tags + num_inputs):
                    padded[idx].append(pad_zeros(item[idx], max_len))
                bucket_len_c.append(max_len)
            else:
                for idx in range(num_tags + num_inputs):
                    padded[idx].append(pad_zeros(item[idx], bucket_len_c[idy]))
    return padded[:num_inputs], padded[num_inputs:], bucket_len_c, bucket_counts


def get_real_batch(counts, b_size):
    real_batch_sizes = []
    for c in counts:
        if c < b_size:
            real_batch_sizes.append(c)
        else:
            real_batch_sizes.append(b_size)
    return real_batch_sizes


def merge_bucket(x):
    out = []
    for item in x:
        m = []
        for i in item:
            m += i
        out.append(m)
    return out


def decode_tags(idx, index2tags):
    out = []
    for id in idx:
        sents = []
        for line in id:
            sent = []
            for item in line:
                tag = index2tags[item]
                tag = tag.replace('E', 'I')
                tag = tag.replace('S', 'B')
                tag = tag.replace('J', 'Z')
                tag = tag.replace('D', 'K')
                sent.append(tag)
            sents.append(sent)
        out.append(sents)
    return out


def decode_chars(idx, idx2chars):
    out = []
    for line in idx:
        line = np.trim_zeros(line)
        out.append([idx2chars[item] for item in line])
    return out


def generate_output(chars, tags, trans_dict, transducer_dict=None, multi_tok=False):
    out = []
    mult_out = []
    raw_out = []
    sent_seg = False

    def map_trans(c_trans):
        if c_trans in trans_dict:
            c_trans = trans_dict[c_trans]
        elif c_trans.lower() in trans_dict:
            c_trans = trans_dict[c_trans.lower()]
        elif transducer_dict is not None:
            c_trans = transducer_dict(c_trans)
        c_trans = c_trans.replace('    ', '  ')
        c_trans = c_trans.replace('   ', '  ')
        return c_trans

    def add_pline(p_line, mt_p_line, c_trans, multi_tok, trans=False):
        c_trans = c_trans.strip()
        if len(c_trans) > 0:
            if trans:
                o_trans = c_trans
                c_trans = map_trans(c_trans)
                if multi_tok:
                    num_tr = len(c_trans.split('  '))
                    mt_p_line += '  ' + o_trans + '!#!' + str(num_tr) + '  ' + c_trans
            else:
                if multi_tok:
                    mt_p_line += '  ' + c_trans
            p_line += '  ' + c_trans
        return p_line, mt_p_line

    def split_sent(lines, s_str):
        for i in range(len(lines)):
            s_line = lines[i].strip()
            while s_line and s_line[-1] == s_str:
                s_line = s_line[:-1]
            sents = s_line.split(s_str)
            lines[i] = [sent.strip() for sent in sents]
        return lines

    for i, tag in enumerate(tags):
        assert len(chars) == len(tag)
        sub_out = []
        sub_raw_out = []
        multi_sub_out = []
        j_chars = []
        j_tags = []
        is_first = True
        for chs, tgs in zip(chars, tag):
            if chs[0] == '<#>':
                assert len(j_chars) > 0
                if is_first:
                    is_first = False
                    j_chars[-1] = j_chars[-1][:-5] + chs[6:]
                    j_tags[-1] = j_tags[-1][:-5] + tgs[6:]
                else:
                    j_chars[-1] = j_chars[-1][:-5] + chs[5:]
                    j_tags[-1] = j_tags[-1][:-5] + tgs[5:]
            else:
                j_chars.append(chs)
                j_tags.append(tgs)
                is_first = True
        chars = j_chars
        tag = j_tags
        for chs, tgs in zip(chars, tag):
            assert len(chs) == len(tgs)
            c_word = ''
            c_trans = ''
            p_line = ''
            r_line = ''
            mt_p_line = ''
            for ch, tg in zip(chs, tgs):
                r_line += ch
                if tg == 'I':
                    if len(c_trans) > 0:
                        p_line, mt_p_line = add_pline(p_line, mt_p_line, c_trans, multi_tok, trans=True)
                        c_trans = ''
                        c_word = ch
                    else:
                        c_word += ch
                elif tg == 'Z':
                    if len(c_word) > 0:
                        p_line, mt_p_line = add_pline(p_line, mt_p_line, c_word, multi_tok)
                        c_word = ''
                        c_trans = ch
                    else:
                        c_trans += ch
                elif tg == 'B':
                    if len(c_word) > 0:
                        c_word = c_word.strip()
                        p_line, mt_p_line = add_pline(p_line, mt_p_line, c_word, multi_tok)
                    elif len(c_trans) > 0:
                        c_trans = c_trans.strip()
                        p_line, mt_p_line = add_pline(p_line, mt_p_line, c_trans, multi_tok, trans=True)
                        c_trans = ''
                    c_word = ch
                elif tg == 'K':
                    if len(c_word) > 0:
                        p_line, mt_p_line = add_pline(p_line, mt_p_line, c_word, multi_tok)
                        c_word = ''
                    elif len(c_trans) > 0:
                        p_line, mt_p_line = add_pline(p_line, mt_p_line, c_trans, multi_tok, trans=True)
                    c_trans = ch
                elif tg == 'T':
                    sent_seg = True
                    if len(c_word) > 0:
                        p_line, mt_p_line = add_pline(p_line, mt_p_line, c_word, multi_tok)
                        c_word = ''
                    elif len(c_trans) > 0:
                        p_line, mt_p_line = add_pline(p_line, mt_p_line, c_trans, multi_tok, trans=True)
                        c_trans = ''
                    p_line += '  ' + ch + '<SENT>'
                    if multi_tok:
                        mt_p_line += '  ' + ch + '<SENT>'
                    r_line += '<SENT>'
                elif tg == 'U':
                    sent_seg = True
                    if len(c_word) > 0:
                        c_word += ch
                        p_line, mt_p_line = add_pline(p_line, mt_p_line, c_word, multi_tok)
                        c_word = ''
                    elif len(c_trans) > 0:
                        c_trans += ch
                        p_line, mt_p_line = add_pline(p_line, mt_p_line, c_trans, multi_tok, trans=True)
                        c_trans = ''
                    elif len(ch.strip()) > 0:
                        p_line += ch
                        if multi_tok:
                            mt_p_line += ch
                    p_line += '<SENT>'
                    if multi_tok:
                        mt_p_line += '<SENT>'
                    r_line += '<SENT>'
                elif tg == 'X' and len(ch.strip()) > 0:
                    if len(c_word) > 0:
                        c_word += ch
                    elif len(c_trans) > 0:
                        c_trans += ch
                    else:
                        c_word = ch
                elif len(ch.strip()) > 0:
                    if len(c_word) > 0:
                        c_word += '  ' + ch
                    elif len(c_trans) > 0:
                        c_trans += '  ' + ch
                    else:
                        c_word = ch
            if len(c_word) > 0:
                c_word = c_word.strip()
                p_line, mt_p_line = add_pline(p_line, mt_p_line, c_word, multi_tok)
            elif len(c_trans) > 0:
                c_trans = c_trans.strip()
                p_line, mt_p_line = add_pline(p_line, mt_p_line, c_trans, multi_tok, trans=True)
            sub_out.append(p_line.strip())
            sub_raw_out.append(r_line.strip())
            if multi_tok:
                multi_sub_out.append(mt_p_line.strip())
        out.append(sub_out)
        raw_out.append(sub_raw_out)
        if multi_tok:
            mult_out.append(multi_sub_out)
    out[0][-1].rstrip('<SENT>')
    raw_out[0][-1].rstrip('<SENT>')
    if sent_seg:
        out = split_sent(out[0], '<SENT>')
        raw_out = split_sent(raw_out[0], '<SENT>')
    if multi_tok:
        mult_out[0][-1].rstrip('<SENT>')
        if sent_seg:
            mult_out = split_sent(mult_out[0], '<SENT>')
        return out, raw_out, mult_out
    else:
        return out, raw_out


def generate_output_sea(chars, tags):
    out = []
    raw_out = []
    sent_seg = False

    def split_sent(lines, s_str):
        for i in range(len(lines)):
            s_line = lines[i].strip()
            while s_line and s_line[-1] == s_str:
                s_line = s_line[:-1]
            sents = s_line.split(s_str)
            lines[i] = [sent.strip() for sent in sents]
        return lines

    for i, tag in enumerate(tags):
        assert len(chars) == len(tag)
        sub_out = []
        sub_raw_out = []
        j_chars = []
        j_tags = []
        is_first = True
        for chs, tgs in zip(chars, tag):
            if chs[0] == '<#>':
                assert len(j_chars) > 0
                if is_first:
                    is_first = False
                    j_chars[-1] = j_chars[-1][:-5] + chs[6:]
                    j_tags[-1] = j_tags[-1][:-5] + tgs[6:]
                else:
                    j_chars[-1] = j_chars[-1][:-5] + chs[5:]
                    j_tags[-1] = j_tags[-1][:-5] + tgs[5:]
            else:
                j_chars.append(chs)
                j_tags.append(tgs)
                is_first = True
        chars = j_chars
        tag = j_tags
        for chs, tgs in zip(chars, tag):
            assert len(chs) == len(tgs)
            p_line = ''
            r_line = ''
            for ch, tg in zip(chs, tgs):
                r_line += ' ' + ch
                if tg == 'I':
                    if ch == '.' or (ch >= '0' and ch <= '9'):
                        p_line += ch
                    else:
                        p_line += ' ' + ch
                elif tg == 'B':
                    p_line += '  ' + ch
                elif tg == 'T':
                    sent_seg = True
                    p_line += '  ' + ch + '<SENT>'
                    r_line += '<SENT>'
                elif tg == 'U':
                    sent_seg = True
                    p_line += ch + '<SENT>'
                    r_line += '<SENT>'
                elif len(ch.strip()) > 0:
                    p_line += '  ' + ch
            sub_out.append(p_line.strip())
            sub_raw_out.append(r_line.strip())
        out.append(sub_out)
        raw_out.append(sub_raw_out)
    out[0][-1].rstrip('<SENT>')
    raw_out[0][-1].rstrip('<SENT>')
    if sent_seg:
        out = split_sent(out[0], '<SENT>')
        raw_out = split_sent(raw_out[0], '<SENT>')
    return out, raw_out


def trim_output(out, length):
    assert len(out) == len(length)
    trimmed_out = []
    for item, l in zip(out, length):
        trimmed_out.append(item[:l])
    return trimmed_out


def generate_trans_out(x, idx2char):
    out = ''
    for i in x:
        if i == 3:
            out += ' '
        elif i in idx2char:
            out += idx2char[i]
    if '<#>' in out:
        out = out[:out.index('<#>')]
    out = out.replace('   ', ' ')
    out = out.replace('  ', ' ')
    return out


def generate_sent_out(raw, predictions):
    out = []
    line = ''
    assert len(raw) == len(predictions)
    for ch, tag in zip(raw, predictions):
        line += ch
        if tag == 1:
            line = line.strip()
            out.append(line)
            line = ''
    if len(line) > 0:
        line = line.strip()
        out.append(line)
    return out


def viterbi(max_scores, max_scores_pre, length, batch_size):
    best_paths = []
    for m in range(batch_size):
        path = []
        last_max_node = np.argmax(max_scores[m][length[m] - 1])
        path.append(last_max_node)
        for t in range(1, length[m])[::-1]:
            last_max_node = max_scores_pre[m][t][last_max_node]
            path.append(last_max_node)
        path = path[::-1]
        best_paths.append(path)
    return best_paths


def get_new_chars(path, char2idx, is_space):
    new_chars = set()
    for line in codecs.open(path, 'rb', encoding='utf-8'):
        line = line.strip()
        if is_space == 'sea':
            line = pre_token(line)
        for ch in line:
            if ch not in char2idx:
                new_chars.add(ch)
    return new_chars


def get_valid_chars(chars, emb_path):
    valid_chars = []
    total = []
    for line in codecs.open(emb_path, 'rb', encoding='utf-8'):
        line = line.strip()
        sets = line.split(' ')
        total.append(sets[0])
    for ch in chars:
        if ch in total:
            valid_chars.append(ch)
    return valid_chars


def get_new_embeddings(new_chars, emb_dim, emb_path):
    assert os.path.isfile(emb_path)
    emb = {}
    new_emb = []
    for line in codecs.open(emb_path, 'rb', encoding='utf-8'):
        line = line.strip()
        sets = line.split(' ')
        emb[sets[0]] = np.asarray(sets[1:], dtype='float32')
    if '<UNK>' not in emb:
        unk = np.random.uniform(-math.sqrt(float(3) / emb_dim), math.sqrt(float(3) / emb_dim), emb_dim)
        emb['<UNK>'] = np.asarray(unk, dtype='float32')
    for ch in new_chars:
        if ch in emb:
            new_emb.append(emb[ch])
        else:
            new_emb.append(emb['<UNK>'])
    return new_emb


def update_char_dict(char2idx, new_chars, unk_chars, valid_chars=None):
    dim = len(char2idx) + 10
    if valid_chars is not None:
        for ch in valid_chars:
            if ch in unk_chars:
                unk_chars.remove(ch)
    for char in new_chars:
        if char not in char2idx and len(char.strip()) > 0:
            char2idx[char] = dim
            if valid_chars is None or char not in valid_chars:
                unk_chars.append(dim)
            dim += 1
    idx2char = {k: v for v, k in char2idx.items()}
    return char2idx, idx2char, unk_chars


def get_new_grams(path, gram2idx, is_raw=False, is_space=True):
    raw = []
    i = 0
    for line in codecs.open(path, 'rb', encoding='utf-8'):
        line = line.strip()
        if is_space == 'sea':
            line = pre_token(line)
        if i == 0 or is_raw:
            raw.append(line)
            i += 1
        if len(line) > 0:
            i += 1
        else:
            i = 0
    new_grams = []
    for g_dic in gram2idx:
        new_g = []
        if is_space == 'sea':
            n = len(g_dic.keys()[0].split('_'))
        else:
            n = 0
            for k in g_dic.keys():
                if '<PAD>' not in k:
                    n = len(k)
                    break
        grams = ngrams(raw, n, is_space)
        for g in grams:
            if g not in g_dic:
                new_g.append(g)
        new_grams.append(new_g)
    return new_grams


def printer(raw, tagged, multi_out, outpath, sent_seg, form='conll'):
    assert len(tagged) == len(multi_out)
    validator(raw, multi_out)
    wt = codecs.open(outpath, 'w', encoding='utf-8')
    if form == 'conll':
        if not sent_seg:
            for raw_t, tagged_t, multi_t in zip(raw, tagged, multi_out):
                if len(multi_t) > 0:
                    wt.write('#sent_raw: ' + raw_t + '\n')
                    wt.write('#sent_tok: ' + tagged_t + '\n')
                    idx = 1
                    tgs = multi_t.split('  ')
                    pl = ''
                    for _ in range(8):
                        pl += '\t' + '_'
                    for tg in tgs:
                        if '!#!' in tg:
                            segs = tg.split('!#!')
                            wt.write(str(idx) + '-' + str(int(segs[1]) + idx - 1) + '\t' + segs[0] + pl + '\n')
                        else:
                            wt.write(str(idx) + '\t' + tg + pl + '\n')
                            idx += 1
                    wt.write('\n')
        else:
            for tagged_t, multi_t in zip(tagged, multi_out):
                if len(tagged_t.strip()) > 0:
                    wt.write('#sent_tok: '+ tagged_t + '\n')
                    idx = 1
                    tgs = multi_t.split('  ')
                    pl = ''
                    for _ in range(8):
                        pl += '\t' + '_'
                    for tg in tgs:
                        if '!#!' in tg:
                            segs = tg.split('!#!')
                            wt.write(str(idx) + '-' + str(int(segs[1]) + idx - 1) + '\t' + segs[0] + pl + '\n')
                        else:
                            wt.write(str(idx) + '\t' + tg + pl + '\n')
                            idx += 1
                    wt.write('\n')
    else:
        for tg in tagged:
            wt.write(tg + '\n')
    wt.close()


def biased_out(prediction, bias):
    out = []
    b_pres = []
    for pre in prediction:
        b_pres.append(pre[:,0] - pre[:,1])
    props = np.concatenate(b_pres)
    props = np.sort(props)[::-1]
    idx = int(bias*len(props))
    if idx == len(props):
        idx -= 1
    th = props[idx]
    print 'threshold: ', th, 1 / (1 + np.exp(-th))
    for pre in b_pres:
        pre[pre >= th] = 0
        pre[pre != 0] = 1
        out.append(pre)
    return out


def to_one_hot(y, nb_classes=None):
    '''Convert class vector (integers from 0 to nb_classes) to binary class matrix, for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
        nb_classes: total number of classes
    # Returns
        A binary matrix representation of the input.
    '''
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y


def validator(raw, generated):
    raw_l = ''.join(raw)
    raw_l = ''.join(raw_l.split())
    for g in generated:
        g_tokens = g.split('  ')
        j = 0
        while j < len(g_tokens):
            if '!#!' in g_tokens[j]:
                segs = g_tokens[j].split('!#!')
                c_t = int(segs[1])
                r_seg = ''.join(segs[0].split())
                l_w = len(r_seg)
                if r_seg == raw_l[:l_w]:
                    raw_l = raw_l[l_w:]
                    raw_l = raw_l.strip()
                else:
                    raise Exception('Error: unmatch...')
                j += c_t
            else:
                r_seg = ''.join(g_tokens[j].split())
                l_w = len(r_seg)
                if r_seg == raw_l[:l_w]:
                    raw_l = raw_l[l_w:]
                    raw_l = raw_l.strip()
                else:
                    print r_seg
                    print raw_l[:l_w]
                    print ''
                    raise Exception('Error: unmatch...')
            j += 1


def mlp_post(raw, prediction, is_space=False, form='mlp1'):
    assert len(raw) == len(prediction)
    out = []
    for r_l, p_l in zip(raw, prediction):
        st = ''
        rtokens = r_l.split()
        ptokens = p_l.split('  ')
        purged = []
        for pt in ptokens:
            purged.append(pt.strip())
        ptokens = purged
        ptokens_str = ''.join(ptokens)
        assert ''.join(rtokens) == ''.join(ptokens_str.split())
        if form == 'mlp1':
            if is_space == 'sea':
                for p_t in ptokens:
                    st += p_t.replace(' ', '_') + ' '
            else:
                while rtokens and ptokens:
                    if rtokens[0] == ptokens[0]:
                        st += ptokens[0] + ' '
                        rtokens.pop(0)
                        ptokens.pop(0)
                    else:
                        if len(rtokens[0]) <= len(ptokens[0]):
                            assert ptokens[0][:len(rtokens[0])] == rtokens[0]
                            st += rtokens[0] + ' '
                            ptokens[0] = ptokens[0][len(rtokens[0]):].strip()
                            rtokens.pop(0)
                        else:
                            can = ''
                            while can != rtokens[0] and ptokens:
                                can += ptokens[0]
                                st += ptokens[0] + '\\\\'
                                ptokens.pop(0)
                            st = st[:-2] + ' '
                            rtokens.pop(0)
        else:
            for p_t in ptokens:
                st += p_t + ' '
        out.append(st.strip())
    return out