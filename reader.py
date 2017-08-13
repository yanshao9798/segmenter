# -*- coding: utf-8 -*-
"""
@author: Yan Shao, yan.shao@lingfil.uu.se
"""
import codecs


def gold(path, is_dev=True, form='conll', is_space=False):
    sents = []
    sent = []
    cter = 0
    sents_dev = None
    if not is_dev:
        sents_dev = []
    for line in codecs.open(path, 'rb', encoding='utf8'):
        line = line.strip()
        if form == 'conll':
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
        elif form == 'mlp1' or form == 'mlp2':
            if len(line) > 0:
                if form == 'mlp1':
                    segs = []
                    for l_seg in line.split(' '):
                        if len(l_seg) > 0:
                            if is_space == 'sea':
                                segs.append(l_seg.replace('_', ' '))
                            else:
                                segs += l_seg.split('\\\\')
                else:
                    segs = line.split()
                for i, seg in enumerate(segs):
                    sent.append((str(i + 1), seg))
                if not is_dev and cter == 9:
                    sents_dev.append(sent)
                    cter = 0
                else:
                    sents.append(sent)
                    cter += 1
                sent = []
        else:
            raise Exception('Format error, available: conll, mlp1, mlp2')
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


def test_gold(path, form='conll', is_space=False):
    sents = []
    st = ''
    for line in codecs.open(path, 'rb', encoding='utf-8'):
        line = line.strip()
        if form == 'conll':
            segs = line.split('\t')
            if len(segs) == 10:
                if '.' not in segs[0] and '-' not in segs[0]:
                    st += '  ' + segs[1]
            elif len(st) > 0:
                sents.append(st.strip())
                st = ''
        elif form == 'mlp1' or form == 'mlp2':
            if len(line) > 0:
                if form == 'mlp1':
                    segs = []
                    for l_seg in line.split(' '):
                        if is_space == 'sea':
                            segs.append(l_seg.replace('_', ' '))
                        else:
                            segs += l_seg.split('\\\\')
                else:
                    segs = line.split()
                for seg in segs:
                    st += '  ' + seg
                sents.append(st.strip())
                st = ''
        else:
            raise Exception('Format error, available: conll, mlp1, mlp2')
    return sents


def get_raw(path, fin, fout, cat='other', new=True, is_dev=True, form='conll', is_space=False):
    fout = codecs.open(path + '/' + fout, 'w', encoding='utf-8')
    fout_dev = None
    if not is_dev:
        fout_dev = codecs.open(path + '/raw_dev.txt', 'w', encoding='utf-8')
    cter = 0
    if form == 'conll':
        if cat == 'gold':
            for line in codecs.open(path + '/' + fin, 'r', encoding='utf-8'):
                line = line.strip()
                line = line.replace('&apos', '\'')
                if len(line) > 0 and ('# sentence' in line or '# text' in line):
                    if new:
                        if not is_dev and cter == 9:
                            fout_dev.write(line[line.index('=') + 1:].lstrip() + '\n')
                            cter = 0
                        else:
                            fout.write(line[line.index('=') + 1:].lstrip() + '\n')
                            cter += 1
                    else:
                        if not is_dev and cter == 9:
                            fout_dev.write(line[line.index(':') + 1:].lstrip() + '\n')
                            cter = 0
                        else:
                            fout.write(line[line.index(':') + 1:].lstrip() + '\n')
                            cter += 1

        elif cat == 'zh':
            pt = ''
            for line in codecs.open(path + '/' + fin, 'r', encoding='utf-8'):
                line = line.strip()
                line = line.split('\t')
                if len(line) == 10:
                    pt += line[1]
                else:
                    if len(pt) > 0:
                        if not is_dev and cter == 9:
                            fout_dev.write(pt + '\n')
                            cter = 0
                        else:
                            fout.write(pt + '\n')
                            cter += 1
                        pt = ''

        else:
            punc_e = ['!', ')', ',', '.', ';', ':', '?', '»', '...', ']', '..', '....', '%', 'º', '²', '°']
            punc_b = ['¿', '¡', '(', '«', '[']
            punc_m = ['"', '\'']
            punc_e = [s.decode('utf-8') for s in punc_e]
            punc_b = [s.decode('utf-8') for s in punc_b]
            punc_m = [s.decode('utf-8') for s in punc_m]
            md = {}
            for p in punc_m:
                md[p] = True
            pt = ''
            ct = 0
            for line in codecs.open(path + '/' + fin, 'r', encoding='utf-8'):
                line = line.strip()
                segs = line.split('\t')
                if len(segs) == 10:
                    if '-' in segs[0]:
                        sp = segs[0].split('-')
                        ct = int(sp[1]) - int(sp[0]) + 1
                        if len(pt) > 0 and pt[-1] in punc_b:
                            pt += segs[1]
                        elif len(pt) > 0 and pt[-1] in punc_m:
                            if md[pt[-1]]:
                                pt += ' ' + segs[1]
                            else:
                                pt += segs[1]
                        else:
                            pt += ' ' + segs[1]
                    elif ct == 0:
                        if segs[1] in punc_e:
                            pt += segs[1]
                        elif len(pt) > 0 and pt[-1] in punc_b:
                            pt += segs[1]
                            if segs[1] in punc_m:
                                if md[segs[1]]:
                                    md[segs[1]] = False
                                else:
                                    md[segs[1]] = True
                        elif segs[1] in punc_m:
                            if md[segs[1]]:
                                pt += ' ' + segs[1]
                                md[segs[1]] = False
                            else:
                                pt += segs[1]
                                md[segs[1]] = True
                        elif len(pt) > 0 and pt[-1] in punc_m:
                            if md[pt[-1]]:
                                pt += ' ' + segs[1]
                            else:
                                pt += segs[1]
                        elif segs[1][0] == '\'':
                            pt += segs[1]
                        else:
                            pt += ' ' + segs[1]
                    else:
                        ct -= 1
                else:
                    if len(pt) > 0:
                        pt = pt.lstrip()
                        pt = pt.replace(' ",', '",')
                        pt = pt.replace(' ".', '".')
                        pt = pt.replace(':"...', ': "...')
                        pt = pt.replace(' n\'t', 'n\'t')
                        pt = pt.replace(' - ', '-')
                        pt = pt.replace(' -- ', '--')
                        pt = pt.replace(' / ', '/')
                        if not is_dev and cter == 9:
                            fout_dev.write(pt + '\n')
                            cter = 0
                        else:
                            fout.write(pt + '\n')
                            cter += 1
                        pt = ''
                        for p in punc_m:
                            md[p] = True

    elif form == 'mlp1' or form == 'mlp2':
        for line in codecs.open(path + '/' + fin, 'r', encoding='utf-8'):
            line = line.strip()
            if len(line) > 0:
                if form == 'mlp1':
                    if is_space == 'sea':
                        line = line.replace('_', ' ')
                    else:
                        line = line.replace('\\\\', '')
                else:
                    line = ''.join(line.split())
                if not is_dev and cter == 9:
                    fout_dev.write(line + '\n')
                    cter = 0
                else:
                    fout.write(line + '\n')
                    cter += 1
    else:
        raise Exception('Format error, available: conll, mlp1, mlp2')
    fout.close()
    if not is_dev:
        fout_dev.close()
