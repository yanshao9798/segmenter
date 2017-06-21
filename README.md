# Segmenter
Universal segmenter, written by Y. Shao, Uppsala University

## Training

### Segmentation:

python segmenter.py train -p ud-treebanks-conll2017/UD_English -gru -cp -m seg_Eng

### Joint sentence segmentation:

python segmenter.py train -p ud-treebanks-conll2017/UD_English -ss -gru -cp -m ss_seg_Eng

## Decoding

python segmenter.py tag -p ud-treebanks-conll2017/UD_English -r ud-raw/en_pud.txt -opth tokenized/en_pud.txt
