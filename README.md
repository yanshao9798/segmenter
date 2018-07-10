## Segmenter
Universal segmenter, written by Y. Shao, Uppsala University

### News

We published a TACL paper on more information and analysis of the segmenter. (https://transacl.org/ojs/index.php/tacl/article/viewFile/1446/315)

The segmenter achieves the best overall word segmentation accuracy and second best overall sentence segmentaton accuracy at the ConLL2018 shared task. (http://universaldependencies.org/conll18/results-words.html)

The segmenter is applied to the MLP 2017 shared tasks (http://mlp.computing.dcu.ie/mlp2017_Shared_Task.html) and achieved outstanding results on all the datasets. (2017.8.13)

## Universal Dependencies

### Training

#### Segmentation:

python segmenter.py train -p ud-treebanks-conll2017/UD_English -m seg_Eng

#### Joint sentence segmentation:

python segmenter.py train -p ud-treebanks-conll2017/UD_English -ss -m ss_seg_Eng

### Decoding

python segmenter.py tag -p ud-treebanks-conll2017/UD_English -m ss_seg_Eng -r ud-raw/en_pud.txt -opth tokenized/en_pud.txt

## MLP 2017

### (Single)

### Training

#### For Basque Finnish Kazakh Marathi Uyghur and Farsi

python segmenter.py train -p mlp/basque -f mlp1 -ng 3 -m basque

(The training and development sets of Basque are in directory mlp/basque)

#### For Vietnamese

python segmenter.py train -p mlp/basque -f mlp1 -ng 3 -sea

#### For Chinese and Japanese

python segmenter.py train -p mlp/tchinese -f mlp2 -ng 3 -m tchinese

### Decoding

#### For Basque Finnish Kazakh Marathi Uyghur Farsi and Vietnamese

python segmenter.py tag -p mlp/basque -f mlp1 -m basque -r testset/basque_raw.txt -opth segmented_mlp/basque_single_out.txt

#### For Chinese and Japanese

python segmenter.py tag -p mlp/tchinese -f mlp2 -m tchinese -r testset/tchinese_raw.txt -opth segmented_mlp/tchinese_single_out.txt

### (Ensemble)

### Training

python segmenter.py train -p mlp/basque -f mlp1 -ng 3 -m basque_1

python segmenter.py train -p mlp/basque -f mlp1 -ng 3 -m basque_2

python segmenter.py train -p mlp/basque -f mlp1 -ng 3 -m basque_3

python segmenter.py train -p mlp/basque -f mlp1 -ng 3 -m basque_4

### Decoding

python segmenter.py tag -ens -p mlp/basque -f mlp1 -m basque -r testset/basque_raw.txt -opth segmented_mlp/basque_ensemble_out.txt

## Reference

Yan Shao, Christian Hardmeier, and Joakim Nivre. 2018. Universal word segmentation: Implementation and interpretation. Transactions of the Association for Computational Linguistics 6:421–435.

https://transacl.org/ojs/index.php/tacl/article/viewFile/1446/315


Yan Shao. "Cross-lingual Word Segmentation and Morpheme Segmentation as Sequence Labelling" arXiv preprint arXiv:1709.03756 (2017).

https://arxiv.org/pdf/1709.03756.pdf

