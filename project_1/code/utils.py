import os

from collections import Counter

STUDENT = {'name': 'Simon Raviv'}
MODULE_ROOT_DIR = os.path.dirname(__file__)

def read_data(fname):
    data = []
    with open(fname, 'r', encoding="utf8") as file:
        for line in file.readlines():
            label, text = line.strip().lower().split("\t", 1)
            data.append((label, text))
    return data


def text_to_bigrams(text):
    return ["%s%s" % (c1, c2) for c1, c2 in zip(text, text[1:])]


def text_to_unigrams(text):
    return ["%s" % (c1) for c1 in text]


train_filename = os.path.join(MODULE_ROOT_DIR, "..", "data", "train")
dev_filename = os.path.join(MODULE_ROOT_DIR, "..", "data", "dev")
test_filename = os.path.join(MODULE_ROOT_DIR, "..", "data", "test")

TRAIN = [(l, text_to_bigrams(t)) for l, t in read_data(train_filename)]
DEV = [(l, text_to_bigrams(t)) for l, t in read_data(dev_filename)]
TEST = [(l, text_to_bigrams(t)) for l, t in read_data(test_filename)]

# TRAIN = [(l, text_to_unigrams(t)) for l, t in read_data("train")]
# DEV = [(l, text_to_unigrams(t)) for l, t in read_data("dev")]


fc = Counter()
for l, feats in TRAIN:
    fc.update(feats)

# 600 most common bigrams in the training set.
vocab = set([x for x, c in fc.most_common(600)])

# label strings to IDs
L2I = {l: i for i, l in enumerate(list(sorted(set([l for l, t in TRAIN]))))}
I2L = {i: l for l, i in L2I.items()}

# feature strings (bigrams) to IDs
F2I = {f: i for i, f in enumerate(list(sorted(vocab)))}
