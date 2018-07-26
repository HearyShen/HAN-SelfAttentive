import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--review-path", default="review.json")
parser.add_argument("--review-max", type=int, default=100000)
args = parser.parse_args()

import os
import ujson as json
import spacy
import pickle
import random
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from yelp import *

en = spacy.load('en')
en.pipeline = [('tagger', en.tagger), ('parser', en.parser)]


def read_reviews():
    reviewCount = 0
    with open(args.review_path, 'rb') as f:
        for line in f:
            reviewCount += 1
            if reviewCount > args.review_max:
                f.close()
                return
            yield json.loads(line)


def build_word_frequency_distribution():
    path = os.path.join(data_dir, 'word_freq.pickle')

    try:
        with open(path, 'rb') as freq_dist_f:
            freq_dist_f = pickle.load(freq_dist_f)
            print('frequency distribution loaded')
            return freq_dist_f
    except IOError:
        pass

    print('building frequency distribution')
    freq = defaultdict(int)
    for i, review in enumerate(read_reviews()):
        doc = en.tokenizer(review['text'])
        for token in doc:
            freq[token.orth_] += 1
        count = i + 1  # count starts from 1 (i starts from 0)
        if count % 10000 == 0:
            with open(path, 'wb') as freq_dist_f:
                pickle.dump(freq, freq_dist_f)
            print('dump at {}'.format(count))
    return freq


def build_vocabulary(lower=3, n=50000):
    try:
        with open(vocab_fn, 'rb') as vocab_file:
            vocab = pickle.load(vocab_file)
            print('vocabulary loaded')
            return vocab
    except IOError:
        print('building vocabulary')
    freq = build_word_frequency_distribution()
    top_words = list(sorted(freq.items(), key=lambda x: -x[1]))[:n - lower + 1]
    vocab = {}
    i = lower
    for w, freq in top_words:
        vocab[w] = i
        i += 1
    with open(vocab_fn, 'wb') as vocab_file:
        pickle.dump(vocab, vocab_file)
    return vocab


UNKNOWN = 2


def make_data(split_points=(0.95, 0.98)):
    train_ratio, dev_ratio = split_points
    vocab = build_vocabulary()
    train_f = open(trainset_fn, 'wb')
    dev_f = open(devset_fn, 'wb')
    test_f = open(testset_fn, 'wb')

    try:
        for review in tqdm(read_reviews()):
            x = []
            for sent in en(review['text']).sents:
                x.append([vocab.get(tok.orth_, UNKNOWN) for tok in sent])
            y = review['stars']

            r = random.random()
            if r < train_ratio:
                f = train_f
            elif r < dev_ratio:
                f = dev_f
            else:
                f = test_f
            pickle.dump((x, y), f)
    except KeyboardInterrupt:
        pass

    train_f.close()
    dev_f.close()
    test_f.close()


if __name__ == '__main__':
    make_data()
