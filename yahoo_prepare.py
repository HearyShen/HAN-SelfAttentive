import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--answers_path", default="FullOct2007.xml")
parser.add_argument("--answers_max", type=int, default=100000)
args = parser.parse_args()

import os
import ujson as json
from bs4 import BeautifulSoup
import spacy
import pickle
import random
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from yahoo import *

en = spacy.load('en')
en.pipeline = [('tagger', en.tagger), ('parser', en.parser)]


def read_answers(path = args.answers_path):
    count = 0
    for i in range(2):      # read part 1~2
        with open(path + '.part' + str(i+1), 'r', encoding='utf-8') as xmlFile:
            active = False
            xml = ''
            for line in xmlFile:
                if count > args.answers_max:
                    return
                if '<document' in line:
                    active = True
                    xml = ''
                elif '</document>' in line:
                    xml += line
                    active = False
                    # parse xml <document>
                    xmlTree = BeautifulSoup(xml, 'lxml')
                    try:
                        subject = xmlTree.find('subject').string.strip()
                        content = xmlTree.find('content').string.strip()
                        bestAnswer = xmlTree.find('bestanswer').string.strip()
                        bestAnswer = str(bestAnswer).replace('<br />', '').replace('\n', '')
                        text = subject + ' ' + content + ' '+ bestAnswer
                        category = xmlTree.find('maincat').string
                        if category in mainCategories.keys():   # only yield main 10 categories
                            count += 1
                            yield (text, category)
                    except:
                        pass
                if active:
                    xml += line

    #
    # xmlTree = BeautifulSoup(open(path), 'lxml')
    # print('building xmlTree successfully, path: %s' % path)
    # # xmlTree = BeautifulSoup(open(r'D:\Codes\Test\YahooAnswers\FullOct2007.xml.part1'), 'lxml')
    # for document in xmlTree.find_all('document'):
    #     category = document.find('maincat').string
    #     yield category

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
    for i, (text, category) in enumerate(read_answers()):
        doc = en.tokenizer(text)
        for token in doc:
            freq[token.orth_] += 1
        count = i + 1  # count starts from 1 (i starts from 0)
        if count % 10000 == 0:
            with open(path, 'wb') as freq_dist_f:
                pickle.dump(freq, freq_dist_f)
            print('dump at {}'.format(count))
    return freq


def build_vocabulary(lower=3, n=150000):
    try:
        with open(vocab_fn, 'rb') as vocab_file:
            vocab = pickle.load(vocab_file)
            print('vocabulary loaded')
            return vocab
    except IOError:
        print('building vocabulary')
    freq = build_word_frequency_distribution()
    words_freqDesc = list(sorted(freq.items(), key=lambda x: -x[1]))
    vocab = {}
    i = lower
    for w, freq in words_freqDesc:
        if freq < vocab_minFreq:
            break
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
        for (text, category) in tqdm(read_answers()):
            x = []
            for sent in en(text).sents:
                x.append([vocab.get(tok.orth_, UNKNOWN) for tok in sent])
            y = mainCategories[category]

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

#
# def test_readXML():
#     for i, (text, category) in enumerate(tqdm(read_answers('FullOct2007.xml'))):
#         if i > 10:
#             break
#         print(text)
#         print("Category: %s" % category)
#         print("***************")


if __name__ == '__main__':
    # test_readXML()
    make_data()
