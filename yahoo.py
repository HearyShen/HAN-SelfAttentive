import os
import pickle

train_dir = os.path.join(os.path.curdir, 'yahoo')
data_dir = os.path.join(train_dir, 'data')

for dir in [train_dir, data_dir]:
    if not os.path.exists(dir):
        os.makedirs(dir)

trainset_fn = os.path.join(data_dir, 'train.dataset')
devset_fn = os.path.join(data_dir, 'dev.dataset')
testset_fn = os.path.join(data_dir, 'test.dataset')
vocab_fn = os.path.join(data_dir, 'vocab.pickle')

reserved_tokens = 5
unknown_id = 2

vocab_minFreq = 6

def get_vocab_size():
    vocab = read_vocab()
    return len(vocab) + unknown_id + 1

mainCategories = {'Society & Culture':0,
                  'Science & Mathematics':1,
                  'Health':2,
                  'Education & Reference':3,
                  'Computers & Internet':4,
                  'Sports':5,
                  'Business & Finance':6,
                  'Entertainment & Music':7,
                  'Family & Relationships':8,
                  'Politics & Government':9
                  }
assert len(mainCategories) == 10


def _read_dataset(fn, review_max_sentences=32, sentence_max_length=32, epochs=1):
    c = 0
    while 1:
        c += 1
        if epochs > 0 and c > epochs:
            return
        print('epoch %s' % c)
        with open(fn, 'rb') as f:
            try:
                while 1:
                    x, y = pickle.load(f)

                    # clip review to specified max lengths
                    x = x[:review_max_sentences]
                    x = [sent[:sentence_max_length] for sent in x]

                    # y -= 1
                    assert y >= 0 and y <= 9
                    yield x, y
            except EOFError:
                continue


def read_trainset(epochs=1):
    return _read_dataset(trainset_fn, epochs=epochs)


def read_devset(epochs=1):
    return _read_dataset(devset_fn, epochs=epochs)

def read_testset(epochs=1):
    return _read_dataset(testset_fn, epochs=epochs)


def read_vocab():
    with open(vocab_fn, 'rb') as f:
        return pickle.load(f)


def read_labels():
    return mainCategories
