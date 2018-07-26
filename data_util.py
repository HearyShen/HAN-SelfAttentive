import numpy as np


def batch(inputs):
    batch_size = len(inputs)

    document_sizes = np.array([len(doc) for doc in inputs], dtype=np.int32)
    document_size = document_sizes.max()        # max count of sentences in a document

    sentence_sizes_ = [[len(sent) for sent in doc] for doc in inputs]
    sentence_size = max(map(max, sentence_sizes_))      # max count of words in a sentence

    # initiallize a batch as all-zero
    b = np.zeros(shape=[batch_size, document_size, sentence_size], dtype=np.int32)  # == PAD

    sentence_sizes = np.zeros(shape=[batch_size, document_size], dtype=np.int32)
    for i, document in enumerate(inputs):
        for j, sentence in enumerate(document):
            sentence_sizes[i, j] = sentence_sizes_[i][j]    # sentence_sizes[i,j] := count(words in document_i, sentence_j)
            for k, word in enumerate(sentence):            # b[i, j, :] := all words in document_i, sentence_j
                b[i, j, k] = word

    return b, document_sizes, sentence_sizes
