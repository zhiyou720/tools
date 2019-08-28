#!/usr/bin/env python
# encoding: utf-8
"""
@author: zk
@contact: kun.zhang@nuance.com
@file: word2vec.py
@time: 5/6/2019 1:19 PM
@desc:
"""

import time
import argparse
import math
import struct
import sys
import numpy as np
from tqdm import tqdm
from tools.dataio import save_variable, load_variable, mkdir, check_dir


class VocabItem:
    def __init__(self, word):
        self.word = word
        self.count = 0
        self.path = None  # Path (list of indices) from the root to the word (leaf)
        self.code = None  # Huffman encoding


class Vocab:
    def __init__(self, fi, min_count):
        vocab_items = []
        vocab_hash = {}
        word_count = 0
        fi = open(fi, 'r', encoding='utf-8')

        # Add special tokens <bol> (beginning of line) and <eol> (end of line)
        for token in ['<bol>', '<eol>']:
            vocab_hash[token] = len(vocab_items)
            vocab_items.append(VocabItem(token))

        for line in fi:
            '''是否要处理标点符号？？？'''
            tokens = line.split()
            for token in tokens:
                if token not in vocab_hash:
                    vocab_hash[token] = len(vocab_items)
                    vocab_items.append(VocabItem(token))

                vocab_items[vocab_hash[token]].count += 1
                word_count += 1

                if word_count % 10000 == 0:
                    sys.stdout.write("\rReading word %d" % word_count)
                    sys.stdout.flush()

            # Add special tokens <bol> (beginning of line) and <eol> (end of line)
            vocab_items[vocab_hash['<bol>']].count += 1
            vocab_items[vocab_hash['<eol>']].count += 1
            word_count += 2
        # [print(x) for x in vocab_hash]
        # [print(x.count, x.word) for x in vocab_items]

        self.bytes = fi.tell()  # 返回文件指针的位置
        self.vocab_items = vocab_items  # List of VocabItem objects
        self.vocab_hash = vocab_hash  # Mapping from each token to its index in vocab
        self.word_count = word_count  # Total number of words in train file

        # Add special token <unk> (unknown),
        # merge words occurring less than min_count into <unk>, and
        # sort vocab in descending order by frequency in train file
        self.__sort(min_count)

        print('Total words in training file: %d' % self.word_count)
        print('Total bytes in training file: %d' % self.bytes)
        print('Vocab size: %d' % len(self))

    def __getitem__(self, i):
        return self.vocab_items[i]

    def __len__(self):
        return len(self.vocab_items)

    def __iter__(self):
        return iter(self.vocab_items)

    def __contains__(self, key):
        return key in self.vocab_hash

    def __sort(self, min_count):
        tmp = [VocabItem('<unk>')]
        unk_hash = 0

        count_unk = 0
        for token in self.vocab_items:
            if token.count < min_count:
                count_unk += 1
                tmp[unk_hash].count += token.count
            else:
                tmp.append(token)

        tmp.sort(key=lambda tokens: tokens.count, reverse=True)

        # Update vocab_hash
        vocab_hash = {}
        for i, token in enumerate(tmp):
            vocab_hash[token.word] = i

        self.vocab_items = tmp
        self.vocab_hash = vocab_hash
        print()
        print('Unknown vocab size:', count_unk)

    def indices(self, tokens):
        return [self.vocab_hash[token] if token in self else self.vocab_hash['<unk>'] for token in tokens]

    def encode_huffman(self):
        # Build a Huffman tree
        vocab_size = len(self)
        count = [t.count for t in self] + [int(1e15)] * (vocab_size - 1)
        parent = [0] * (2 * vocab_size - 2)
        binary = [0] * (2 * vocab_size - 2)

        pos1 = vocab_size - 1
        pos2 = vocab_size

        for i in range(vocab_size - 1):
            # Find min1
            if pos1 >= 0:
                if count[pos1] < count[pos2]:
                    min1 = pos1
                    pos1 -= 1
                else:
                    min1 = pos2
                    pos2 += 1
            else:
                min1 = pos2
                pos2 += 1

            # Find min2
            if pos1 >= 0:
                if count[pos1] < count[pos2]:
                    min2 = pos1
                    pos1 -= 1
                else:
                    min2 = pos2
                    pos2 += 1
            else:
                min2 = pos2
                pos2 += 1

            count[vocab_size + i] = count[min1] + count[min2]
            parent[min1] = vocab_size + i
            parent[min2] = vocab_size + i
            binary[min2] = 1

        # Assign binary code and path pointers to each vocab word
        root_idx = 2 * vocab_size - 2
        for i, token in enumerate(self):
            path = []  # List of indices from the leaf to the root
            code = []  # Binary Huffman encoding from the leaf to the root

            node_idx = i
            while node_idx < root_idx:
                if node_idx >= vocab_size:
                    path.append(node_idx)
                code.append(binary[node_idx])
                node_idx = parent[node_idx]
            path.append(root_idx)

            # These are path and code from the root to the leaf
            token.path = [j - vocab_size for j in path[::-1]]
            token.code = code[::-1]


class UnigramTable:
    """
    A list of indices of tokens in the vocab following a power law distribution,
    used to draw negative samples.
    """

    def __init__(self, vocab):
        # vocab_size = len(vocab)
        power = 0.75
        norm = sum([math.pow(t.count, power) for t in vocab])  # Normalizing constant

        table_size = int(1e8)  # Length of the unigram table
        table = np.zeros(table_size, dtype=np.uint32)

        print('Filling unigram table')
        p = 0  # Cumulative probability
        i = 0
        for j, unigram in tqdm(enumerate(vocab)):
            p += float(math.pow(unigram.count, power)) / norm
            while i < table_size and float(i) / table_size < p:
                table[i] = j
                i += 1
        self.table = table

    def sample(self, count):
        indices = np.random.randint(low=0, high=len(self.table), size=count)
        return [self.table[i] for i in indices]


def sigmoid(z):
    try:
        return 1 / (1 + math.exp(-z))
    except OverflowError:
        if z > 6:
            return 1.0
        if z < -6:
            return 0.0


def init_net(dim, vocab_size):
    return np.random.uniform(low=-0.5 / dim, high=0.5 / dim, size=(vocab_size, dim)), np.zeros(shape=(vocab_size, dim))


def save(vocab, syn0, fo, binary):
    print('Saving model to', fo)
    dim = len(syn0[0])
    if binary:
        f = open(fo, 'wb')
        f.write(b'%d %d\n' % (len(syn0), dim))
        f.write(b'\n')
        for token, vector in zip(vocab, syn0):
            f.write(b'%s ' % token.word)
            for s in vector:
                f.write(struct.pack('f', s))
            f.write(b'\n')
    else:
        f = open(fo, 'w')
        f.write('%d %d\n' % (len(syn0), dim))
        for token, vector in zip(vocab, syn0):
            word = token.word
            vector_str = ' '.join([str(s) for s in vector])
            f.write('%s %s\n' % (word, vector_str))

    f.close()


def train(fi, vocab, table, model_type='cbow', neg=0, dim=100, alpha=0.001, win=5, epoch_num=1, batch_size=8):
    # Init net
    if epoch_num == 1:
        syn0, syn1 = init_net(dim, len(vocab))
    else:
        syn0 = load_variable(
            './check_point/{}/look_up_table_vocab_size_{}_epoch_{}'.format(model_type, len(vocab), epoch_num - 1))
        syn1 = load_variable(
            './check_point/{}/format_vector_vocab_size_{}_epoch_{}'.format(model_type, len(vocab), epoch_num - 1))
    print('Epoch: {}'.format(epoch_num))
    print('Model type: {}'.format(model_type.upper()))

    suf = None

    if neg > 0:
        suf = 'Negative Sampling'
        print('Speed up method: {}, size: {}'.format(suf, neg))
    else:
        suf = 'Hierarchical Softmax'
        print('Speed up method: {}'.format(suf))
    word_count = 0
    sent_count = 0

    f = open(fi, 'r', encoding='utf-8')

    tb = time.time()
    i = 0
    batch = {}
    compare = [' ']
    sent_num = len(f.readlines())
    f.seek(0)

    for line in f.readlines():
        t0 = time.time()
        line = line.strip()
        # Skip blank lines
        if not line:
            continue

        # Init sent, a list of indices of words in line
        sent = vocab.indices(['<bol>'] + line.split() + ['<eol>'])

        for sent_pos, token in enumerate(sent):

            current_win = np.random.randint(low=1, high=win + 1)
            context_start = max(sent_pos - current_win, 0)
            context_end = min(sent_pos + current_win + 1, len(sent))
            context = sent[context_start:sent_pos] + sent[sent_pos + 1:context_end]  # Turn into an iterator?
            # CBOW
            if model_type == 'cbow':
                # Compute neu1
                neu1 = np.mean(np.array([syn0[c] for c in context]), axis=0)
                assert len(neu1) == dim, 'neu1 and dim do not agree'

                # Init neu1e with zeros
                neu1e = np.zeros(dim)

                # Compute neu1e and update syn1
                if neg > 0:
                    classifiers = [(token, 1)] + [(target, 0) for target in table.sample(neg)]
                else:
                    classifiers = zip(vocab[token].path, vocab[token].code)
                for target, label in classifiers:
                    # target = vocab index; label = True of False
                    x = np.dot(neu1, syn1[target])
                    p = sigmoid(x)
                    g = alpha * (label - p)
                    if target in batch:
                        batch[target].append(g)
                    else:
                        batch[target] = [g]

                # Update syn0
                if i >= batch_size:
                    for tar in batch:
                        g = np.mean(batch[tar])
                        neu1e += g * syn1[tar]  # Error to back propagate to syn0
                        syn1[tar] += g * neu1  # Update syn1

                    for context_word in context:
                        syn0[context_word] += neu1e

                    i = 0
                    batch = {}
                else:
                    last = compare[0]
                    compare.pop(0)
                    if last != line:
                        i += 1
                    compare.append(line)

            # Skip-gram
            elif model_type == 'skip-gram':
                for context_word in context:
                    # Init neu1e with zeros
                    neu1e = np.zeros(dim)

                    # Compute neu1e and update syn1
                    if neg > 0:
                        classifiers = [(token, 1)] + [(target, 0) for target in table.sample(neg)]
                    else:
                        classifiers = zip(vocab[token].path, vocab[token].code)
                    for target, label in classifiers:
                        x = np.dot(syn0[context_word], syn1[target])
                        p = sigmoid(x)
                        g = alpha * (label - p)

                        neu1e += g * syn1[target]  # Error to backpropagate to syn0
                        syn1[target] += g * syn0[context_word]  # Update syn1

                    # Update syn0
                    syn0[context_word] += neu1e
            else:
                print('Invaild model type {}'.format(model_type))
                raise ValueError

            word_count += 1
        sent_count += 1
        t1 = time.time()
        already_use = t1 - tb
        average = (t1 - tb) / sent_count
        rest = (sent_num - sent_count) * average / 60
        current_use = t1 - t0
        sys.stdout.write("\rTraining word %d/%d sentence %d/%d, Time cost: %fs/%fm [%fs/sentence]" %
                         (word_count, vocab.word_count, sent_count, sent_num, already_use, rest, current_use))
        sys.stdout.flush()

    f.close()
    # Save model to file
    save(vocab, syn0, 'out.model', binary=True)
    # syn0 = w
    # w * vocab_one_hot = word_vector
    # so we must save w and vacab_one_hot
    return syn0, syn1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', help='Training file', dest='fi', required=True)
    parser.add_argument('-model_type', help='cbow or skip-gram', dest='cbow', default='cbow', type=str)
    parser.add_argument('-negative',
                        help='Number of negative examples (>0) for negative sampling, 0 for hierarchical softmax',
                        dest='neg', default=5, type=int)
    parser.add_argument('-dim', help='Dimensionality of word embeddings', dest='dim', default=100, type=int)
    parser.add_argument('-alpha', help='Starting alpha', dest='alpha', default=0.025, type=float)
    parser.add_argument('-window', help='Max window length', dest='win', default=5, type=int)
    parser.add_argument('-min_count', help='Min count for words used to learn <unk>', dest='min_count', default=3,
                        type=int)
    parser.add_argument('-batch_size', help='Size of batch', dest='batch', default=8, type=int)
    parser.add_argument('-epoch', help='Number of training epochs', dest='epoch', default=1, type=int)
    args = parser.parse_args()

    vocab_check_path = './check_point/{}/vocab_for_file_{}'.format(args.model_type, args.fi)

    if check_dir(vocab_check_path):
        sys.stdout.write("\rFound vocabulary file")
        sys.stdout.flush()
        vocab = load_variable(vocab_check_path)
    else:
        # Read train file to init vocab
        vocab = Vocab(args.fi, args.min_count)
        mkdir('./check_point/{}/'.format(args.model_type))
        save_variable(vocab, vocab_check_path)

    # Speed up method
    table = None
    if args.negative > 0:
        print('Initializing unigram table')
        table = UnigramTable(vocab)
    else:
        print('Initializing Huffman tree')
        vocab.encode_huffman()
    # Begin training
    for epoch_num in tqdm(range(1, args.epoch)):
        sy0, sy1 = train(args.fi, vocab, table, model_type=args.model_type, neg=args.neg,
                         dim=args.dim, alpha=args.alpha, win=args.win, epoch_num=epoch_num, batch_size=8)
        save_variable(
            sy0, './check_point/{}/look_up_table_vocab_size_{}_epoch_{}'.format(args.model_type, len(vocab), epoch_num))
        save_variable(
            sy1, './check_point/{}/format_vector_vocab_size_{}_epoch_{}'.format(args.model_type, len(vocab), epoch_num))


if __name__ == '__main__':
    # # file_in = './data/corpus.merge.txt'
    # file_in = './data/test.txt'
    # _vocab = Vocab(file_in, 0)
    #
    # # Speed up method
    # _table = None
    #
    # _vocab.encode_huffman()
    #
    # # for epoch in range(1, 4):
    # #     print('Epoch {}'.format(epoch))
    # train(file_in, _vocab, 'cbow', 0, table, 150, 0.001, 2)

    file_in = './data/corpus.merge.txt'
    model_type = 'cbow'
    min_count = 2
    neg = 0
    _epoch = 2
    dim = 150
    win = 5
    alpha = 0.025

    _vocab_check_path = './check_point/{}/vocab_for_file_{}'.format(model_type, 'corpus.merge.txt')

    if check_dir(_vocab_check_path):
        print("Found vocabulary file")
        _vocab = load_variable(_vocab_check_path)
    else:
        # Read train file to init vocab
        _vocab = Vocab(file_in, min_count)
        mkdir('./check_point/{}/'.format(model_type))
        save_variable(_vocab, _vocab_check_path)

    # Speed up method
    _table = None
    if neg > 0:
        print('Initializing unigram table')
        _table = UnigramTable(_vocab)
    else:
        print('Initializing Huffman tree')
        _vocab.encode_huffman()
    # Begin training
    for _epoch_num in range(1, _epoch):
        _sy0, _sy1 = train(file_in, _vocab, _table, model_type=model_type, neg=neg,
                           dim=dim, alpha=alpha, win=win, epoch_num=_epoch_num, batch_size=32)
        save_variable(
            _sy0, './check_point/{}/look_up_table_vocab_size_{}_epoch_{}'.format(model_type, len(_vocab), _epoch_num))
        save_variable(
            _sy1, './check_point/{}/format_vector_vocab_size_{}_epoch_{}'.format(model_type, len(_vocab), _epoch_num))
