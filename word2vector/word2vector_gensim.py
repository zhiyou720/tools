#!/usr/bin/env python
# encoding: utf-8
"""
@author: zk
@contact: kun.zhang@nuance.com
@file: word2vector_gensim.py
@time: 5/10/2019 11:19 AM
@desc:
"""
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')  # 忽略警告

import logging
import os.path
import sys
import multiprocessing
from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


if __name__ == '__main__':
    # print open('/Users/sy/Desktop/pyRoot/wiki_zh_vec/cmd.txt').readlines()
    # sys.exit()

    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # inp为输入语料, outp1 为输出模型, outp2为原始c版本word2vec的vector格式的模型
    fdir = './'
    inp = fdir + 'data/corpus.merge.txt'
    outp1 = fdir + 'corpus.merge.text.model'
    outp2 = fdir + 'corpus.merge.vector'

    # 训练skip-gram模型
    model = Word2Vec(LineSentence(inp), size=150, window=5, min_count=2,
                     workers=multiprocessing.cpu_count())

    # 保存模型
    model.save(outp1)
    model.wv.save_word2vec_format(outp2, binary=False)
