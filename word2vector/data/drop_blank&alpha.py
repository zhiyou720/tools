#!/usr/bin/env python
# encoding: utf-8
"""
@author: zk
@contact: kun.zhang@nuance.com
@file: drop_blank&alpha.py
@time: 5/8/2019 5:26 PM
@desc:
"""
import re
import jieba
from tqdm import tqdm
from hanziconv import HanziConv
from tools.load_data import load_txt_data
from tools.save_data import save_txt_file

if __name__ == '__main__':
    data = load_txt_data('./corpus.zhwiki.txt')

    for i in tqdm(range(len(data))):
        tmp = jieba.cut(HanziConv.toSimplified(re.sub('[A-Za-z ]', '', data[i])), cut_all=False)
        data[i] = ' '.join(tmp)
    save_txt_file(data, './out.txt')
