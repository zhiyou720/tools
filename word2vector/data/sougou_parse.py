#!/usr/bin/env python
# encoding: utf-8
"""
@author: zk
@contact: kun.zhang@nuance.com
@file: sougou_parse.py
@time: 5/8/2019 5:39 PM
@desc:
"""
# -*- coding: utf-8 -*-
# word_segment.py用于语料分词

import re
import jieba
from tqdm import tqdm
from hanziconv import HanziConv
from tools.load_data import load_txt_data
from tools.save_data import save_txt_file


# 先用正则将<content>和</content>去掉
def reTest(content):
    reContent = re.sub('<content>|</content>', '', content)
    return reContent


if __name__ == '__main__':
    print('Read file...')
    raw_data = load_txt_data('./corpus.sogou.txt')
    print('Finish read file')
    res = []
    for data in tqdm(raw_data):
        if '<content>' in data:
            content = HanziConv.toSimplified(re.sub('<content>|</content>', '', data)).split('。')
            for item in content:
                sent = ' '.join(jieba.cut(item, cut_all=False))
                res.append(sent)
    save_txt_file(res, './out2.txt', end='\n')
