#!/usr/bin/env python
# encoding: utf-8
"""
@author: zk
@contact: kun.zhang@nuance.com
@file: translate_and_tag.py
@time: 3/29/2019 1:28 PM
@desc:
"""
from googletrans import Translator

translator = Translator()
line_head = []
res = []
with open('./data/zhangkun.txt', encoding='utf-8-sig') as f:
    for line in f.readlines():
        nline = line.strip().split('\t')
        line_head = nline[:2]
        sentence_tag = nline[2:][0].split(' ')
        sentence = []
        for item in sentence_tag:
            word = item.split('|')[0]
            sentence.append(word)
        words = " ".join(sentence)
        trans = translator.translate(words, dest='zh-CN')
        print(words)
        print(trans.text)
        res.append(line)
        res.append('\t'.join(line_head) + '\t' + trans.text + '\n')
        res.append('\n')

with open('./result/res.txt', 'a', encoding='utf-8-sig') as f1:
    for item in res:
        if '\n' not in item:
            item += '\n'
        f1.write(item)
