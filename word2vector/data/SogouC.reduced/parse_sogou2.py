#!/usr/bin/env python
# encoding: utf-8
"""
@author: zk
@contact: kun.zhang@nuance.com
@file: parse_sogou2.py
@time: 5/9/2019 11:19 AM
@desc:
"""
import re
import jieba
from tqdm import tqdm
from tools.dir import load_all_file_name
from tools.load_data import load_txt_data
from tools.save_data import save_txt_file

if __name__ == '__main__':
    tmp = []
    load_all_file_name('./', tmp, suffix='.txt')

    res = []

    for file_path in tqdm(tmp):
        file_data = load_txt_data(file_path, mode='gbk')
        res += file_data

    res2 = []
    for i in tqdm(range(len(res))):
        pattern = ['。', '？', '！']
        res[i] = res[i].strip()
        for pa in pattern:
            res[i] = re.sub(pa, '\n', res[i])
            res[i] = re.sub(' ', '', res[i])
        res2.append(res[i])

    res3 = []
    for item in tqdm(res2):
        tmp = item.split('\n')
        for line in tmp:
            _line = line.strip()
            if _line:
                res3.append(re.sub('&nbsp', '', _line))

    res4 = []
    for item in tqdm(res3, desc='分词'):
        res4.append(' '.join(jieba.cut(item, cut_all=False)))

    save_txt_file(res4, './corpus.sogou.news.txt', end='\n')
