#!/usr/bin/env python
# encoding: utf-8
"""
@author: zk
@contact: kun.zhang@nuance.com
@file: parse_sogou2.py
@time: 5/9/2019 11:19 AM
@desc:
"""
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

    save_txt_file(res, './out3.txt', end='\n', re_sub=['&nbsp', '【', '】'])
