#!/usr/bin/env python
# encoding: utf-8
"""
@author: zk
@contact: kun.zhang@nuance.com
@file: number_trans.py
@time: 5/21/2019 10:50 AM
@desc:
"""
import re


def arab2cn(arab):
    num_dict = {'1': '一', '2': '二', '3': '三', '4': '四', '5': '五', '6': '六', '7': '七', '8': '八', '9': '九', '0': '零', }
    index_dict = {1: '', 2: '十', 3: '百', 4: '千', 5: '万', 6: '十', 7: '百', 8: '千', 9: '亿'}

    nums = list(str(arab))
    nums_index = [x for x in range(1, len(nums) + 1)][-1::-1]
    string = ''
    for index, item in enumerate(nums):
        string = "".join((string, num_dict[item], index_dict[nums_index[index]]))
        string = re.sub("零[十百千零]*", "零", string)
        string = re.sub("零万", "万", string)
        string = re.sub("亿万", "亿零", string)
        string = re.sub("零零", "零", string)
        string = re.sub("零\\b", "", string)
    return string


if __name__ == '__main__':
    for i in range(101):
        print(arab2cn(i))

