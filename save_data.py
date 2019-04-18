#!/usr/bin/env python
# encoding: utf-8
"""
@author: zk
@contact: kun.zhang@nuance.com
@file: save_data.py
@time: 4/16/2019 6:01 PM
@desc:
"""


def save_txt_file(data, path, end='', re_sub=''):
    """
    This func is used to saving data to txt file
    support data type:
    list: Fully support
    dict: Only save dict key
    str: will save single char to each line
    tuple: Fully support
    set: Fully support
    :param data: data
    :param path: path to save
    :type path: str
    :param end:
    :param re_sub:
    :return: None
    """
    import re
    if type(data) not in [list, dict, str, tuple, set] or type(path) != str:
        raise TypeError

    with open(path, 'a', encoding='utf-8-sig') as f:
        for item in data:
            tmp = re.sub(re_sub, '', item)
            f.write(tmp + end)
