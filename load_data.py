#!/usr/bin/env python
# encoding: utf-8
"""
@author: zk
@contact: kun.zhang@nuance.com
@file: load_data.py
@time: 4/16/2019 5:49 PM
@desc:
"""


def load_txt_data(path):
    """
    This func is used to reading txt file
    :param path: path where file stored
    :type path: str
    :return: string lines in file in a list
    :rtype: list
    """
    if type(path) != str:
        raise TypeError
    res = []
    with open(path, 'r', encoding='utf-8-sig') as f:
        for line in f.readlines():
            res.append(line)
    return res


def load_excel_data(path):
    """
    This func is used to reading excel file
    :param path: path where file stored
    :type path: str
    :return: data saved in a pandas DataFrame
    :rtype: pandas.DataFrame
    """
    if type(path) != str:
        raise TypeError
    import pandas as pd
    return pd.read_excel(path).loc[:]


def load_variable(path):
    """
    :param path:
    :return:
    """
    import pickle
    return pickle.load(open(path, 'rb'))


if __name__ == '__main__':
    pass
