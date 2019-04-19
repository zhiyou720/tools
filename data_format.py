#!/usr/bin/env python
# encoding: utf-8
"""
@author: zk
@contact: kun.zhang@nuance.com
@file: data_format.py
@time: 4/19/2019 10:08 AM
@desc:
"""


def add_label(data, topic, sep='\t'):
    """
    add label for data
    :param data:
    :param topic:
    :param sep:
    :return:
    """
    res = []
    for item in data:
        new_item = topic + sep + item
        res.append(new_item)
    return res
