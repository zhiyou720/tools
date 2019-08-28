#!/usr/bin/env python
# encoding: utf-8
"""
@author: zk
@contact: kun.zhang@nuance.com
@file: parameter_check.py
@time: 5/24/2019 5:48 PM
@desc:
"""
import sys


def none_parameter_check(paras):
    """
    检查空参数
    :param paras:
    :type paras: list
    :return:
    """
    for item in paras:
        if not item:
            sys.stdout.write('none value Error: %s' % item)
            raise ValueError
