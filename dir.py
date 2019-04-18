#!/usr/bin/env python
# encoding: utf-8
"""
@author: zk
@contact: kun.zhang@nuance.com
@file: dir.py
@time: 4/16/2019 6:14 PM
@desc:
"""
import os


def load_file_name(path):
    """
    This func can get root, subdir, file_names
    :param path:
    :type path:str
    :return:
    """
    for root, dirs, files in os.walk(path):
        return root, dirs, files


def check_dir(path):
    """
    check dir exists
    :param path:
    :type path:str
    :return:
    :rtype: bool
    """
    return os.path.exists(path)


def mkdir(path):
    """
    :param path:
    :type path: str
    :return: None
    """
    path = path.strip()
    if not check_dir(path):
        os.makedirs(path)


if __name__ == '__main__':
    pass
