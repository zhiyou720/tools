#!/usr/bin/env python
# encoding: utf-8
"""
@author: zk
@contact: kun.zhang@nuance.com
@file: time.py
@time: 4/25/2019 11:30 AM
@desc:
"""
import time
import re


def get_time():
    t = time.asctime(time.localtime(time.time())).split(' ')
    t[3] = re.sub(':', '_', t[3])
    return '_'.join(t)
