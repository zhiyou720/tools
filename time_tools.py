#!/usr/bin/env python
# encoding: utf-8
"""
@author: zk
@contact: kun.zhang@nuance.com
@file: time_tools.py
@time: 4/25/2019 11:30 AM
@desc:
"""
import time
import re


def get_time():
    t = time.asctime(time.localtime(time.time())).split(' ')
    t[3] = re.sub(':', '_', t[3])
    return '_'.join(t)


class LoopTime:
    def __init__(self, total_loop_times):
        self.t0 = time.time()
        self.total = total_loop_times
        self.cost = 0
        self.rest = 0

    def cost_time(self):
        self.cost = time.time() - self.t0
        return time.time() - self.t0

    def rest_time(self, i):
        self.cost = time.time() - self.t0
        self.rest = self.cost / (i + 1) * (self.total - i + 1)
        return self.rest


if __name__ == '__main__':
    _t = LoopTime(100000)
    for _i in range(100000):
        time.sleep(1)
        print(_t.cost_time())
        print(_t.rest_time(_i))
