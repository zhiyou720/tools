#!/usr/bin/env python
# encoding: utf-8
"""
@author: zk
@contact: kun.zhang@nuance.com
@file: loss_funtion.py
@time: 5/7/2019 3:52 PM
@desc:
"""
import numpy as np


def root_mean_square_error(prediction, target):
    return np.sqrt(((prediction - target) ** 2).mean())


def softmax(x):
    exp = np.exp(x)
    return exp / np.sum(exp)


def cross_entropy(prediction, target):
    return -np.mean(np.sum(target * np.log(prediction)))


if __name__ == '__main__':
    _x = np.array([0.01, 0.08, 0.00000003, 0.98])
    y = np.array([0, 0, 0, 1])
    print(cross_entropy(_x, y))
    print(root_mean_square_error(_x, y))
