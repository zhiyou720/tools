#!/usr/bin/env python
# encoding: utf-8
"""
@author: zk
@contact: kun.zhang@nuance.com
@file: gen_corpus.py
@time: 4/15/2019 5:54 PM
@desc:
"""


def gen_corpus(grammar):
    res = []
    for i in grammar[0]:
        tmp1 = i
        for j in grammar[1]:
            tmp2 = j
            for k in grammar[2]:
                tmp3 = k
                res.append(tmp1 + tmp2 + tmp3)

    for i in grammar[1]:
        tmp4 = i
        for j in grammar[2]:
            tmp5 = j
            res.append(tmp4 + tmp5)

    return res


def gen_corpus2(grammar, rep='', f=True):
    res = []
    for i in grammar[0]:
        tmp1 = i
        for j in grammar[1]:
            tmp2 = j
            for k in grammar[2]:
                tmp3 = k
                for l in grammar[3]:
                    tmp4 = l
                    res.append(tmp1 + tmp2 + tmp3 + tmp4)

    for i in grammar[1]:
        tmp5 = i
        for j in grammar[2]:
            tmp6 = j
            for k in grammar[3]:
                tmp7 = k
                res.append(tmp5 + tmp6 + tmp7)
    if f:
        import re
        for i in grammar[1]:
            tmp8 = i

            for j in grammar[3]:
                tmp9 = j
                for item in ['成', '为', '到']:
                    if item in tmp9:
                        tmp9 = re.sub(item, rep, tmp9)
                res.append(tmp8 + tmp9)

    return res


if __name__ == '__main__':
    _grammar = [
        ['为我', '我想', '我想要', '请请帮我', '帮我'],
        ['关闭', '停止', '停掉', '停了', '关上', '停', '关', '关掉'],
        ['空调', '风扇', '自动空调', 'AC', '压缩机', '冷气', '制冷']
    ]

    corpus1 = gen_corpus(_grammar)
    # corpus2 = solution2(_grammar, 1, _grammar[0])
    # from tools.save_data import save_txt_file
    # save_txt_file(corpus1, './res.txt', end='\n')
    print(corpus1)
    # print(corpus2)
