#!/usr/bin/env python
# encoding: utf-8
"""
@author: zk
@contact: kun.zhang@nuance.com
@file: topic_slot.py
@time: 3/22/2019 2:51 PM
@desc:
我看了你发的excel, 有两个建议， 一个是slot name之间用 ", " （逗号空格）隔开。
因为我们之前的NLUDPS 大致都是这个格式，这样大家看起来比较习惯
还有就是 希望slot name 能够按照字母序排列一下， 这样有些topic要提取的slot比较多的，
如果大家想找某个slot有没有，也能比较快速地定位到
"""

import os
import pandas as pd
from tqdm import tqdm
import xml.dom.minidom as xmldom


def read_xml_slot_name(project_name, slot_name_dict):
    """
    :param project_name:
    :param slot_name_dict:
    :return:
    """
    xml_file_path = os.path.abspath("data/{}_pattern_generate.xml".format(project_name))
    element_obj = xmldom.parse(xml_file_path).documentElement.getElementsByTagName("ITEM")
    slot_name = {}
    for i in tqdm(range(len(element_obj))):

        data = element_obj[i].firstChild.data.split('\t')

        if len(data) >= 2 and data[1]:
            slot_name[data[0]] = data[1].split('#')

    slot_name_dict[project_name] = slot_name


def update_slot_name(slot_name_dict, in_path='./data/topics_and_texts.xlsx', out_path='./result/result.xls'):
    df = pd.read_excel(in_path, sheet_name='Sheet1')

    for i in tqdm(range(len(df))):
        key = df['topic'][i]
        temp_slot_name = []
        for project in slot_name_dict:
            if key in slot_name_dict[project] and project in df['projects'][i].split(';'):
                for k in slot_name_dict[project][key]:
                    if k not in temp_slot_name and k:
                        temp_slot_name.append(k)
        temp_slot_name.sort()
        df['slot_name'][i] = ', '.join(temp_slot_name)

    df.to_excel(out_path)


if __name__ == '__main__':
    slot_names = {}
    for item in ['banma', 'denso', 'ecarx', 'jlr']:
        read_xml_slot_name(item, slot_names)

    # print(slot_names)

    update_slot_name(slot_names)
