#!/usr/bin/env python
# encoding: utf-8
"""
@author: zk
@contact: kun.zhang@nuance.com
@file: slot_value.py
@time: 3/28/2019 1:44 PM
@desc:
1. Default order: ecarx > saic > denso > jlr
2. Slot value: enum preferred (*_slot.map.dat)
3. Slot name sorted in alphabetical order
4. The max length of slot values is 20
"""
import os
import re
import pandas as pd
from tqdm import tqdm


def get_file_name(file_dir):
    """
    :param file_dir: path
    :type file_dir: str
    :return: a list for file names
    :rtype: list
    """
    for root, dirs, files in os.walk(file_dir):
        return files


def de_duplication_sort(result_dict, max_l):
    """
    :param result_dict:
    :param max_l:
    :return:
    """
    for slot in result_dict:
        value = result_dict[slot]
        i = 0
        new_value = []
        if len(set(value[:])) > max_l:
            flag = True
        else:
            flag = False
        while i < max_l and i < len(value):
            if value[i] not in new_value:
                new_value.append(value[i])
            i += 1
        new_value.sort()

        if flag:
            new_value.append('...')

        result_dict[slot] = new_value
    return result_dict


def clean_copy_file(file_names, err='- Copy'):
    result = []
    for file_name in file_names:
        if err not in file_name:
            result.append(file_name)
    return result


def extract_parameter_from_dat_file(line):
    line = line.strip().split("###")
    line.pop(-1)
    slot = line[0].split('=')[0]
    value = line[1:]
    return slot, value


def extract_parameter_from_text_file(line):
    return re.sub(' ', '', line).strip()


def update_result(result_dict, slot, value):
    if slot not in result_dict:
        result_dict[slot] = []

    if type(value) == list:
        for v in value:
            result_dict[slot].append(v)
        return result_dict

    elif type(value) == str:
        result_dict[slot].append(value)
        return result_dict

    else:
        raise ValueError


def read_dat_file(project_names, result_dict):
    """
    :param project_names: as we must deal with multi projects, we need a list to store their names.
                          In order to locate them.
    :type project_names: list
    :param result_dict: use a dict to store result
    :type result_dict: dict
    :return: result
    :rtype: dict
    """
    for project_name in project_names:
        with open('./data/slots/{}_slot.map.dat'.format(project_name), encoding='utf-8-sig') as f:
            for line in f.readlines():
                slot, value = extract_parameter_from_dat_file(line)
                result_dict = update_result(result_dict, slot, value)
    return result_dict


def read_text_data(project_names, result_dict):
    for name in project_names:
        file_names = clean_copy_file(get_file_name('./data/slots/SLOT_DB_{}'.format(name)))
        for file_name in tqdm(file_names):
            if name == 'saic' and file_name[6] == '-':
                slot = file_name[6:-4]
            else:
                slot = file_name[5:-4]
            with open('./data/slots/SLOT_DB_{}/{}'.format(name, file_name), encoding='utf-8-sig') as f:
                for line in f.readlines():
                    value = extract_parameter_from_text_file(line)
                    update_result(result_dict, slot, value)
    return result_dict


def dict2excel(result_dict):
    df_dict = {'slot': [], 'value': []}

    for slot in result_dict:
        value = ', '.join(result_dict[slot])

        df_dict['slot'].append(slot)
        df_dict['value'].append(value)
    df = pd.DataFrame(df_dict)
    df.to_excel('./result/slot_value_result.xls')


if __name__ == '__main__':
    res = {}
    project = ['ecarx', 'saic', 'denso', 'jlr']

    read_dat_file(project, res)

    read_text_data(project, res)

    res = de_duplication_sort(res, 20)

    dict2excel(res)
