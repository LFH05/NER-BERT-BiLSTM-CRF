#!/usr/bin/env python3
# coding: utf-8

"""
@File   : Data_Pro.py
@Author : FuHuang Liu
@Date   : 2022/9/13
@Desc   : 处理数据
"""

import json
import numpy
import pandas as pd
from itertools import chain


def read_by_lines(path, t_code="utf-8"):
    """逐行读取数据"""
    result = list()
    with open(path, "r", encoding=t_code) as infile:
        for line in infile:
            result.append(line.strip())
    return result


def write_by_lines(path, data, t_code="utf-8"):
    """write the data"""
    with open(path, "w", encoding=t_code) as outfile:
        [outfile.write(d + "\n") for d in data]


def _event_schema_pro(data_path):
    """
    :param data_path: 事件类型文件的路径
    :return: 返回trigger，role的集合列表
    采用BIO标注方式，对触发词和事件论元进行统计，返回集合的列表
    """
    trigger_label = []
    role_label = []
    index_trigger = 0
    index_role = 0
    for line in read_by_lines(data_path):
        data = json.loads(line)
        event_type = data["event_type"]
        trigger_label.append(u"B-{}".format(event_type))
        trigger_label.append(u"I-{}".format(event_type))
        role_list = data["role_list"]
        for role in role_list:
            if (u"B-{}".format(role["role"])) not in role_label:
                role_label.append(u"B-{}".format(role["role"]))
                role_label.append(u"I-{}".format(role["role"]))
    return trigger_label, role_label


def _trigger_label_dict(trigger_label_list):
    """
    :param trigger_label_list: 触发词列表
    :return: 返回触发词对应编号的字典
    """
    index = 0
    trigger_label_map = {}
    for trigger in trigger_label_list:
        trigger_label_map[trigger] = index
        index += 1
    return trigger_label_map


def _role_label_dict(role_label_list):
    index = 0
    role_label_map = {}
    for role in role_label_list:
        role_label_map[role] = index
        index += 1
    return role_label_map


def _push_label(label, start_index, role_length, _type):
    for i in range(start_index, start_index+role_length):
        tag = u"B-" if i == start_index else u"I-"
        label[i] = u"{}{}".format(tag, _type)
    return label


def _split_data(data_path):
    all_text = []
    all_label = []
    for line in read_by_lines(data_path):
        data = json.loads(line)
        text = data["text"]    # 获取数据集的文本
        label = ["O"] * len(text)
        for event in data["event_list"]:            # 数据集中触发词、论元的列表
            event_type = event["event_type"]        # 获取事件类型
            trigger = event["trigger"]              # 获取事件触发词
            start_trigger = event["trigger_start_index"]    # 获取触发词开始下标
            # print("事件类型为：{0}，触发词为:{1}，触发词的下标为:{2}".format(event_type, trigger, start_trigger))

            for arg in event["arguments"]:
                role = arg["role"]
                argument = arg["argument"]
                start_role = arg['argument_start_index']
                # print("事件论元为:{0}，事件论元参数为:{1}，论元的下标为:{2}".format(role, argument, start_role))
                # 调用函数，将事件论元的标签打上，暂未处理触发词
                label = _push_label(label, start_role, len(argument), role)
        all_text.append(text)
        all_label.append(label)
    return all_text, all_label


def _merge_text_label(all_text, all_label):
    if len(all_text) != len(all_label):
        print("总的文本和标签数量不一样")
    all_data = []
    for i in range(len(all_text)):
        text = all_text[i]
        label = all_label[i]
        for j in range(len(text)):
            if(text[j] != ' '):
                data = '{0} {1}'.format(text[j], label[j])
                all_data.append(data)
        all_data.append(' ')
    return all_data


def _test_data_pro(data_path):
    res = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            text = data["text"]
            id = data["id"]
            res.append(id + '\t' + ' '.join(text))
    return res


def _create_vocab(data_path):
    """
    构建自己的词汇表
    :param data_path:  文件路径
    :return:  构建好的词汇表
    """
    vocab = []
    index = 1
    for line in read_by_lines(data_path):
        data = json.loads(line)
        text = data["text"]
        index += 1
        word = ' '.join(text)
        word = word.split()
        for i in word:
            if i not in vocab:
                vocab.append(i)
    vocab.append('[CLS]')
    vocab.append('[SEP')
    print("共有{0}条".format(index))
    return vocab


if __name__ == "__main__":
    origin_train_data_path = './data/raw_data/duee_sample.json'
    origin_event_schema_data_path = './data/raw_data/duee_event_schema.json'

    # 1、先将event_schema中的事件类型和论元角色，全部提取出来 并保存为txt文件
    trigger_label_list, role_label_list = _event_schema_pro(origin_event_schema_data_path)
    # print(trigger_label_list)
    # print(role_label_list)

    # 2、映射为字典，编号
    trigger_label_map = _trigger_label_dict(trigger_label_list)
    role_label_map = _role_label_dict(role_label_list)
    # print(trigger_label_map)
    # print(role_label_map)

    # 3、按照触发词、事件论元，将文本的标签表示处理出来
    # 数据集处理
    # 将整个example的数据处理成text，label的形式
    all_text, all_label = _split_data(origin_train_data_path)

    # 5、划分数据集
    # 简单的前60%作为训练集、10%作为验证集、30%作为测试集
    train_text = all_text[:240]
    train_label = all_label[:240]
    dev_text = all_text[240:280]
    dev_label = all_label[240:280]
    test_text = all_text[280:]
    test_label = all_label[280:]

    # 5、将text与label结合起来，形成 [word label word label ....]的形式
    train_data = _merge_text_label(train_text, train_label)
    dev_data = _merge_text_label(dev_text, dev_label)
    test_data = _merge_text_label(test_text, test_label)

    # 5、保存数据
    write_by_lines("./data/fina_data/train.txt", train_data)
    write_by_lines("./data/fina_data/dev.txt", dev_data)
    write_by_lines("./data/fina_data/test.txt", test_data)

    # ===========================================================
    # 构建自己的词汇表
    # role_vocab = _create_vocab(origin_train_data_path)
    # write_by_lines("../bert_base_chinese/vocab.txt", role_vocab)
    # print(len(role_vocab))

    # ===========================================================
    # 用来筛选train和dev中存在的某些不合格数据
    # data_1 = []
    # index = 0
    # with open('../data/fina_data/role_train.tsv', 'r', encoding='utf-8') as f:
    #     for line in f:
    #         try:
    #             data, BIO = line.strip().split('\t')
    #             data_1.append(line)
    #         except:
    #             index+=1
    #             print(line)
    #
    # print(index)
    # write_by_lines("../data/fina_data/role_train.tsv", data_1)

    pass
