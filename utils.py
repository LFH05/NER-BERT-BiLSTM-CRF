#!/usr/bin/env python3
# coding: utf-8

"""
@File   : utils.py
@Author : FuHuang Liu
@Date   : 2022/9/23
@Desc   : 需要的函数/文件
"""
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertConfig, BertTokenizer, BertModel


def read_data2(path):
    all_text = []
    all_label = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data, BIO = line.strip().split('\t')
            data = data.split()
            BIO = BIO.split()
            all_text.append(data)
            all_label.append(BIO)
    return all_text, all_label


def read_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        all_data = f.read().split('\n')

    all_text = []
    all_label = []
    text = []
    labels = []
    for data in all_data:
        if data == ' ':
            all_text.append(text)
            all_label.append(labels)
            text = []
            labels = []
        else:
            # print(data)
            try:
                t, l = data.split(' ')
                text.append(t)
                labels.append(l)
            except:
                print(data)
    return all_text, all_label


def build_label_2_index(all_label):
    label_2_index = {'PAD': 0, 'UNK': 1}
    for labels in all_label:
        for label in labels:
            if label not in label_2_index:
                label_2_index[label] = len(label_2_index)
    return label_2_index, list(label_2_index)


class MyData(Dataset):
    def __init__(self, all_text, all_label, tokenizer, label2index, max_len):
        self.all_text = all_text
        self.all_label = all_label
        self.tokenizer = tokenizer
        self.label2index = label2index
        self.max_len = max_len

    def __getitem__(self, item):
        text = self.all_text[item]
        labels = self.all_label[item][:self.max_len]
        mask = []
        # 需要对text编码，让bert可以接受
        input = self.tokenizer.encode_plus(text,
                                           padding='max_length',
                                           max_length=(self.max_len + 2),
                                           truncation=True,
                                           return_tensors='pt',
                                           add_special_tokens=True,
                                           )
        # text_index = self.tokenizer.encode(text,
        #                                    add_special_tokens=True,
        #                                    padding='max_length',
        #                                    max_length=(self.max_len + 2),
        #                                    truncation=True,
        #                                    return_tensors='pt',
        #                                    )
        # 也需要将label进行编码
        # 那么我们需要构建一个函数来传入label2index
        # labels_index = [self.label2index.get(label, 1) for label in labels]
        # 上面那个就仅仅是转化，我们需要将label和text对齐
        text_index = input.input_ids
        labels_index = [0] + [self.label2index.get(label, 1) for label in labels] + [0] + [0] * (self.max_len - len(labels))
        mask = input.attention_mask
        # print("text_index: {0}".format(text_index))
        # print("mask: {0}".format(mask))
        # print("text的长度：{0}".format(len(text)))
        # print("裁剪后的label长度：{0}".format(len(labels)))
        # print("文本：{0}".format(text_index.squeeze().shape))
        # print("label：{0}".format(torch.tensor(labels_index).shape))
        # print("mask：{0}".format(torch.tensor(mask).shape))
        return text_index.squeeze(), torch.tensor(labels_index), len(text), mask.squeeze().bool()

    def __len__(self):
        return len(self.all_text)


if __name__ == '__main__':
    path = './data/fina_data/role_train.tsv'
    all_text, all_label = read_data2(path)
    # 得到label2index, index2label
    label2index, index2label = build_label_2_index(all_label)
    print()


