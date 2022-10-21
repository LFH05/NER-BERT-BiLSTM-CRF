#!/usr/bin/env python3
# coding: utf-8

"""
@File   : predict.py
@Author : FuHuang Liu
@Date   : 2022/9/23
@Desc   : 预测
"""
from utils import *
from model import *
from config import *
from transformers import BertModel, BertTokenizer
import os
import re


def predict():
    train_filename = os.path.join('data', 'fina_data', 'role_train.tsv')
    train_text, train_label = read_data2(train_filename)

    test_filename = os.path.join('data', 'fina_data','test.txt')
    test_text, test_label = read_data(test_filename)
    text = test_text[16]

    print(text)

    input = tokenizer.encode_plus(text, return_tensors='pt')
    inputs = input.input_ids
    inputs = inputs.to(DEVICE)
    mask = input.attention_mask.bool()
    mask = mask.to(DEVICE)
    model = torch.load(MODEL_DIR + 'model_all.pth')
    # y_pre = model(inputs).reshape(-1)  # 或者是y_pre[0]也行,因为y_pre是一个batch，需要进行reshape
    y_pre = model(inputs, mask)[0]

    _, id2label = build_label_2_index(train_label)

    label = [id2label[l] for l in y_pre[1:-1]]

    result = {}
    for i in range(len(text)):
        if label[i] != 'O':
            pre_text = re.findall('[\u4e00-\u9fa5]', label[i])
            pre_text = ''.join(pre_text)
            if pre_text in result.keys():
                result[pre_text] = result[pre_text] + text[i]
            else:
                result[pre_text] = text[i]

    print("原始文本为：{0}".format(text))
    print("预测文本为：{0}".format(result))
    print("原始的标签为：{0}".format(test_label[3]))
    print("预测的标签为：{0}".format(label))



if __name__ == '__main__':
    predict()