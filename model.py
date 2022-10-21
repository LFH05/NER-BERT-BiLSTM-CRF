#!/usr/bin/env python3
# coding: utf-8

"""
@File   : model.py
@Author : FuHuang Liu
@Date   : 2022/9/23
@Desc   : 模型，简单的BERT+BiLSTM+CRF
"""
import torch
import torch.nn as nn
from config import *
from torchcrf import CRF


class MyModel(nn.Module):
    def __init__(self, class_num):
        super(MyModel, self).__init__()
        self.class_num = class_num
        self.bert = BertModel.from_pretrained(BERT_PATH)
        self.lstm = nn.LSTM(768, 768 // 2, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(768, class_num)
        self.crf = CRF(self.class_num, batch_first=True)
        self.loss_fn = nn.CrossEntropyLoss()

    def _get_lstm_feature(self, batch_text):
        output = self.bert(batch_text)  # 通过bert转换为词向量
        # print(batch_text.attention_mask)
        # print("第0个位置{0},  第1个位置{1}".format(output[0],output[1]))
        bert_out0, bert_out1 = output[0], output[1]  # bert输出分别为input_ids、token_type_ids、attention_mask
        output1, _ = self.lstm(bert_out0)  # 将inputs_ids通过BiLSTM模型
        out = self.linear(output1)  # 将BiLSTM模型的输出通过全连接层，得到最终的输出
        return out

    def forward(self, batch_text, mask, batch_label=None):
        pre = self._get_lstm_feature(batch_text)
        # pre = pre.type(torch.FloatTensor)
        return self.crf.decode(pre, mask)

        # if batch_label is not None:
        #     # a.reshape(m,n) 的意思是将a原有的形状转换为m*n形状的矩阵
        #     # -1表示无意义，不规定行数，自行判断
        #     # reshape(-1)表示改成一串，没用行列
        #     # pre.reshape(-1, pre.shape[-1]) == （batch_size*max_len， class_num）
        #     # batch_label.reshape(-1) == （batch_size*max_len)
        #     loss = self.loss_fn(pre.reshape(-1, pre.shape[-1]), batch_label.reshape(-1))
        #     return loss
        # else:
        #     # 当batch_label是空的时候，就是验证
        #     # 直接返回：return torch.argmax(pre, dim=-1)。
        #     return torch.argmax(pre, dim=-1)
        #     # decode = self.crf.decode(pre)
        #     # return decode

    def loss_fn(self, batch_text, batch_label, mask):
        y_pred = self._get_lstm_feature(batch_text)
        return -self.crf.forward(y_pred, batch_label, mask).mean()


