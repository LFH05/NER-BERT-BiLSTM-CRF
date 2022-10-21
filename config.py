#!/usr/bin/env python3
# coding: utf-8

"""
@File   : config.py
@Author : FuHuang Liu
@Date   : 2022/9/23
@Desc   : 基本配置文件
"""
import torch
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader, Dataset

EPOCHS = 100           # 训练次数
BATCH_SIZE = 128        # 每次训练的数据大小
LEARNING_RATE = 2e-5    # 学习率
MAX_LEN = 50            # 文本最大长度
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # 调用GPU

BERT_PATH = r'BERT_MODEL'  # 你自己的bert模型地址

tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
MODEL_DIR = 'model/'  # 这是保存模型的地址，建在你代码的同一级即可
